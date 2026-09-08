//! The CLI. Scan logic lives in the library; this file parses arguments and decides
//! where output goes.

use clap::{Parser, Subcommand};
use keyforge::vuln;

#[derive(Parser)]
#[command(name = "keyforge", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List the vulnerabilities that can be scanned, or explain one in detail.
    Vulns {
        /// A vulnerability id or alias. Omit to list them all.
        name: Option<String>,
    },
}

fn main() -> std::process::ExitCode {
    match Cli::parse().command {
        Command::Vulns { name: None } => {
            list_vulnerabilities();
            std::process::ExitCode::SUCCESS
        }
        Command::Vulns { name: Some(name) } => match vuln::find(&name) {
            Some(v) => {
                print!("{}", vuln::render_guide(v));
                std::process::ExitCode::SUCCESS
            }
            None => {
                eprintln!(
                    "unknown vulnerability `{name}`\n\navailable: {}",
                    vuln::names().join(", ")
                );
                std::process::ExitCode::FAILURE
            }
        },
    }
}

/// One line per vulnerability: the name to type, what it costs, and what it is.
///
/// The space size is shown because it is the number that decides whether a sweep is an
/// afternoon or a fortnight, and it is the first thing worth knowing.
fn list_vulnerabilities() {
    let width = vuln::registry().iter().map(|v| v.id().len()).max().unwrap_or(0);
    for v in vuln::registry() {
        let space = match v.space().len() {
            Some(n) if n >= 1 << 20 => format!("2^{:.0}", (n as f64).log2()),
            Some(n) => n.to_string(),
            None => "corpus".to_string(),
        };
        println!("{:<width$}  {space:>7}  {}", v.id(), summarise(v.guide().what, 62));
    }
    println!("\nkeyforge vulns <name>   for the full guide");
}

/// The opening of a guide's `what`, trimmed to fit one terminal line.
///
/// A listing is an index rather than documentation: it has to be scannable down the
/// left edge, so every row gets the same budget and the guide holds the rest. Cuts on a
/// word boundary, because a name sliced in half reads as a different name.
fn summarise(text: &str, budget: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.len() <= budget {
        return flat;
    }
    // Walk characters rather than bytes. Guide text is prose, and a single em dash
    // straddling the budget would make byte slicing panic the listing instead of
    // shortening a line. `char_indices` also gives the last space for free.
    let mut cut = 0;
    let mut last_space = None;
    for (i, c) in flat.char_indices() {
        if i >= budget {
            break;
        }
        cut = i + c.len_utf8();
        if c == ' ' {
            last_space = Some(i);
        }
    }
    let cut = last_space.unwrap_or(cut);
    format!("{}...", flat[..cut].trim_end_matches([',', ';', ':', '-']).trim_end())
}
