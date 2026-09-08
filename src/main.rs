//! The CLI. Scan logic lives in the library; this file parses arguments, decides where
//! output goes, and prints the banner.

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use keyforge::scan::derive::{Route, Scope};
use keyforge::scan::engine::{self, ScanConfig};
use keyforge::scan::state::State;
use keyforge::target::{Target, bloom::BloomFilter};
use keyforge::ui::{self, Stream, Ui};
use keyforge::vuln::{self, Space, Vulnerability};
use keyforge::wallet::address::HashForm;
use keyforge::wallet::path::PathSpec;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "keyforge", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Sweep a vulnerability's search space against a bloom filter of funded addresses.
    Scan(ScanArgs),
    /// List the vulnerabilities that can be scanned, or explain one in detail.
    Vulns {
        /// A vulnerability id or alias. Omit to list them all.
        name: Option<String>,
    },
}

/// What to derive for each point.
///
/// Flattened into every subcommand that walks a tree, so a `verify` run can be told
/// exactly what a `scan` searched. `Option<Vec<_>>` on the vulnerability-dependent ones
/// is how "unset" is told apart from "explicitly set to what the default happens to be".
#[derive(Args, Clone)]
#[command(next_help_heading = "Derivation scope")]
struct ScopeArgs {
    /// Which vulnerability to sweep. See `keyforge vulns`.
    #[arg(long, value_name = "ID")]
    vuln: String,

    /// Prefix lengths of the secret material to try, in bytes (16, 20, 24, 28, 32).
    #[arg(long, value_name = "N", value_delimiter = ',')]
    material: Option<Vec<usize>>,

    /// How the material became keys: bip39, bip32-seed, privkey.
    #[arg(long, value_name = "ROUTE", value_delimiter = ',')]
    routes: Option<Vec<String>>,

    /// Derivation paths, repeatable. E.g. "m/44'/0'/0'/{0,1}/{0..9}".
    #[arg(long = "path", value_name = "PATH")]
    paths: Vec<String>,

    /// Which hash160 forms to derive: compressed, uncompressed, p2sh-p2wpkh.
    #[arg(long, value_name = "FORM", value_delimiter = ',')]
    hash_forms: Option<Vec<String>>,
}

impl ScopeArgs {
    fn vulnerability(&self) -> Result<&'static dyn Vulnerability> {
        vuln::find(&self.vuln).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown vulnerability `{}`\n\navailable: {}",
                self.vuln,
                vuln::names().join(", ")
            )
        })
    }

    /// Build the scope: start from the vulnerability's defaults, then apply overrides.
    ///
    /// All validation lives here, so a scan that is going to be refused is refused
    /// before the filter is read rather than three seconds later.
    fn to_scope(&self, v: &dyn Vulnerability) -> Result<Scope> {
        let mut scope = Scope::default();
        v.defaults().apply(&mut scope);

        if let Some(sizes) = &self.material {
            for size in sizes {
                if !keyforge::wallet::bip39::VALID_ENTROPY_SIZES.contains(size) {
                    bail!(
                        "material size {size} is not a BIP39 entropy size; \
                         expected one of 16, 20, 24, 28, 32"
                    );
                }
            }
            scope.material_sizes = sizes.clone();
        }
        if let Some(routes) = &self.routes {
            scope.routes = routes
                .iter()
                .map(|r| {
                    Route::parse(r).ok_or_else(|| {
                        anyhow::anyhow!("unknown route `{r}`; expected bip39, bip32-seed or privkey")
                    })
                })
                .collect::<Result<_>>()?;
        }
        if !self.paths.is_empty() {
            scope.paths = self
                .paths
                .iter()
                .map(|p| PathSpec::parse(p).map_err(|e| anyhow::anyhow!("{e}")))
                .collect::<Result<_>>()?;
        }
        if let Some(forms) = &self.hash_forms {
            scope.forms = forms
                .iter()
                .map(|f| {
                    HashForm::parse(f).ok_or_else(|| {
                        anyhow::anyhow!(
                            "unknown hash form `{f}`; expected compressed, uncompressed \
                             or p2sh-p2wpkh"
                        )
                    })
                })
                .collect::<Result<_>>()?;
        }

        scope.validate().map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(scope)
    }

    /// A stable string describing everything that decides what this scan walks.
    ///
    /// Resuming with any of it changed is refused, because the resulting file would
    /// describe a complete sweep of neither configuration.
    fn fingerprint_input(&self, scope: &Scope) -> Vec<String> {
        let mut parts = vec![self.vuln.clone()];
        parts.push(
            scope.material_sizes.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(","),
        );
        parts.push(scope.routes.iter().map(|r| r.as_str()).collect::<Vec<_>>().join(","));
        parts.push(scope.paths.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(" "));
        parts.push(scope.forms.iter().map(|f| f.as_str()).collect::<Vec<_>>().join(","));
        parts
    }
}

#[derive(Args)]
struct ScanArgs {
    #[command(flatten)]
    scope: ScopeArgs,

    /// Bloom filter of funded addresses, in keyscan `.bf` format.
    #[arg(short, long, value_name = "PATH")]
    filter: PathBuf,

    /// Where candidate secrets are written, one per line.
    #[arg(short, long, value_name = "PATH", default_value = "matches.txt")]
    out: PathBuf,

    /// Also write full context for each hit as JSON, one object per line.
    #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "matches.jsonl")]
    details: Option<PathBuf>,

    /// First point of the range. Defaults to the start of the vulnerability's space.
    #[arg(long, value_name = "N")]
    start: Option<u128>,

    /// One past the last point. Defaults to the end of the vulnerability's space.
    #[arg(long, value_name = "N")]
    end: Option<u128>,

    /// Worker threads. Defaults to the number of cores.
    #[arg(short, long, value_name = "N")]
    threads: Option<usize>,

    /// Points per work block. Defaults to a value derived from the range.
    #[arg(long, value_name = "N")]
    block: Option<u64>,

    /// Ignore any existing checkpoint and start from the beginning.
    #[arg(long)]
    restart: bool,
}

fn main() -> std::process::ExitCode {
    let command = Cli::parse().command;
    // Scan writes its real output to a file, so its terminal chatter goes to stderr and
    // the file is the thing you redirect. `vulns` is the opposite: its output *is* the
    // answer, so it goes to stdout and pipes.
    let ui = Ui::new(match command {
        Command::Scan(_) => Stream::Stderr,
        Command::Vulns { .. } => Stream::Stdout,
    });

    let result = match command {
        Command::Scan(args) => run_scan(&ui, args),
        Command::Vulns { name } => run_vulns(&ui, name),
    };

    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            ui.error(&format!("{error}"));
            for cause in error.chain().skip(1) {
                ui.error_cause(&format!("{cause}"));
            }
            std::process::ExitCode::FAILURE
        }
    }
}

fn run_scan(ui: &Ui, args: ScanArgs) -> Result<()> {
    let v = args.scope.vulnerability()?;
    let scope = args.scope.to_scope(v)?;

    // A corpus vulnerability walks a file, not a number range, and needs a reader and a
    // different checkpoint. Saying so plainly beats scanning an empty range and
    // reporting a clean pass.
    let (space_start, space_end) = match v.space() {
        Space::Integers { start, end } => (start, end),
        Space::Corpus => bail!(
            "`{}` walks a passphrase corpus rather than a number range, and corpus \
             scanning is not wired up yet.\nEvery other vulnerability in `keyforge \
             vulns` is an integer space and can be scanned now.",
            v.id()
        ),
    };

    let start = args.start.unwrap_or(space_start);
    let end = args.end.unwrap_or(space_end);
    if start >= end {
        bail!("--start {start} is not below --end {end}");
    }
    if start < space_start || end > space_end {
        bail!(
            "range {start}..{end} falls outside `{}`'s space of {space_start}..{space_end}",
            v.id()
        );
    }

    let threads = args.threads.unwrap_or_else(|| {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    });
    if threads == 0 {
        bail!("--threads must be at least 1");
    }

    ui.title(env!("CARGO_PKG_VERSION"));
    for line in v.describe() {
        ui.cont_plain(&line);
    }
    ui.gap();

    // Narrowing a range is only sound for a vulnerability whose points have structure a
    // user can reason about. Everywhere else it leaves a hole, and saying so is the
    // whole reason `range_is_narrowable` exists.
    let narrowed = start != space_start || end != space_end;
    if narrowed && !v.range_is_narrowable() {
        ui.warn(&format!(
            "`{}`'s points have no time structure, so a narrowed range is a hole \
             rather than a saving: the wallets outside it are just as likely.",
            v.id()
        ));
    }

    let target = load_filter(ui, &args.filter, &scope)?;

    let total = end - start;
    ui.row("vulnerability", v.id());
    if let Some(cve) = v.cve() {
        ui.row("cve", cve);
    }
    ui.row("range", &format!("{start} .. {end}  ({} points)", ui::commas_u128(total)));
    ui.row("material", &fmt(&scope.material_sizes, |s| format!("{s}B")));
    ui.row("routes", &fmt(&scope.routes, |r| r.as_str().to_string()));
    for (i, path) in scope.paths.iter().enumerate() {
        ui.row(if i == 0 { "paths" } else { "" }, &path.to_string());
    }
    ui.row("hash forms", &fmt(&scope.forms, |f| f.as_str().to_string()));
    ui.row(
        "per point",
        &format!(
            "{} probes · {} keys · {} pbkdf2",
            ui::commas(scope.probes_per_point()),
            ui::commas(scope.ec_ops_per_point()),
            ui::commas(scope.pbkdf2_per_point())
        ),
    );
    ui.row("threads", &threads.to_string());
    ui.row("output", &args.out.display().to_string());
    ui.gap();

    let state_path = args.filter.with_extension("state");
    let fingerprint_parts = {
        let mut parts = args.scope.fingerprint_input(&scope);
        parts.push(args.filter.display().to_string());
        parts.push(start.to_string());
        parts.push(end.to_string());
        parts
    };
    let refs: Vec<&str> = fingerprint_parts.iter().map(|s| s.as_str()).collect();
    let fingerprint = State::fingerprint(&refs);

    let config = ScanConfig {
        filter: args.filter.clone(),
        out: args.out,
        details: args.details,
        start,
        end,
        threads,
        block: args.block,
        restart: args.restart,
    };

    let report = engine::run(ui, v, &scope, &target, &config, &fingerprint, &state_path)?;

    ui.gap();
    let rate = report.points_done as f64 / report.elapsed.as_secs_f64().max(1e-9);
    ui.row_strong(
        if report.finished { "finished" } else { "stopped" },
        &format!(
            "{} points in {} ({:.0}/s)",
            ui::commas(report.points_done),
            ui::duration(report.elapsed.as_secs_f64()),
            rate
        ),
    );
    ui.row(
        "candidates",
        &format!(
            "{} secrets from {} matching positions",
            ui::commas(report.candidates),
            ui::commas(report.locations)
        ),
    );
    if !report.finished {
        ui.cont("re-run the same command to pick up where this stopped");
    }
    if report.candidates > 0 {
        ui.cont("these are candidates, not confirmed funds: check them on chain");
    }
    Ok(())
}

/// Open the filter, and say plainly if it cannot answer for part of the scope.
fn load_filter(ui: &Ui, path: &std::path::Path, scope: &Scope) -> Result<Target> {
    let primary = BloomFilter::open(path)
        .with_context(|| format!("opening the filter {}", path.display()))?;
    let verify = Target::open_verify(path)?;
    let target = Target::new(primary, verify);

    // A filter built without, say, P2SH entries will never match a form the scan spends
    // a third of its time deriving. Silence here is what turns that into a clean pass
    // that proves less than it appears to.
    let unmatchable = target.unmatchable_forms(&scope.forms);
    if !unmatchable.is_empty() {
        ui.warn(&format!(
            "this filter holds no entries for {}, so those forms can never match; \
             drop them with --hash-forms to stop deriving them",
            fmt(&unmatchable, |f| f.as_str().to_string())
        ));
    }
    Ok(target)
}

fn fmt<T>(items: &[T], show: impl Fn(&T) -> String) -> String {
    items.iter().map(show).collect::<Vec<_>>().join(", ")
}

fn run_vulns(ui: &Ui, name: Option<String>) -> Result<()> {
    let _ = ui;
    match name {
        None => {
            list_vulnerabilities();
            Ok(())
        }
        Some(name) => match vuln::find(&name) {
            Some(v) => {
                print!("{}", vuln::render_guide(v));
                Ok(())
            }
            None => bail!(
                "unknown vulnerability `{name}`\n\navailable: {}",
                vuln::names().join(", ")
            ),
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
