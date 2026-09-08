//! Prints the README's per-vulnerability sections from each plugin's own `Guide`, so
//! the documentation is generated from the code rather than kept in step by hand.
//!
//!     cargo run --example dump_guides
fn main() {
    for v in keyforge::vuln::registry() {
        let g = v.guide();
        let title = match v.cve() {
            Some(cve) => format!(" — {cve}"),
            None => String::new(),
        };
        println!("### `{}`{}\n", v.id(), title);
        println!("{}\n", one_line(g.what));
        println!("**Affected:** {}\n", one_line(g.affected));
        println!(
            "**Runs on:** {}\n",
            match v.kernel().is_some() {
                true => "CPU, or GPU via CUDA and Metal.",
                false => "CPU only -- this one has no device kernel.",
            }
        );
        println!("```\n{}\n```\n", g.command);
        println!("**Time:** {}\n", one_line(g.time));
        println!("**A hit looks like:** {}\n", one_line(g.hit));
    }
}

fn one_line(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
