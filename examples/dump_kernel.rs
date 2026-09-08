//! Print the assembled kernel translation unit, for reading or for feeding to a compiler
//! by hand.
//!
//! Which vulnerability matters: every plugin contributes its own `vuln_expand`, and the
//! scope it narrows to decides which kernels survive the preprocessor. Dumping one and
//! debugging another is a way to spend an afternoon.
//!
//! ```text
//! cargo run --example dump_kernel                    # milksad, Metal
//! cargo run --example dump_kernel -- glibc-rand      # one plugin, at its own defaults
//! cargo run --example dump_kernel -- glibc-rand cuda
//! ```

use keyforge::gpu::source::{Dialect, assemble};
use keyforge::gpu::Layout;
use keyforge::scan::derive::Scope;

fn main() {
    let mut args = std::env::args().skip(1);
    let id = args.next().unwrap_or_else(|| "milksad".into());
    let dialect = match args.next().as_deref() {
        None | Some("metal") => Dialect::Metal,
        Some("cuda") => Dialect::Cuda,
        Some(other) => {
            eprintln!("unknown dialect `{other}`: expected `metal` or `cuda`");
            std::process::exit(2);
        }
    };

    let Some(vuln) = keyforge::vuln::find(&id) else {
        eprintln!("unknown vulnerability `{id}`. Try `keyforge vulns`.");
        std::process::exit(2);
    };
    if vuln.kernel().is_none() {
        eprintln!("`{}` has no GPU kernel, so there is nothing to assemble.", vuln.id());
        std::process::exit(2);
    }

    // The plugin's own defaults, not a bare `Scope::default()`: a dump that does not match
    // what a scan of this vulnerability compiles is a dump of the wrong kernel set.
    let mut scope = Scope::default();
    vuln.defaults().apply(&mut scope);
    print!("{}", assemble(&Layout::new(&scope, 1), vuln, dialect));
}
