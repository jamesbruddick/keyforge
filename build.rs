//! Nothing is generated here, and nothing is checked here any more.
//!
//! This once stopped a `--features cuda,cuda12` build, which asked for two CUDA toolkit
//! versions at once and failed a few hundred duplicate-definition errors deep inside
//! `cudarc` with nothing mentioning features. There is one CUDA feature now, so the
//! combination cannot be spelled. What remains is the two rebuild triggers, which cargo
//! does not infer on its own.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // `src/ec.rs` reads this through `option_env!`, so cargo has to be told that changing
    // it invalidates the build -- otherwise a sweep silently re-measures the same binary.
    println!("cargo:rerun-if-env-changed=KEYFORGE_EC_WINDOW");
}
