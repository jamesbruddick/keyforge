//! Nothing is generated here. This exists for one check that has to happen before
//! `cudarc` is compiled, because after that it is too late to say anything useful.
//!
//! `--features cuda,cuda12` asks for two CUDA toolkit versions at once. `cudarc` gates its
//! bindings per version, so two versions define every type twice and the build stops with
//! a few hundred duplicate-definition errors inside a dependency, none of which mention
//! features. A `compile_error!` in `gpu::cuda` cannot catch it -- this crate never gets
//! compiled -- but a build script runs before the dependency does.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // `src/ec.rs` reads this through `option_env!`, so cargo has to be told that changing
    // it invalidates the build -- otherwise a sweep silently re-measures the same binary.
    println!("cargo:rerun-if-env-changed=KEYFORGE_EC_WINDOW");

    // Features reach a build script as environment variables, not as `cfg`.
    let thirteen = std::env::var_os("CARGO_FEATURE_CUDA").is_some();
    let twelve = std::env::var_os("CARGO_FEATURE_CUDA12").is_some();
    if thirteen && twelve {
        panic!(
            "\n\nfeatures `cuda` (CUDA 13.x) and `cuda12` (CUDA 12.x) select different \
             CUDA toolkit versions and cannot both be enabled.\nPick the major version \
             `nvidia-smi` reports:\n    \
             cargo build --release --features cuda\n    \
             cargo build --release --features cuda12\n\n"
        );
    }
}
