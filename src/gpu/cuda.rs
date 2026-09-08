//! The CUDA backend.
//!
//! The same kernels as the Metal backend, compiled through NVRTC instead of
//! `newLibraryWithSource`. If the discipline in `kernels/compat.h` held -- pointer-free
//! arithmetic, address spaces behind macros -- then this file plus that header is the
//! whole of the port, and no kernel source changes at all.
//!
//! Two deliberate choices, both about running on a box that was rented five minutes ago:
//!
//! - **NVRTC, not nvcc.** The kernels compile in-process at startup, so there is no
//!   build-time compilation step and no shipped PTX, and the scope can be baked in as
//!   constants. Same reasoning as the Metal side, where `xcrun metal` is absent on a
//!   Command Line Tools install.
//! - **`dynamic-loading`**, so `cudarc` dlopens its libraries at run time rather than
//!   linking them. The binary therefore *builds* anywhere -- `cargo build --features cuda`
//!   on a Mac is a sensible thing to do -- and resolves what it needs when it runs.
//!
//! **Running it needs the CUDA toolkit, not just the driver.** An earlier version of this
//! comment claimed otherwise and it is wrong: the driver installs `libcuda`, but NVRTC is
//! `libnvrtc`, which comes with the toolkit. `open` reports an absent one as an absent
//! device, so a machine without a toolkit skips the GPU tests instead of failing them.
//!
//! **Which toolkit is a run-time question.** It used to be a build-time one -- `--features
//! cuda` meant CUDA 13.x and `--features cuda12` meant 12.x, because `cudarc` derives the
//! library filenames it searches for from the version it was compiled against. That put a
//! decision about the machine the binary runs on into the build, and it got the important
//! case backwards: CUDA 13 dropped Maxwell, Pascal and Volta, so on a GTX 1080 Ti the newer
//! toolkit is the one that cannot be used. NVRTC is therefore loaded and called directly
//! here -- see `nvrtc`, which picks the newest installed compiler that can emit code for
//! this card. The feature now selects only `cudarc`'s driver bindings, which resolve one
//! entry point at a time and are not version-sensitive in the way the compiler is.
//!
//! **The filter is copied here, not shared.** A discrete card has its own memory and
//! `buffer_no_copy` cannot mean what it means on unified memory, so it allocates and
//! uploads -- about half a second for 7.6 GB over PCIe 4 x16, once. The alternative,
//! mapping host memory and letting the device read it across the bus, is not offered on
//! purpose: the scan issues roughly 2,300 random 8-byte probes per seed, and at an
//! optimistic 10 M random reads/s that caps the whole sweep near 4,300 seeds/s -- below
//! what a laptop CPU already does. **The filter has to fit in VRAM**, and if it does not
//! the answer is a smaller filter from `keyscan bf-gen` at a higher false-positive rate,
//! not a slower path.

use anyhow::{Result, anyhow, bail};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::collections::HashMap;
use std::sync::Arc;

use super::{Arg, Backend, BufferId};

mod nvrtc;

/// Threads per block. 256 matches the Metal side and is a reasonable default across
/// every generation this is likely to meet; the kernels bounds-check `gid`, so a partial
/// final block is safe.
///
/// **It has never been swept on a discrete card.** It was picked for an integrated GPU and
/// then applied flat to all fourteen dispatches, which do not have remotely the same
/// register footprint: `ec_mul_gen` carries a Jacobian accumulator, a table point and nine
/// live field temporaries inside `gej_add_ge`, where `k_entropy` carries almost nothing.
/// `MILKSAD_CUDA_BLOCK` overrides it -- see `Tuning`.
const BLOCK: u32 = 256;

/// Compile- and launch-time knobs, read from the environment once.
///
/// These exist because none of them has a defensible default on a card nobody has measured
/// them on, and all three are the first things anyone tunes on CUDA. Reading them from the
/// environment rather than baking them in makes a sweep a shell loop instead of fourteen
/// rebuilds:
///
/// ```text
///   MILKSAD_CUDA_BLOCK       threads per block, default 256
///   MILKSAD_CUDA_MAXREG      --maxrregcount=N, unset by default
///   MILKSAD_KERNEL_DEFINES   extra NVRTC tokens, space separated
/// ```
///
/// ```sh
/// for r in 0 64 96 128 160; do
///     MILKSAD_CUDA_MAXREG=$r ./target/release/milksad-scan bench \
///         -f addresses.bf --gpu only --seeds 3000000
/// done
/// ```
///
/// Anything that changes what NVRTC produces is part of the PTX cache key, so switching a
/// knob cannot hand back the previous build -- see `compile_cached`. `MILKSAD_CUDA_BLOCK`
/// is not, because it changes the launch and not the translation unit.
///
/// A bad value is a panic rather than a fallback. Silently measuring the default while the
/// operator believes they are measuring something else is the exact failure the README
/// warns about when it says to take medians and to distrust a single run.
///
/// Once a value has been measured on real hardware it belongs in `BLOCK` above, or in a
/// `__launch_bounds__` in the kernel, with the table that chose it. The variable is for
/// finding the answer, not for holding it.
///
/// **`MILKSAD_CUDA_MAXREG` has now been swept on an RTX 5070 Ti, and the answer is that
/// occupancy is not the constraint.** Default scope, 3,000,000 seeds against the real
/// filter, medians of three:
///
/// ```text
///   cap    seeds/s
///   none   107,054
///   64     106,658
///   96     106,610
///   128    106,813
///   160    107,093
/// ```
///
/// 0.4% across the whole range, inside the sweep's own 0.8% noise. That is a stronger
/// result than it looks. Uncapped, `k_kmul` takes 121 registers and `k_pbkdf2` 160, which
/// on a 64K-register SM is roughly a third and a quarter occupancy; capping at 64 more than
/// doubles the resident warps and must spill heavily to get there. Doing both and moving
/// nothing says the kernels are neither occupancy-starved nor especially hurt by spilling
/// -- so there is no `__launch_bounds__` worth adding, and the register allocator's own
/// choice is left alone.
///
/// **Swept again after the PTX field arithmetic went in, and the answer did not move: no
/// cap still wins.** That is worth more than the first result. The PTX rewrite changed the
/// register pressure of every kernel that does curve work, so this is a second, independent
/// look at a different program -- and ptxas's own allocation beat every cap both times.
///
/// Two things follow. There is no `__launch_bounds__` worth adding, and the case for
/// trading occupancy the *other* way is weaker than it looks: the obvious remaining idea for
/// `k_pbkdf2` is to interleave two PBKDF2 chains per thread, which buys instruction-level
/// parallelism by spending registers and halving the resident warps. The compiler is already
/// sitting at a balance that resists being pushed either direction, which is not proof that
/// the trade fails but is the reason to measure it rather than assume it.
///
/// The knob stays because it is the cheap way to re-establish that on a different card, and
/// because the profile's `regs` and `local` columns are only interpretable next to it.
///
/// ## And on a card old enough, occupancy *is* the constraint
///
/// Everything above was measured on Blackwell. A GTX 1080 Ti -- Pascal, sm_61 -- ran the
/// same scan at **204 seeds/s**, against the 10,000 or so its throughput suggests even
/// after Pascal's penalties, and moving one of these two knobs to 128 was the fastest thing
/// the operator tried. The occupancy arithmetic says why, and says both knobs point the
/// same way:
///
/// ```text
///   65,536 registers per SM, k_pbkdf2 at 160 registers
///   block 256, no cap    1 block resident     256 of 2,048 threads    4 warps
///   block 128, no cap    3 blocks resident    384 threads             6 warps
///   block 256, cap 128   2 blocks resident    512 threads             8 warps
///   block 128, cap 128   4 blocks resident    512 threads             8 warps
/// ```
///
/// Four warps is nothing to hide 784 bytes of per-thread local memory behind, and the two
/// architectures are not in the same situation despite the identical register file: Turing
/// and later have native 32-bit multiply, several times the L1, and an L2 that swallows the
/// whole comb table. Pascal has none of that, so the same allocation that costs Blackwell
/// nothing leaves this card waiting.
///
/// So **sm_6x and older default to a 128-thread block and `--maxrregcount=128`**, and
/// Turing and newer keep the 256 and the uncapped allocator that were measured to be right
/// for them. The two are picked apart by compute capability because that is what the
/// difference actually is.
///
/// **This is a considered guess, not a measurement.** It is one figure from one operator on
/// one card, chosen because the arithmetic above says the two knobs compose and 128/128 is
/// where they land. The sweep that settles it is the loop at the top of this comment, and
/// if it comes back with a different pair the fix is this table -- not a flag, and not a
/// note in the README telling people which environment variable to set.
struct Tuning {
    block: u32,
    /// The `--maxrregcount` in force, kept beside the options it is already inside so the
    /// banner can name it without parsing them back out.
    maxreg: Option<u32>,
    /// Everything passed to NVRTC, including the fixed options.
    options: Vec<String>,
}

/// Threads per block and register cap for a compute capability, before the environment gets
/// a say. See the table above: `0` is "no cap", which is what an uncapped allocator is
/// spelled as in `MILKSAD_CUDA_MAXREG` too.
const fn defaults_for(major: i32) -> (u32, u32) {
    match major {
        ..=6 => (128, 128),
        _ => (BLOCK, 0),
    }
}

impl Tuning {
    /// The tuning for this card, with the environment overriding both halves of it.
    ///
    /// Per device rather than a global, because the defaults are: a process that opened two
    /// cards of different generations would otherwise hand the second one the first one's
    /// answer.
    fn for_capability(major: i32) -> Self {
        let (default_block, default_maxreg) = defaults_for(major);

        let block = match std::env::var("MILKSAD_CUDA_BLOCK") {
            Ok(v) => {
                let n: u32 = v.trim().parse().unwrap_or_else(|_| {
                    panic!("MILKSAD_CUDA_BLOCK must be a number, got `{v}`")
                });
                assert!(
                    n > 0 && n <= 1024 && n % 32 == 0,
                    "MILKSAD_CUDA_BLOCK must be a multiple of 32 between 32 and 1024, got {n}"
                );
                n
            }
            Err(_) => default_block,
        };

        // Trading registers for occupancy. `0` means "no cap", so a sweep can include the
        // unset case without special-casing the loop in the shell -- and so an operator on
        // a Pascal card can ask for the uncapped allocator back the same way.
        let maxreg = match std::env::var("MILKSAD_CUDA_MAXREG") {
            Ok(v) => {
                let n: u32 = v
                    .trim()
                    .parse()
                    .unwrap_or_else(|_| panic!("MILKSAD_CUDA_MAXREG must be a number, got `{v}`"));
                assert!(
                    n == 0 || (16..=255).contains(&n),
                    "MILKSAD_CUDA_MAXREG must be 0 (no cap) or between 16 and 255, got {n}"
                );
                n
            }
            Err(_) => default_maxreg,
        };
        let maxreg = (maxreg != 0).then_some(maxreg);

        // Nothing here is floating point -- the whole program is 32-bit integer
        // multiply-accumulate -- so the fast-math family has nothing to act on and is left
        // alone rather than set for the look of it.
        let mut options = vec!["--extra-device-vectorization".to_string()];
        if let Some(n) = maxreg {
            options.push(format!("--maxrregcount={n}"));
        }

        // The general escape hatch, so an A/B inside the kernels is a `-D` rather than an
        // edit. `MILKSAD_PORTABLE_ROTR64` in kernels/sha512.h is the one this was added for.
        if let Ok(extra) = std::env::var("MILKSAD_KERNEL_DEFINES") {
            options.extend(extra.split_whitespace().map(str::to_string));
        }

        Self {
            block,
            maxreg,
            options,
        }
    }

    /// What to add to the banner when this card was given something other than the values
    /// the kernels were written against.
    ///
    /// A default that varies by device is still a decision made for the operator, and the
    /// whole reason it exists is that one card wanted different numbers -- so the card that
    /// gets different numbers says so, on the line that already names its compiler.
    fn describe(&self) -> String {
        let mut parts = Vec::new();
        if self.block != BLOCK {
            parts.push(format!("{} threads per block", self.block));
        }
        if let Some(cap) = self.maxreg {
            parts.push(format!("{cap}-register cap"));
        }
        match parts.is_empty() {
            true => String::new(),
            false => format!(", {}", parts.join(", ")),
        }
    }
}

pub struct Cuda {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// The NVRTC that can compile for this card, and the architecture it will be asked
    /// for. Chosen in `open`, because both halves of that are properties of the device.
    compiler: nvrtc::Compiler,
    /// Launch and compile settings for this card. A property of the device for the same
    /// reason: what a Pascal card wants is not what Blackwell was measured to want.
    tuning: Tuning,
    module: Option<Arc<cudarc::driver::CudaModule>>,
    /// Functions are looked up once and kept; the launch loop asks for the same handful
    /// thousands of times.
    functions: HashMap<String, CudaFunction>,
    buffers: Vec<CudaSlice<u8>>,
    /// Wall time and call count per kernel, kept only when MILKSAD_GPU_PROFILE is set.
    ///
    /// **This did not exist, and its absence is why every figure in the README's kernel
    /// tables was taken on an M1 Pro.** The Metal backend has had one from the start; the
    /// CUDA backend accepted the same environment variable and silently reported nothing,
    /// so the one card that does 35x the work of this project's CPU was the one that could
    /// not say where its time went.
    ///
    /// Unlike Metal, a CUDA dispatch is asynchronous, so a clock around the launch measures
    /// the enqueue and not the kernel. Profiling therefore synchronises the stream after
    /// every dispatch. That is honest about what it costs: the **total** it prints is a
    /// little worse than an unprofiled launch, because the dispatches no longer overlap at
    /// all, while the **shares** -- which is what the number is for -- are what to read.
    /// Free when the variable is unset, which is the case a sweep runs in.
    profile: Option<HashMap<String, (std::time::Duration, u64)>>,
}

/// Compile to PTX, reusing an earlier result for identical source.
///
/// NVRTC is slow on this kernel set -- the field arithmetic alone is thousands of unrolled
/// multiply-accumulates -- and the source depends only on the scope, which is fixed for a
/// run. A scan therefore compiles once and gains nothing here. The test suite is the
/// opposite: it opens a device per test, serialised behind one lock, and without a cache
/// every one of them pays the full compile again -- `narrowed_scopes_match_the_cpu_walk`
/// opens five by itself.
///
/// Keyed on the source, **the compiler that produced it, the architecture it was compiled
/// for, and the options it was compiled with**, so a changed kernel is a different key and
/// no stale entry can be picked up. The architecture is part of the key because it is part
/// of the answer: PTX for `compute_61` is not PTX for `compute_120`, and a cache under
/// `~/.cache` outlives the card it was filled for -- a shared home directory, a machine
/// whose GPU was swapped, or a box with two different cards in it would otherwise hand back
/// a translation unit built for the wrong one. The NVRTC version is in the key for the same
/// reason one step out: which compiler runs is now decided at run time from what is
/// installed, so installing a toolkit changes the output without changing anything else the
/// key can see. The options are in the key for a sharper reason still -- they are what a
/// sweep varies, and a cache that ignored them would return the previous build and report
/// it as a result. Cache failures are ignored on purpose: a read-only or absent cache
/// directory should make this slower, never broken.
fn compile_cached(
    compiler: &nvrtc::Compiler,
    tuning: &Tuning,
    source: &str,
) -> Result<cudarc::nvrtc::Ptx> {
    use sha2::Digest;
    let (major, minor) = compiler.version();
    let mut hasher = sha2::Sha256::new();
    hasher.update(source.as_bytes());
    hasher.update(b"\0");
    hasher.update(compiler.arch().as_bytes());
    hasher.update(b"\0");
    hasher.update(format!("nvrtc{major}.{minor}").as_bytes());
    for opt in &tuning.options {
        hasher.update(b"\0");
        hasher.update(opt.as_bytes());
    }
    let key: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
    // `LOCALAPPDATA` is what makes this work on Windows at all: `HOME` is a unix
    // convention that PowerShell does not set, so without it every run there recompiled
    // from scratch and the cache silently did nothing.
    let dir = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))
        .or_else(|| std::env::var_os("LOCALAPPDATA").map(std::path::PathBuf::from))
        .map(|c| c.join("milksad-scan"));
    let path = dir.as_ref().map(|d| d.join(format!("{key}.ptx")));

    if let Some(Ok(text)) = path.as_ref().map(std::fs::read_to_string) {
        return Ok(cudarc::nvrtc::Ptx::from_src(text));
    }

    let ptx = compiler.compile(source, &tuning.options)?;

    if let (Some(dir), Some(path)) = (dir.as_ref(), path.as_ref()) {
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(path, &ptx);
    }
    Ok(cudarc::nvrtc::Ptx::from_src(ptx))
}

/// Where a CUDA install puts its libraries, beyond the loader's own search path.
///
/// Shared by the driver preflight below and by NVRTC discovery in `nvrtc`, which want the
/// same answer for different files. NVIDIA's packages install under a versioned root and
/// add no `ldconfig` entry, and the Windows installer does not reliably put its `bin` on
/// `PATH`, so a correct install being invisible to a bare `dlopen`/`LoadLibrary` is the
/// normal case rather than the unusual one -- scanning these directories is what makes it
/// visible.
///
/// Newest last, since a caller that wants the newest sorts by the version in the filename
/// and a caller that does not is better off with the current install winning.
#[cfg(unix)]
fn search_dirs() -> Vec<std::path::PathBuf> {
    let mut dirs: Vec<std::path::PathBuf> = Vec::new();
    for root in ["/usr/local/cuda", "/opt/cuda"] {
        dirs.push(format!("{root}/lib64").into());
        dirs.push(format!("{root}/targets/x86_64-linux/lib").into());
    }
    // Versioned siblings: /usr/local/cuda-13.3 and friends.
    if let Ok(entries) = std::fs::read_dir("/usr/local") {
        let mut versioned: Vec<_> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("cuda-"))
            })
            .collect();
        versioned.sort();
        for base in versioned {
            dirs.push(base.join("lib64"));
            dirs.push(base.join("targets/x86_64-linux/lib"));
        }
    }
    dirs
}

#[cfg(windows)]
fn search_dirs() -> Vec<std::path::PathBuf> {
    use std::path::PathBuf;

    // Where the installer puts every toolkit it has ever installed, one directory per
    // version.
    let root = std::env::var_os("ProgramFiles")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Program Files"))
        .join("NVIDIA GPU Computing Toolkit")
        .join("CUDA");

    let mut dirs: Vec<PathBuf> = Vec::new();
    // Set by the installer to whichever toolkit it made current. The versioned
    // `CUDA_PATH_V13_0` siblings it also sets are covered by the scan below.
    for var in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
        if let Some(path) = std::env::var_os(var) {
            dirs.push(PathBuf::from(path).join("bin"));
        }
    }
    match std::fs::read_dir(&root) {
        Ok(entries) => {
            let mut versioned: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
            versioned.sort();
            dirs.extend(versioned.into_iter().map(|base| base.join("bin")));
        }
        // With no toolkit installed there are no directories to list, and an empty list
        // reads as a broken message rather than as the answer it is. The root an install
        // would have created is listed instead -- its absence is the whole diagnosis, and
        // scanning a directory that is not there costs nothing.
        Err(_) if dirs.is_empty() => dirs.push(root),
        Err(_) => {}
    }
    dirs
}

#[cfg(not(any(unix, windows)))]
fn search_dirs() -> Vec<std::path::PathBuf> {
    Vec::new()
}

/// Load the NVIDIA driver library, or say that this machine has no NVIDIA driver.
///
/// **This has to happen before `cudarc` is touched at all.** It dlopens its libraries from
/// a lazy static and *panics* when one is missing, and the release profile sets
/// `panic = "abort"` -- so `catch_unwind` is useless exactly where it matters, and a
/// missing library would take the whole scan down with a page of loader diagnostics
/// instead of a sentence. Asking first is the only thing that works in both profiles.
///
/// Only the driver is checked here. The kernel compiler is not a yes/no question any more
/// -- which NVRTC to use depends on the card, so it is chosen in `open` once the card can
/// be asked what it is.
#[cfg(unix)]
fn missing_library() -> Option<String> {
    fn dlopen(name: &str) -> bool {
        let Ok(c) = std::ffi::CString::new(name) else {
            return false;
        };
        // SAFETY: a valid NUL-terminated name. RTLD_LAZY resolves nothing eagerly, and
        // RTLD_GLOBAL registers the object so a later dlopen of its SONAME -- cudarc's --
        // finds it already loaded. The handle is deliberately never closed for that reason.
        unsafe { !libc::dlopen(c.as_ptr(), libc::RTLD_LAZY | libc::RTLD_GLOBAL).is_null() }
    }

    // The driver installs this on the loader's own path; the CUDA install directories are
    // searched too, because a container image with the toolkit mounted in is a normal way
    // to run this.
    if ["libcuda.so", "libcuda.so.1"].iter().any(|n| dlopen(n)) {
        return None;
    }
    for dir in search_dirs() {
        for name in ["libcuda.so", "libcuda.so.1"] {
            if dlopen(&dir.join(name).to_string_lossy()) {
                return None;
            }
        }
    }
    Some("libcuda, the NVIDIA driver library. Install the driver for your card.".to_string())
}

/// The same preflight for Windows, where the driver's own library is the only thing that
/// is reliably on the loader's path.
#[cfg(windows)]
fn missing_library() -> Option<String> {
    // SAFETY: loading a DLL runs its entry point, which is what the CUDA loader is about
    // to do to this same file regardless. The handle is deliberately leaked -- cudarc's
    // later bare-name lookup only finds this module while it stays open, and that is the
    // entire purpose of loading it here.
    let loaded = match unsafe { libloading::Library::new("nvcuda.dll") } {
        Ok(lib) => {
            std::mem::forget(lib);
            true
        }
        Err(_) => false,
    };
    (!loaded)
        .then(|| "nvcuda.dll, the NVIDIA driver library. Install the driver for your card.".to_string())
}

#[cfg(not(any(unix, windows)))]
fn missing_library() -> Option<String> {
    None
}

impl Cuda {
    pub fn open() -> Result<Self> {
        if let Some(what) = missing_library() {
            return Err(anyhow::Error::new(super::NoDevice(format!(
                "cannot use CUDA here: this machine has no {what}"
            ))));
        }
        // Absence is typed, so a machine with no NVIDIA card skips the GPU tests instead
        // of failing them. A device that exists and then misbehaves is a plain error.
        let context = CudaContext::new(0).map_err(|e| {
            anyhow::Error::new(super::NoDevice(format!(
                "no usable CUDA device: {e:?}. Is a driver installed?"
            )))
        })?;
        // The card decides which compiler can be used, so this is the first moment the
        // question can be asked at all -- and it is asked once, here, rather than at the
        // first compile, so that a box with no toolkit is an absent device (skipped) and
        // one whose toolkit cannot target its card is an error with a name (reported).
        let (major, minor) = context
            .compute_capability()
            .map_err(|e| anyhow!("asking the device for its compute capability: {e:?}"))?;
        let compiler = nvrtc::for_capability(major, minor).map_err(|e| match e {
            nvrtc::Failure::NotInstalled(what) => {
                anyhow::Error::new(super::NoDevice(format!("cannot use CUDA here: {what}")))
            }
            nvrtc::Failure::Unsupported(what) => anyhow!("{what}"),
        })?;
        let stream = context.default_stream();
        Ok(Self {
            context,
            stream,
            compiler,
            tuning: Tuning::for_capability(major),
            module: None,
            functions: HashMap::new(),
            buffers: Vec::new(),
            profile: std::env::var_os("MILKSAD_GPU_PROFILE").map(|_| HashMap::new()),
        })
    }

    fn function(&mut self, name: &str) -> Result<CudaFunction> {
        if let Some(f) = self.functions.get(name) {
            return Ok(f.clone());
        }
        let module = self
            .module
            .as_ref()
            .ok_or_else(|| anyhow!("dispatch of `{name}` before the kernels were compiled"))?;
        let f = module
            .load_function(name)
            .map_err(|e| anyhow!("the compiled module has no kernel named `{name}`: {e:?}"))?;
        self.functions.insert(name.to_string(), f.clone());
        Ok(f)
    }

    fn get(&self, id: BufferId) -> Result<&CudaSlice<u8>> {
        self.buffers
            .get(id.0)
            .ok_or_else(|| anyhow!("buffer {} does not exist", id.0))
    }
}

impl Backend for Cuda {
    fn name(&self) -> String {
        self.context
            .name()
            .unwrap_or_else(|_| "unknown CUDA device".to_string())
    }

    fn compiler(&self) -> Option<String> {
        Some(format!("{}{}", self.compiler.describe(), self.tuning.describe()))
    }

    fn compile(&mut self, source: &str) -> Result<()> {
        let ptx = compile_cached(&self.compiler, &self.tuning, source)?;
        super::trace("PTX ready, loading module");
        let module = self.context.load_module(ptx).map_err(|e| {
            // NVRTC produced PTX and the driver then refused it. That is a version
            // relationship rather than a broken kernel, and it is worth saying so: PTX
            // carries an ISA version, a driver only JITs the versions it knows, and CUDA's
            // minor-version compatibility promise covers linked binaries but explicitly not
            // JIT. So a toolkit newer than the driver fails here and nowhere earlier --
            // installation looks perfect right up to this line.
            //
            // The other way to land here was a target the card is older than, and that one
            // is gone: the architecture now comes from the card and the compiler is one
            // that said it could emit for it.
            use cudarc::driver::sys::CUresult;
            if matches!(
                e.0,
                CUresult::CUDA_ERROR_INVALID_PTX | CUresult::CUDA_ERROR_UNSUPPORTED_PTX_VERSION
            ) {
                anyhow!(
                    "the driver rejected the compiled kernels: {e:?}.\n\n\
                     They were compiled by {} for this card's own architecture, so \
                     neither the source nor the target is the problem -- what is left is a \
                     compiler newer than the driver. PTX carries an ISA version and a \
                     driver only JITs the versions it knows. Compare the two:\n    \
                     nvidia-smi        the newest CUDA the driver can JIT\n\n\
                     Fix it from either end: update the driver, or install an NVRTC no \
                     newer than it -- the older one will be picked up on the next run \
                     without a rebuild.",
                    self.compiler.describe()
                )
            } else {
                anyhow!("loading the compiled module: {e:?}")
            }
        })?;
        self.module = Some(module);
        self.functions.clear();
        Ok(())
    }

    fn buffer(&mut self, bytes: usize) -> Result<BufferId> {
        let buf = self
            .stream
            .alloc_zeros::<u8>(bytes.max(1))
            .map_err(|e| anyhow!("allocating a {bytes}-byte device buffer: {e:?}"))?;
        self.buffers.push(buf);
        Ok(BufferId(self.buffers.len() - 1))
    }

    fn buffer_from(&mut self, data: &[u8]) -> Result<BufferId> {
        let id = self.buffer(data.len())?;
        let buf = self
            .buffers
            .get_mut(id.0)
            .ok_or_else(|| anyhow!("buffer {} vanished", id.0))?;
        self.stream
            .memcpy_htod(data, buf)
            .map_err(|e| anyhow!("uploading {} bytes: {e:?}", data.len()))?;
        Ok(id)
    }

    unsafe fn buffer_no_copy(&mut self, ptr: *const u8, bytes: usize) -> Result<BufferId> {
        let free = self
            .context
            .mem_get_info()
            .map(|(free, _total)| free)
            .unwrap_or(0);
        if free > 0 && bytes > free {
            bail!(
                "the filter is {bytes} bytes and this card has {free} free. It must fit in \
                 device memory: reaching across PCIe for ~2,300 random probes per seed \
                 caps the whole sweep near 4,300 seeds/s, which is slower than scanning on \
                 the CPU. Build a smaller filter with `keyscan bf-gen` at a higher \
                 false-positive rate."
            );
        }
        // SAFETY: the caller guarantees `ptr` is valid for `bytes` and outlives every
        // dispatch. Unlike the Metal path this is a genuine copy -- a discrete card cannot
        // share the host's pages -- so the lifetime requirement is stronger than needed
        // here, and honouring it costs nothing.
        let host = unsafe { std::slice::from_raw_parts(ptr, bytes) };
        self.buffer_from(host)
    }

    fn dispatch(&mut self, kernel: &str, threads: usize, args: &[Arg<'_>]) -> Result<()> {
        if threads == 0 {
            return Ok(());
        }
        let func = self.function(kernel)?;
        // The kernel's own ceiling, not just the tuned value. A register-heavy kernel --
        // `k_kmul` is the candidate -- can be unable to fill a large block at all, and the
        // driver answers an over-sized launch with an invalid-configuration error rather
        // than a smaller block. Clamping keeps a block-size sweep from failing on the one
        // kernel the sweep is about, and the trace says when it bit.
        let ceiling = func.max_threads_per_block().unwrap_or(BLOCK as i32).max(32) as u32;
        let block = self.tuning.block.min(ceiling);
        if block != self.tuning.block {
            super::trace(&format!(
                "{kernel} caps at {block} threads per block, not the requested {}",
                self.tuning.block
            ));
        }
        let config = LaunchConfig {
            grid_dim: ((threads as u32).div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        // Scalars are widened to owned values first. `LaunchArgs` holds raw pointers to
        // whatever it is handed, so every argument has to outlive the launch; a temporary
        // built inside the loop would not.
        let mut scalars32: Vec<u32> = Vec::new();
        let mut scalars64: Vec<u64> = Vec::new();
        for arg in args {
            if let Arg::Scalar(bytes) = arg {
                match bytes.len() {
                    4 => scalars32.push(u32::from_le_bytes((*bytes).try_into().unwrap())),
                    8 => scalars64.push(u64::from_le_bytes((*bytes).try_into().unwrap())),
                    n => bail!("a {n}-byte kernel scalar has no CUDA parameter type"),
                }
            }
        }

        let mut builder = self.stream.launch_builder(&func);
        let (mut at32, mut at64) = (0, 0);
        for arg in args {
            match arg {
                Arg::Buffer(id) => {
                    let buf = self
                        .buffers
                        .get(id.0)
                        .ok_or_else(|| anyhow!("buffer {} does not exist", id.0))?;
                    builder.arg(buf);
                }
                Arg::Scalar(bytes) => match bytes.len() {
                    4 => {
                        builder.arg(&scalars32[at32]);
                        at32 += 1;
                    }
                    _ => {
                        builder.arg(&scalars64[at64]);
                        at64 += 1;
                    }
                },
            }
        }

        super::trace(&format!("dispatch {kernel} over {threads} threads"));
        let began = self.profile.is_some().then(std::time::Instant::now);
        // SAFETY: the kernel exists in the loaded module, the argument list matches its
        // signature by construction (`kernels/compat.h` numbers parameters the same way
        // the host orders this slice), and every buffer outlives the launch because
        // `self.buffers` owns them.
        unsafe { builder.launch(config) }
            .map_err(|e| anyhow!("dispatching `{kernel}` over {threads} threads: {e:?}"))?;
        if let Some(began) = began {
            // The launch above only enqueued. Waiting here is what makes the clock measure
            // the kernel rather than the driver call; see the note on `profile`.
            self.stream
                .synchronize()
                .map_err(|e| anyhow!("waiting for `{kernel}`: {e:?}"))?;
            let entry = self
                .profile
                .as_mut()
                .expect("profiling was on when the dispatch began")
                .entry(kernel.to_string())
                .or_default();
            entry.0 += began.elapsed();
            entry.1 += 1;
        }
        Ok(())
    }

    /// Where a launch's time went, and what each kernel costs in registers.
    ///
    /// The occupancy columns are the point of doing this on CUDA rather than only timing.
    /// A kernel's register count decides how many warps an SM can hold, and nothing in this
    /// project has ever set `__launch_bounds__` or `--maxrregcount`, so the allocator has
    /// been optimising each kernel for one thread's latency with no view of how many
    /// threads that leaves resident. `local` is the tell for having pushed too hard the
    /// other way: it is bytes spilled per thread, and anything above zero on `k_kmul` or
    /// `k_pbkdf2` means a `MILKSAD_CUDA_MAXREG` sweep has gone past the knee.
    fn report(&mut self) -> Option<String> {
        let profile = self.profile.as_ref()?;
        let mut rows: Vec<_> = profile.iter().collect();
        rows.sort_by_key(|(_, (d, _))| std::cmp::Reverse(*d));
        let total: f64 = rows.iter().map(|(_, (d, _))| d.as_secs_f64()).sum();

        let mut out = format!("gpu kernel profile ({total:.2}s total, stream serialised)\n");
        out.push_str(&format!(
            "  {:<16} {:>8} {:>7} {:>8} {:>5} {:>6} {:>6}\n",
            "kernel", "time", "share", "calls", "regs", "local", "block"
        ));
        for (name, (d, calls)) in rows {
            // A kernel that was timed is a kernel that was dispatched, so it is in
            // `functions`. Missing attributes are reported as `-` rather than guessed at.
            let (regs, local, max_block) = match self.functions.get(name) {
                Some(f) => (
                    f.num_regs().map(|n| n.to_string()).unwrap_or_else(|_| "-".into()),
                    f.local_size_bytes().map(|n| n.to_string()).unwrap_or_else(|_| "-".into()),
                    f.max_threads_per_block()
                        .map(|n| (self.tuning.block.min(n.max(32) as u32)).to_string())
                        .unwrap_or_else(|_| "-".into()),
                ),
                None => ("-".into(), "-".into(), "-".into()),
            };
            out.push_str(&format!(
                "  {name:<16} {:>7.2}s {:>6.1}% {calls:>8} {regs:>5} {local:>6} {max_block:>6}\n",
                d.as_secs_f64(),
                100.0 * d.as_secs_f64() / total.max(1e-9),
            ));
        }
        if self.tuning.options.len() > 1 || self.tuning.block != BLOCK {
            out.push_str(&format!(
                "  tuning: block {}, nvrtc {}\n",
                self.tuning.block,
                self.tuning.options.join(" ")
            ));
        }
        Some(out)
    }

    fn sync(&mut self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| anyhow!("waiting for the CUDA stream: {e:?}"))
    }

    /// Free VRAM, not total: whatever the display and anything else on the card is
    /// already holding is not available to this process, and on a workstation that can be
    /// a great deal.
    fn device_memory(&self) -> Option<u64> {
        match self.context.mem_get_info() {
            Ok((free, _total)) if free > 0 => Some(free as u64),
            _ => None,
        }
    }

    fn allocated_bytes(&self) -> usize {
        self.buffers.iter().map(|b| b.len()).sum()
    }

    fn read(&mut self, buffer: BufferId, out: &mut [u8]) -> Result<()> {
        let buf = self.get(buffer)?;
        if buf.len() < out.len() {
            bail!("reading {} bytes from a {}-byte buffer", out.len(), buf.len());
        }
        let view = buf.slice(0..out.len());
        self.stream
            .memcpy_dtoh(&view, out)
            .map_err(|e| anyhow!("reading back {} bytes: {e:?}", out.len()))
    }

    fn write(&mut self, buffer: BufferId, data: &[u8]) -> Result<()> {
        let len = self.get(buffer)?.len();
        if len < data.len() {
            bail!("writing {} bytes into a {len}-byte buffer", data.len());
        }
        let buf = self
            .buffers
            .get_mut(buffer.0)
            .ok_or_else(|| anyhow!("buffer {} does not exist", buffer.0))?;
        let mut view = buf.slice_mut(0..data.len());
        self.stream
            .memcpy_htod(data, &mut view)
            .map_err(|e| anyhow!("writing {} bytes: {e:?}", data.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rule, stated as the two cards it came from. Pure arithmetic over a capability,
    /// so it is checkable on a machine with no NVIDIA card in it -- which is where this is
    /// most likely to be edited.
    #[test]
    fn pascal_and_older_get_the_occupancy_defaults() {
        // GTX 1080 Ti (6.1) and the Maxwell generation behind it.
        assert_eq!(defaults_for(6), (128, 128));
        assert_eq!(defaults_for(5), (128, 128));
        // Turing through Blackwell keep what was measured on an RTX 5070 Ti: the stock
        // block, and the register allocator left alone.
        assert_eq!(defaults_for(7), (BLOCK, 0));
        assert_eq!(defaults_for(12), (BLOCK, 0));
    }

    /// The banner names a default that varies by card, and says nothing when the card got
    /// the values the kernels were written against.
    #[test]
    fn the_banner_names_a_card_that_was_tuned_differently() {
        // Environment overrides would make this a test of the machine it runs on.
        if std::env::var_os("MILKSAD_CUDA_BLOCK").is_some()
            || std::env::var_os("MILKSAD_CUDA_MAXREG").is_some()
        {
            return;
        }
        assert_eq!(Tuning::for_capability(12).describe(), "");
        assert_eq!(
            Tuning::for_capability(6).describe(),
            ", 128 threads per block, 128-register cap"
        );
    }
}
