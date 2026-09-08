//! NVRTC, found and bound at run time.
//!
//! `cudarc` wraps NVRTC too, and this deliberately does not use it. Its bindings are
//! generated per CUDA version and the library filename is derived from that same version,
//! so a build pinned to 13 asks the loader for `libnvrtc.so.13` / `nvrtc64_130_0.dll` and
//! for nothing else. That makes the toolkit major a **build-time** decision about a machine
//! the binary has not met yet, which is the wrong shape for the only question that
//! actually matters:
//!
//! > can this compiler generate code for the card in this box?
//!
//! CUDA 13 removed Maxwell, Pascal and Volta, so for a GTX 1080 Ti the answer is no for
//! 13.x and yes for 12.x -- whatever `nvidia-smi` reports, since the driver's CUDA version
//! is a ceiling and says nothing about the floor. Deciding that at build time got it wrong
//! in the worst way available: the kernels compiled for an architecture the card is older
//! than, and the failure surfaced two layers later as `CUDA_ERROR_INVALID_PTX` from the
//! driver, which reads as a broken toolchain rather than as the wrong compiler.
//!
//! So NVRTC is loaded here instead. Every version installed is a candidate, each one is
//! asked what it can emit through `nvrtcGetSupportedArchs`, and the newest one that covers
//! this card compiles the kernels. **The list of architectures comes from the library**, so
//! nothing here has to know what a future toolkit adds or drops -- the next removal answers
//! itself. Installing a second toolkit is enough to fix a card that the first one cannot
//! target; there is nothing to rebuild and no flag to pass.
//!
//! The driver side stays on `cudarc`, where the version pin is harmless: it resolves driver
//! entry points lazily, one at a time, and this backend calls only long-stable ones.
//!
//! Eleven entry points, all of them ABI-stable since CUDA 11.2 -- which is where
//! `nvrtcGetSupportedArchs` came in, and therefore the oldest NVRTC this will use. The
//! kernels are one self-contained translation unit with no `#include` of a local file (see
//! `gpu::source`), so there are no header callbacks to bind and no include paths to get
//! right.

use anyhow::{Result, anyhow, bail};
use std::ffi::{CStr, CString, OsString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};

/// `nvrtcProgram`, an opaque handle.
type Program = *mut c_void;

/// `NVRTC_SUCCESS`. Every other value is an error whose text `nvrtcGetErrorString` knows.
const SUCCESS: c_int = 0;

/// The entry points, as plain function pointers taken out of a loaded library.
///
/// `libloading::Symbol` borrows the `Library` it came from, so it cannot be stored beside
/// it. Dereferencing one gives the raw pointer, which stays valid for as long as the
/// library stays loaded -- `Nvrtc::_lib` is what guarantees that, and it is never unloaded.
struct Api {
    version: unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int,
    num_archs: unsafe extern "C" fn(*mut c_int) -> c_int,
    archs: unsafe extern "C" fn(*mut c_int) -> c_int,
    create: unsafe extern "C" fn(
        *mut Program,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> c_int,
    compile: unsafe extern "C" fn(Program, c_int, *const *const c_char) -> c_int,
    log_size: unsafe extern "C" fn(Program, *mut usize) -> c_int,
    log: unsafe extern "C" fn(Program, *mut c_char) -> c_int,
    ptx_size: unsafe extern "C" fn(Program, *mut usize) -> c_int,
    ptx: unsafe extern "C" fn(Program, *mut c_char) -> c_int,
    destroy: unsafe extern "C" fn(*mut Program) -> c_int,
    error_string: unsafe extern "C" fn(c_int) -> *const c_char,
}

impl Api {
    /// Bind every entry point, or say which one is missing.
    ///
    /// All or nothing on purpose: a library that cannot answer `nvrtcGetSupportedArchs` is
    /// one this cannot ask the only question it cares about, so it is passed over here
    /// rather than used blind. In practice that means NVRTC older than 11.2.
    ///
    /// # Safety
    ///
    /// The signatures below have to match the ones in `nvrtc.h`. They have not changed
    /// since the functions were introduced.
    unsafe fn bind(lib: &libloading::Library) -> Result<Self> {
        unsafe fn sym<T: Copy>(lib: &libloading::Library, name: &str) -> Result<T> {
            let symbol: libloading::Symbol<T> =
                unsafe { lib.get(name) }.map_err(|e| anyhow!("no `{name}` in it: {e}"))?;
            Ok(*symbol)
        }
        unsafe {
            Ok(Self {
                version: sym(lib, "nvrtcVersion")?,
                num_archs: sym(lib, "nvrtcGetNumSupportedArchs")?,
                archs: sym(lib, "nvrtcGetSupportedArchs")?,
                create: sym(lib, "nvrtcCreateProgram")?,
                compile: sym(lib, "nvrtcCompileProgram")?,
                log_size: sym(lib, "nvrtcGetProgramLogSize")?,
                log: sym(lib, "nvrtcGetProgramLog")?,
                ptx_size: sym(lib, "nvrtcGetPTXSize")?,
                ptx: sym(lib, "nvrtcGetPTX")?,
                destroy: sym(lib, "nvrtcDestroyProgram")?,
                error_string: sym(lib, "nvrtcGetErrorString")?,
            })
        }
    }
}

/// One loaded NVRTC, and what it says about itself.
struct Nvrtc {
    api: Api,
    /// Kept so the function pointers stay valid. Never unloaded, and never dropped.
    _lib: libloading::Library,
    version: (c_int, c_int),
    /// The `sm_` values it can emit, ascending, exactly as it reports them.
    archs: Vec<c_int>,
    /// Where it came from, for the message that lists what was found.
    origin: String,
}

impl Nvrtc {
    /// Load one candidate and interrogate it.
    ///
    /// `Ok(None)` is "there is no such library", which is the ordinary answer for most
    /// candidates -- the loader path is probed by name, so every name that is not installed
    /// misses. An `Err` is the interesting one: something is there and cannot be used, and
    /// that has to be told apart from absence or a broken install gets reported as a
    /// missing one.
    fn open(candidate: &Candidate) -> Result<Option<Self>> {
        // SAFETY: loading a library runs its initialisers, which is what the CUDA loader
        // would do to this exact file anyway. The handle is kept for the process lifetime,
        // so the pointers `bind` takes out of it never dangle.
        let Ok(lib) = (unsafe { libloading::Library::new(&candidate.name) }) else {
            return Ok(None);
        };
        let api = unsafe { Api::bind(&lib) }?;

        let (mut major, mut minor) = (0, 0);
        // SAFETY: two out-parameters, both of which NVRTC only writes.
        let status = unsafe { (api.version)(&mut major, &mut minor) };
        if status != SUCCESS {
            bail!("nvrtcVersion failed ({status})");
        }

        let mut count = 0;
        // SAFETY: same shape -- a count out-parameter, then a buffer of exactly that many
        // ints, which is the two-call protocol `nvrtcGetSupportedArchs` documents.
        let status = unsafe { (api.num_archs)(&mut count) };
        if status != SUCCESS || count <= 0 {
            bail!("it reports no supported architectures ({status})");
        }
        let mut archs = vec![0; count as usize];
        let status = unsafe { (api.archs)(archs.as_mut_ptr()) };
        if status != SUCCESS {
            bail!("nvrtcGetSupportedArchs failed ({status})");
        }
        archs.sort_unstable();

        Ok(Some(Self {
            api,
            _lib: lib,
            version: (major, minor),
            archs,
            origin: candidate.origin(),
        }))
    }

    /// The best virtual architecture this compiler can offer a card of this `sm`.
    fn best_arch(&self, sm: c_int) -> Option<c_int> {
        best_arch(&self.archs, sm)
    }

    /// NVRTC's own text for a status code.
    fn message(&self, status: c_int) -> String {
        // SAFETY: `nvrtcGetErrorString` returns a static string for every value, including
        // ones it does not recognise.
        let text = unsafe { CStr::from_ptr((self.api.error_string)(status)) };
        format!("{} ({status})", text.to_string_lossy())
    }

    /// `NVRTC 13.0  targets sm_75 .. sm_121  /usr/local/cuda-13.0/lib64/libnvrtc.so.13`
    fn describe(&self) -> String {
        let (lo, hi) = (self.archs.first(), self.archs.last());
        let range = match (lo, hi) {
            (Some(lo), Some(hi)) if lo != hi => format!("sm_{lo} .. sm_{hi}"),
            (Some(only), _) => format!("sm_{only}"),
            _ => "nothing".to_string(),
        };
        format!(
            "NVRTC {}.{}  targets {range}  {}",
            self.version.0, self.version.1, self.origin
        )
    }
}

/// The best virtual architecture a compiler that supports `archs` can offer a card of this
/// `sm`: the highest supported one that is not newer than the card.
///
/// PTX is forward compatible and only forward. The driver JITs `compute_61` onto anything
/// from sm_61 up and refuses `compute_75` on sm_61 outright, so "close enough" has a
/// direction: a card *newer* than everything the compiler knows still gets the best target
/// available, and a card *older* than all of it correctly gets no answer at all rather than
/// a module the driver will reject. Falling the wrong way across that line is the whole of
/// the bug this replaced.
fn best_arch(archs: &[c_int], sm: c_int) -> Option<c_int> {
    archs.iter().copied().filter(|a| *a <= sm).max()
}

/// The two-call `nvrtcGet*Size` / `nvrtcGet*` shape, which the log and the PTX share.
///
/// The size includes the terminator and the buffer comes back NUL-terminated, so the tail
/// is trimmed here: a `String` carrying an interior NUL is rejected by `CString::new` when
/// the PTX is handed to the driver, several steps from anything that would explain it.
fn fetch(
    prog: Program,
    size: unsafe extern "C" fn(Program, *mut usize) -> c_int,
    get: unsafe extern "C" fn(Program, *mut c_char) -> c_int,
) -> Option<String> {
    let mut n = 0usize;
    // SAFETY: `prog` is live for the whole call, and the buffer is exactly the size NVRTC
    // just asked for.
    unsafe {
        if size(prog, &mut n) != SUCCESS || n == 0 {
            return None;
        }
        let mut buf = vec![0u8; n];
        if get(prog, buf.as_mut_ptr().cast()) != SUCCESS {
            return None;
        }
        while buf.last() == Some(&0) {
            buf.pop();
        }
        String::from_utf8(buf).ok()
    }
}

/// The compiler chosen for one card: an NVRTC, and the architecture it will be asked for.
pub struct Compiler {
    nvrtc: Nvrtc,
    arch: c_int,
}

impl Compiler {
    /// The `--gpu-architecture` value, which is also part of the PTX cache key.
    pub fn arch(&self) -> String {
        format!("compute_{}", self.arch)
    }

    /// The NVRTC version, in the cache key because two versions do not produce the same
    /// PTX from the same source -- and a cache under `~/.cache` outlives an install.
    pub fn version(&self) -> (i32, i32) {
        self.nvrtc.version
    }

    /// One line for the banner, naming a choice that was made without being asked.
    pub fn describe(&self) -> String {
        let (major, minor) = self.nvrtc.version;
        format!("NVRTC {major}.{minor}, compiling for {}", self.arch())
    }

    /// Compile a translation unit to PTX, or return NVRTC's own diagnostics.
    ///
    /// The compiler's log is the error rather than a footnote: a runtime-compiled kernel
    /// has no other channel, and "compilation failed" without it is unactionable.
    pub fn compile(&self, source: &str, options: &[String]) -> Result<String> {
        let api = &self.nvrtc.api;
        let src = CString::new(source).map_err(|_| anyhow!("the kernel source has a NUL in it"))?;

        let mut prog: Program = std::ptr::null_mut();
        // SAFETY: a NUL-terminated source and name, and no headers -- the translation unit
        // is self-contained, so the two header arrays are legitimately null.
        let status = unsafe {
            (api.create)(
                &mut prog,
                src.as_ptr(),
                c"milksad.cu".as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if status != SUCCESS {
            bail!("nvrtcCreateProgram: {}", self.nvrtc.message(status));
        }
        /// Destroys the program on every path out, including the compile-failed one.
        struct Owned<'a> {
            api: &'a Api,
            prog: Program,
        }
        impl Drop for Owned<'_> {
            fn drop(&mut self) {
                // SAFETY: created by `nvrtcCreateProgram` above and not destroyed since.
                unsafe { (self.api.destroy)(&mut self.prog) };
            }
        }
        let prog = Owned { api, prog };

        let mut opts = vec![CString::new(format!("--gpu-architecture={}", self.arch()))?];
        for option in options {
            opts.push(
                CString::new(option.as_str())
                    .map_err(|_| anyhow!("the NVRTC option `{option}` has a NUL in it"))?,
            );
        }
        let ptrs: Vec<*const c_char> = opts.iter().map(|o| o.as_ptr()).collect();
        // SAFETY: `ptrs` and the `CString`s it points into both outlive the call.
        let status = unsafe { (api.compile)(prog.prog, ptrs.len() as c_int, ptrs.as_ptr()) };
        let log = fetch(prog.prog, api.log_size, api.log).unwrap_or_default();
        if status != SUCCESS {
            bail!(
                "compiling the CUDA kernels with {}:\n{}\n{}",
                self.describe(),
                self.nvrtc.message(status),
                log.trim_end()
            );
        }
        // A compile that succeeded can still have said something. Nothing prints it, so it
        // goes where the rest of the startup detail goes.
        if !log.trim().is_empty() {
            crate::gpu::trace(&format!("NVRTC log:\n{}", log.trim_end()));
        }
        fetch(prog.prog, api.ptx_size, api.ptx)
            .ok_or_else(|| anyhow!("NVRTC reported success and produced no PTX"))
    }
}

/// Why no compiler could be chosen. The two cases want opposite handling, so they are
/// separate: one is a machine without a toolkit, the other is a real misconfiguration.
pub enum Failure {
    /// Nothing that looks like NVRTC is installed anywhere that was looked. Reported as an
    /// absent device, so a box with no toolkit skips the GPU tests rather than failing
    /// every one of them.
    NotInstalled(String),
    /// NVRTC is installed, and none of what is installed can generate code for this card.
    /// A hard error: the card is there, the driver is there, and the one missing piece can
    /// be named exactly.
    Unsupported(String),
}

/// Load the newest installed NVRTC that can generate code for a card of this capability.
///
/// Candidates are tried newest first and the search stops at the first one that covers the
/// card, so the usual machine loads exactly one library. Only a failure walks the whole
/// list -- and then having loaded them all is what makes the message specific.
pub fn for_capability(major: i32, minor: i32) -> std::result::Result<Compiler, Failure> {
    let sm = major * 10 + minor;
    let mut candidates = candidates();
    // Newest first. Two candidates can be the same file reached two ways -- a bare name
    // found on the loader path and the same library found by scanning an install directory
    // -- which costs one redundant load at worst and is caught by the version check below.
    candidates.sort_by_key(|c| std::cmp::Reverse(c.major));

    let mut found: Vec<Nvrtc> = Vec::new();
    let mut rejected: Vec<String> = Vec::new();
    for candidate in &candidates {
        let nvrtc = match Nvrtc::open(candidate) {
            Ok(Some(nvrtc)) => nvrtc,
            Ok(None) => continue,
            Err(e) => {
                rejected.push(format!("{}: {e}", candidate.origin()));
                continue;
            }
        };
        if found.iter().any(|f| f.version == nvrtc.version) {
            continue;
        }
        if let Some(arch) = nvrtc.best_arch(sm) {
            crate::gpu::trace(&format!("NVRTC chosen: {}", nvrtc.describe()));
            // The companion library is opened by bare name at the first compile, so a
            // toolkit that is off the loader path fails there rather than here unless it
            // is loaded now, from beside the library that will ask for it.
            if let Some(dir) = candidate.dir.as_deref() {
                load_builtins(dir);
            }
            return Ok(Compiler { nvrtc, arch });
        }
        found.push(nvrtc);
    }

    for reason in &rejected {
        crate::gpu::trace(&format!("NVRTC passed over -- {reason}"));
    }
    Err(if found.is_empty() {
        Failure::NotInstalled(not_installed(&rejected))
    } else {
        Failure::Unsupported(unsupported(major, minor, &found))
    })
}

/// What to say when no usable NVRTC could be found.
///
/// `rejected` is anything that was there and could not be used -- a library that failed to
/// load, or one too old to be asked what it supports. It is almost always empty, and when
/// it is not it is the entire answer: reporting a broken install as an absent one sends
/// someone off to install what they already have.
fn not_installed(rejected: &[String]) -> String {
    let mut message = format!(
        "libnvrtc, the CUDA kernel compiler. It ships with the CUDA *toolkit*, not with \
         the driver -- the driver installs libcuda alone. The compiler package by itself \
         is enough: `cuda-nvrtc-<major>-<minor>` from NVIDIA's repo, or the CUDA Toolkit \
         installer on Windows.\n\n\
         Any version is usable, so install whichever one covers your card. Searched the \
         loader path and:\n{}",
        list(
            &super::search_dirs()
                .iter()
                // A directory that is not there is not a failed search, it is the answer:
                // on Windows the toolkit root's absence *is* "no CUDA toolkit installed".
                .map(|d| match d.is_dir() {
                    true => d.display().to_string(),
                    false => format!("{} (does not exist)", d.display()),
                })
                .collect::<Vec<_>>()
        )
    );
    if !rejected.is_empty() {
        message.push_str(&format!(
            "\n\nSomething is installed and could not be used. NVRTC older than CUDA 11.2 \
             cannot be asked which architectures it supports, so it is passed over rather \
             than used blind:\n{}",
            list(rejected)
        ));
    }
    message
}

/// What to say when NVRTC is installed and none of it targets this card.
///
/// The interesting failure now, and the one worth spending a paragraph on: every piece is
/// correctly installed, `nvidia-smi` is happy, and the compiler that is present cannot
/// emit code for the card that is present. Naming what each one *does* target is what
/// turns that from a contradiction into a shopping list.
fn unsupported(major: i32, minor: i32, found: &[Nvrtc]) -> String {
    format!(
        "no installed CUDA compiler can generate code for this GPU.\n\n    \
         this card  is compute capability {major}.{minor} (sm_{})\n{}\n\n\
         A toolkit drops architectures as well as gaining them, so for an old card the \
         older toolkit is the more capable one: CUDA 13 removed Maxwell, Pascal and Volta, \
         which is what a GTX 10-series needs 12.x for, and 12.x in turn reaches back to \
         Maxwell but no further.\n\n\
         Installing one that covers this card is the whole fix. The compiler is chosen at \
         run time from what is present, so nothing has to be rebuilt and no flag has to \
         change, and the NVRTC package alone will do. The driver already installed serves \
         it: a driver runs any toolkit no newer than itself.",
        major * 10 + minor,
        list(&found.iter().map(Nvrtc::describe).collect::<Vec<_>>())
    )
}

/// Indent a list under the sentence that introduces it, or say it is empty.
fn list(items: &[String]) -> String {
    if items.is_empty() {
        return "    (nothing)".to_string();
    }
    items
        .iter()
        .map(|i| format!("    {i}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// One library to try.
struct Candidate {
    /// What the loader is handed: an absolute path, or a bare filename to search for.
    name: OsString,
    /// The directory it was scanned out of, where its builtins sit. `None` when the name
    /// is being left to the loader's own search path, which has no directory to point at.
    dir: Option<PathBuf>,
    /// The CUDA major version the filename encodes, for ordering. Zero when it carries
    /// none, which sorts last -- an unversioned symlink is a worse answer than a file that
    /// says what it is.
    major: u32,
}

impl Candidate {
    fn origin(&self) -> String {
        match self.dir {
            Some(_) => self.name.to_string_lossy().into_owned(),
            None => format!("{} (loader path)", self.name.to_string_lossy()),
        }
    }
}

/// The CUDA major version an NVRTC filename encodes, and the filenames to ask the loader
/// for directly.
///
/// A whole major line ships one Windows filename -- every CUDA 12.x is `nvrtc64_120_0.dll`
/// and every 13.x is `nvrtc64_130_0.dll` -- so the leading two digits are the entire
/// version signal there, and the trailing pair are not a minor version to compare against.
/// Unix versions its SONAME by major alone for the same reason.
#[cfg(windows)]
mod names {
    /// Tried against the loader's own path, which can only be asked and not scanned. The
    /// scan below is what finds versions this list has never heard of.
    pub const BARE: &[(u32, &str)] = &[(13, "nvrtc64_130_0.dll"), (12, "nvrtc64_120_0.dll")];
    pub const BUILTINS: &str = "nvrtc-builtins64";

    pub fn major_of(name: &str) -> Option<u32> {
        if !name.ends_with(".dll") {
            return None;
        }
        let digits: String = name
            .strip_prefix("nvrtc64_")?
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        digits.get(..2)?.parse().ok()
    }
}

#[cfg(unix)]
mod names {
    pub const BARE: &[(u32, &str)] = &[
        (13, "libnvrtc.so.13"),
        (12, "libnvrtc.so.12"),
        (11, "libnvrtc.so.11"),
        (0, "libnvrtc.so"),
    ];
    pub const BUILTINS: &str = "libnvrtc-builtins";

    pub fn major_of(name: &str) -> Option<u32> {
        let digits: String = name
            .strip_prefix("libnvrtc.so.")?
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        digits.parse().ok()
    }
}

/// Everything worth trying to load: the loader's path by name, then every NVRTC sitting in
/// an install directory, whatever version it turns out to be.
#[cfg(any(unix, windows))]
fn candidates() -> Vec<Candidate> {
    let mut out: Vec<Candidate> = names::BARE
        .iter()
        .map(|(major, name)| Candidate {
            name: (*name).into(),
            dir: None,
            major: *major,
        })
        .collect();
    for dir in super::search_dirs() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let Some(major) = names::major_of(&entry.file_name().to_string_lossy()) else {
                continue;
            };
            out.push(Candidate {
                name: entry.path().into(),
                dir: Some(dir.clone()),
                major,
            });
        }
    }
    out
}

#[cfg(not(any(unix, windows)))]
fn candidates() -> Vec<Candidate> {
    Vec::new()
}

/// Load every `nvrtc-builtins` beside a chosen NVRTC.
///
/// NVRTC opens this companion by bare name when it compiles, so an install that is off the
/// loader path fails at the first compile even though NVRTC itself loaded fine -- with a
/// "failed to open libnvrtc-builtins.so.13.3" that looks like a broken installation and is
/// not one. Loading it here by absolute path puts it where that lookup will find it: both
/// loaders resolve a bare name against the modules already loaded before searching. The
/// version suffix is whatever the toolkit shipped, so the directory is scanned rather than
/// a name being guessed.
#[cfg(any(unix, windows))]
fn load_builtins(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if entry
            .file_name()
            .to_string_lossy()
            .starts_with(names::BUILTINS)
        {
            // SAFETY: as in `Nvrtc::open` -- and this one is loaded purely so that NVRTC's
            // own lookup finds it, so the handle is leaked rather than kept.
            if let Ok(lib) = unsafe { libloading::Library::new(entry.path()) } {
                std::mem::forget(lib);
            }
        }
    }
}

#[cfg(not(any(unix, windows)))]
fn load_builtins(_dir: &Path) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// What CUDA 13.0 and 12.9 answer `nvrtcGetSupportedArchs` with. Hard-coded because
    /// the point of the test is the rule applied to them, not the lists themselves -- the
    /// running code always asks the library rather than consulting a table like this.
    const THIRTEEN: [c_int; 10] = [75, 80, 86, 87, 89, 90, 100, 103, 120, 121];
    const TWELVE: [c_int; 16] = [50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90, 100, 120];

    /// The 1080 Ti, which is what this was all for: CUDA 13 cannot compile for it and must
    /// say so, CUDA 12 can and must pick its exact architecture.
    #[test]
    fn a_card_older_than_the_compiler_gets_no_architecture_rather_than_a_wrong_one() {
        assert_eq!(best_arch(&THIRTEEN, 61), None);
        assert_eq!(best_arch(&TWELVE, 61), Some(61));
    }

    /// The other direction, where guessing is safe: PTX JITs forward, so a card newer than
    /// anything the compiler lists still runs the newest target it does list.
    #[test]
    fn a_card_newer_than_the_compiler_falls_back_to_its_newest_architecture() {
        assert_eq!(best_arch(&TWELVE, 121), Some(120));
        assert_eq!(best_arch(&THIRTEEN, 130), Some(121));
        // An exact match is still preferred wherever there is one.
        assert_eq!(best_arch(&THIRTEEN, 89), Some(89));
    }

    /// Every library found is a candidate, so the filename has to be read the same way the
    /// installers write it -- including the Windows quirk that a whole major line ships one
    /// filename and the trailing digits are not a minor version.
    #[test]
    fn a_filename_says_which_cuda_line_it_belongs_to() {
        #[cfg(windows)]
        {
            assert_eq!(names::major_of("nvrtc64_130_0.dll"), Some(13));
            assert_eq!(names::major_of("nvrtc64_120_0.dll"), Some(12));
            assert_eq!(names::major_of("nvrtc64_112_0.dll"), Some(11));
            assert_eq!(names::major_of("nvrtc-builtins64_130.dll"), None);
            assert_eq!(names::major_of("nvrtc64_130_0.lib"), None);
        }
        #[cfg(unix)]
        {
            assert_eq!(names::major_of("libnvrtc.so.13"), Some(13));
            assert_eq!(names::major_of("libnvrtc.so.12.9.86"), Some(12));
            assert_eq!(names::major_of("libnvrtc.so"), None);
            assert_eq!(names::major_of("libnvrtc-builtins.so.13.3"), None);
        }
    }
}


