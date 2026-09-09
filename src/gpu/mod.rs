//! The GPU backend: the same sweep, on a device with thousands of lanes.
//!
//! The CPU path in `derive` is close to its ceiling -- it uses the ARMv8 SHA-512 and
//! SHA-256 instructions, a hand-tuned fixed-base comb, and batched probes -- so more
//! throughput has to come from a different processor rather than a better loop.
//!
//! **What the GPU is, and is not.** It is a *filter*. It walks the same tree and probes
//! the same bloom filter, and when a hash160 survives it reports the seed that produced
//! it and nothing more. The host then re-derives that one seed through `derive::Deriver`
//! and pushes it through the ordinary visitor, so every phrase ever written to
//! `matches.txt` came out of the CPU implementation that the test suite pins. A GPU
//! record the CPU cannot confirm is a bug, and `Gpu::unconfirmed` counts them; that
//! number reading zero across a multi-day sweep is a continuous, free correctness check
//! on the whole kernel set.
//!
//! **Why the kernels are compiled at runtime.** Both backends compile from source when
//! the scan starts: Metal through `newLibraryWithSource` (the compiler ships with the OS,
//! so no Xcode and no `xcrun metal`, which is not present on a Command Line Tools
//! install) and CUDA through NVRTC inside the driver. That removes any build-time GPU
//! toolchain, lets one binary run on a freshly rented box, and buys something better: the
//! `Scope` is baked into the source as `-D` defines, so every level loop unrolls and
//! every stride is a compile-time constant. See `source`.
//!
//! **Where the GPU deliberately differs from the CPU.** Three of the CPU's optimisations
//! are not ported, because each exists to fill a single core's issue slots and is a
//! pessimisation given millions of lanes:
//!
//! - `ec::Batch`'s cross-lane affine machinery. A Fermat inversion is ~270 *sequential*
//!   field operations; sharing one across a threadgroup stalls every lane in it. The GPU
//!   accumulates in Jacobian coordinates per thread and converts a whole level to affine
//!   with one inversion afterwards, so the `ABANDONED` lane fallback has no analogue here.
//! - `sha512x2`'s two-lane scheduling, which exists because one PBKDF2 chain cannot fill
//!   an ARMv8 SHA-512 unit. A launch has tens of thousands of independent chains, so the
//!   GPU runs one stream per thread.
//! - `bloom::contains_batch`'s two-pass probe, which exists to overlap DRAM misses inside
//!   one core's out-of-order window. A GPU has thousands of requests in flight already.

use anyhow::Result;

pub mod layout;
#[cfg(test)]
mod parity;
pub mod source;
#[cfg(all(test, feature = "gpu"))]
mod sweep;

// Metal is preferred wherever both are compiled in, which can only be a Mac -- and a Mac
// has no CUDA device to find. The module still has to compile in that combination, so it
// is built and then unused.
#[cfg(feature = "cuda")]
#[cfg_attr(feature = "metal", allow(dead_code))]
mod cuda;
#[cfg(feature = "metal")]
mod metal;

pub use layout::Layout;

/// How the scan should use the GPU.
///
/// Absent from the command line entirely, this is `None` and the scan is exactly what it
/// was before any of this existed.
#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum)]
pub enum Mode {
    /// GPU workers and `--threads` CPU workers both pull from the block queue.
    Both,
    /// GPU workers alone. The CPU still re-derives hits, checkpoints, and draws progress
    /// -- it just does not walk any seeds of its own.
    Only,
}

/// How many bytes one wordlist entry takes on the device.
///
/// Fixed width, so a lookup is a multiply rather than a prefix sum over lengths. One
/// length byte then eight characters, padded -- "abandon" is 7 and the longest BIP39
/// English word is 8. Mirrored by `BIP39_WORD_STRIDE` in kernels/bip39.h.
pub const WORD_STRIDE: usize = 9;

/// The BIP39 wordlist as the flat table `bip39_phrase` indexes.
pub fn wordlist() -> Vec<u8> {
    let mut out = vec![0u8; 2048 * WORD_STRIDE];
    for (i, word) in crate::wallet::bip39::WORDLIST.iter().enumerate() {
        let bytes = word.as_bytes();
        debug_assert!(bytes.len() < WORD_STRIDE, "{word} is too long for the stride");
        out[i * WORD_STRIDE] = bytes.len() as u8;
        out[i * WORD_STRIDE + 1..i * WORD_STRIDE + 1 + bytes.len()].copy_from_slice(bytes);
    }
    out
}

/// There is no usable device here -- as distinct from a device that exists and then went
/// wrong.
///
/// A typed error rather than a string, because the tests have to tell those two apart and
/// getting it wrong is silent in the worst direction: a build that skips every GPU test on
/// a machine that has a GPU reports success having checked nothing. This used to be a
/// `contains("no Metal device")` match, which could never be true of a CUDA error and so
/// turned "no NVIDIA card" into a panic.
#[derive(Debug)]
pub struct NoDevice(pub String);

impl std::fmt::Display for NoDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for NoDevice {}

/// Is this the absence of a device, rather than a fault in one?
///
/// Only the tests need the distinction: a scan cannot proceed either way and reports the
/// error as it stands, while a test must skip on absence and fail on a fault. Conflating
/// the two is silent in the worst direction -- a suite that skips every GPU test on a
/// machine that has a GPU reports success having checked nothing, which is exactly what
/// hid the zero-length-array bug in `source::list`.
#[cfg(test)]
pub fn is_unavailable(error: &anyhow::Error) -> bool {
    error.downcast_ref::<NoDevice>().is_some()
}

/// A buffer living on the device, identified by index rather than by a typed handle so
/// the trait stays object-safe and the backends stay free to store whatever they need.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BufferId(pub usize);

/// One argument of a dispatch.
///
/// Small scalars go inline rather than through a buffer, which is what both APIs want for
/// a handful of bytes and saves a round trip per launch.
///
/// Only a backend ever reads the payloads, so a build with none compiled in genuinely has
/// no reader for them. That is the honest state of a `cargo build` with no features, not
/// something to hide behind a blanket allow.
#[cfg_attr(not(any(feature = "metal", feature = "cuda")), allow(dead_code))]
pub enum Arg<'a> {
    Buffer(BufferId),
    Scalar(&'a [u8]),
}

/// What a GPU has to be able to do. Deliberately small: everything above this line is
/// shared between Metal and CUDA, and anything that leaks a platform concept into the
/// launch loop belongs below it instead.
pub trait Backend: Send {
    /// What to call this device in the startup banner.
    fn name(&self) -> String;

    /// The kernel compiler this backend ended up using, when that was a choice it made
    /// rather than a property of the machine.
    ///
    /// CUDA picks an NVRTC out of whatever is installed, against what the card can run --
    /// see `gpu::cuda::nvrtc` -- and a choice made on the user's behalf should be visible
    /// in the banner rather than inferred from a failure. Metal has one compiler, which is
    /// the operating system, so it has nothing to say here.
    fn compiler(&self) -> Option<String> {
        None
    }

    /// Compile the assembled translation unit. Errors carry the compiler's own
    /// diagnostics, which is the only way a runtime-compiled kernel is debuggable.
    fn compile(&mut self, source: &str) -> Result<()>;

    /// Allocate `bytes` of zeroed device memory.
    fn buffer(&mut self, bytes: usize) -> Result<BufferId>;

    /// Allocate and fill from host memory.
    fn buffer_from(&mut self, data: &[u8]) -> Result<BufferId>;

    /// Expose a host allocation to the device without copying it, where the platform can.
    ///
    /// This is what keeps the 7.6 GB filter to one copy on unified memory. Backends that
    /// cannot do it (any discrete card) copy instead and say so in the banner.
    ///
    /// # Safety
    /// `ptr` must be page-aligned, `bytes` must be a multiple of the page size, and the
    /// allocation must outlive every dispatch that reads the returned buffer.
    unsafe fn buffer_no_copy(&mut self, ptr: *const u8, bytes: usize) -> Result<BufferId>;

    /// Run `kernel` over `threads` threads.
    fn dispatch(&mut self, kernel: &str, threads: usize, args: &[Arg<'_>]) -> Result<()>;

    /// Block until every dispatch issued so far has finished.
    fn sync(&mut self) -> Result<()>;

    /// Copy a buffer's contents back to the host.
    fn read(&mut self, buffer: BufferId, out: &mut [u8]) -> Result<()>;

    /// Does this device share its memory with the host?
    ///
    /// It decides how much of that memory a launch may take. On a discrete card the VRAM
    /// is otherwise idle, so a large launch costs nothing but the allocation. On unified
    /// memory the launch buffers come out of the same pool as the resident filter and every
    /// CPU worker's working set, and a 16 GB machine holding a 7.6 GB filter has no spare
    /// gigabytes to spend on a launch that measures no faster.
    ///
    /// Defaults to discrete, which is right for every card this has data for.
    fn has_unified_memory(&self) -> bool {
        false
    }

    /// Device memory this backend can reasonably be asked for, in bytes.
    ///
    /// What "reasonably" means differs by platform and neither number is a hard limit:
    /// CUDA reports free VRAM, Metal reports the working set it recommends staying under.
    /// Both are inputs to choosing a launch size, not promises. `None` where the platform
    /// will not say, which auto-sizing treats as "assume very little".
    fn device_memory(&self) -> Option<u64> {
        None
    }

    /// Device memory allocated so far, summed over every live buffer.
    ///
    /// Asked of the backend rather than derived from the `Layout`, so it is what was
    /// actually allocated and cannot drift from `allocate` as the pipeline changes. It
    /// exists because `--gpu-batch` is the knob most worth sweeping on a new card and its
    /// only real ceiling is VRAM: the scratch scales linearly with the launch size and has
    /// to fit alongside the filter.
    fn allocated_bytes(&self) -> usize;

    /// A per-kernel timing breakdown, where the backend was asked to collect one.
    fn report(&mut self) -> Option<String> {
        None
    }

    /// Overwrite the start of a buffer from the host. Only the hit counter needs it, once
    /// per launch, which is why it takes a slice rather than a whole-buffer replacement.
    fn write(&mut self, buffer: BufferId, data: &[u8]) -> Result<()>;
}

/// Serialises whole launches across every `Gpu` in the process.
///
/// A scan opens one device and drives it from one thread, so this is never contended in
/// production and costs nothing there. It exists because the test suite is not that shape:
/// `cargo test` runs the differential tests in parallel threads, each with its own device,
/// queue and buffers, and under that load the results are wrong in ways no command buffer
/// reports -- `error()` stays nil and the hit counter comes back holding another launch's
/// total.
///
/// Measured rather than assumed: the same tests pass every time under `--test-threads=1`
/// and fail differently on each parallel run. The guard is here, around a whole launch,
/// rather than around a single dispatch -- serialising dispatches alone did not fix it,
/// which is what says the interference is between pipelines and not between submissions.
static ONE_LAUNCH_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A device with the kernels for one scope compiled onto it.
///
/// Opening is deliberately eager -- device, compile, and a smoke dispatch all happen
/// before the sweep starts. A runtime-compiled kernel set can only fail at runtime, and
/// the moment to discover that is during startup, next to the filter load, rather than
/// two hours into a detached sweep.
pub struct Gpu {
    backend: Box<dyn Backend>,
    layout: Layout,
    buffers: Buffers,
    /// How many hit records a launch can hold before it starts dropping them.
    ///
    /// A launch is expected to produce a handful: at the default scope the filter's
    /// false-positive rate puts one record per ~600 seeds. Sized far above that so
    /// overflow means something is wrong, and reported rather than silently truncated.
    hit_capacity: usize,
    /// The bound `bloom_probe` reduces every index against. Zero until `bind_filter`,
    /// which is the state in which a probe finds nothing rather than divides by zero.
    filter_bits: u64,
    /// Whether the bound filter lays its bits out in blocks, as the kernel's `blocked`.
    ///
    /// Carried rather than compiled in: the kernels are built when the device is opened,
    /// which is before the filter has been read -- see the head of `kernels/bloom.h` for why
    /// that order is worth a uniform branch per hash.
    filter_blocked: u32,
}

/// Every device allocation a launch uses, made once and reused.
///
/// Allocating per launch would be the obvious shape and is the wrong one: these total
/// hundreds of megabytes, a sweep does thousands of launches, and none of the sizes
/// change once the scope and batch size are fixed.
#[derive(Clone, Copy)]
struct Buffers {
    entropy: BufferId,
    bip39_seeds: BufferId,
    masters: BufferId,
    /// The two intermediate levels, used as a ping-pong pair.
    ///
    /// A path can now be any depth, so a level cannot have a buffer of its own the way the
    /// old fixed five-level tree did -- and a level is read while the next is written, so
    /// two are needed rather than one. Both are sized to the *widest* level any spec
    /// reaches, not the sum, because the walk reuses them between specs and between
    /// depths; sizing them to the leaf level is what once held 1.26 GB of VRAM for a
    /// buffer nothing ever wrote to.
    level_a: BufferId,
    level_b: BufferId,
    leaf_nodes: BufferId,
    comb: BufferId,
    wordlist: BufferId,
    /// Every segment's child indices, concatenated and uploaded once. A segment names its
    /// own run by an offset and a length, so one buffer serves every level of every spec.
    child_values: BufferId,
    /// Curve scratch, sized for the largest level (the leaves).
    gej: BufferId,
    prefix_products: BufferId,
    run_totals: BufferId,
    run_inverses: BufferId,
    /// The second level of the inversion scan: the run totals reduced again, so the
    /// single-threaded pass walks count/4096 elements instead of count/64.
    run_prefix2: BufferId,
    run_totals2: BufferId,
    run_inverses2: BufferId,
    zinv: BufferId,
    filter: BufferId,
    hit_count: BufferId,
    hits: BufferId,
}

/// One record `k_leaf` emits: which point of the launch, the leaf slot, the form, and the
/// hash160.
///
/// Deliberately thin. It says "look at this point" and nothing more; the phrase, the key
/// and the path all come from re-deriving that point through `derive::Deriver` on the
/// host, so nothing a user ever reads was produced by a kernel. `point` is an *offset
/// within the launch*, not an absolute point -- the kernel has no need for the base, and
/// keeping it out means a launch beyond 2^32 does not have to widen this record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hit {
    pub point: u32,
    pub leaf: u32,
    pub form: u32,
    pub hash: [u8; 20],
}

/// Bytes per hit record. Mirrored by `HIT_STRIDE` in kernels/kernels.h.
const HIT_STRIDE: usize = 32;
/// Bytes per BIP32 node. Mirrored by `NODE_WORDS` in kernels/kernels.h.
const NODE_STRIDE: usize = 20 * 4;
/// Bytes per Jacobian point. Mirrored by `GEJ_WORDS`.
const GEJ_STRIDE: usize = 28 * 4;
/// Elements per run in the batched inversion. Mirrored by `INVERT_RUN`.
const INVERT_RUN: usize = 64;

impl Buffers {
    fn ids(&self) -> Buffers {
        *self
    }
}

/// Every segment's child indices, concatenated, as the device buffer holds them.
///
/// One buffer for the whole scope rather than one per level: a segment names its own run
/// by an offset, so a spec of any depth costs one upload. The values already carry the
/// hardened bit where the segment is hardened, which is the same convention
/// `derive::derive_hardened` and `bip32_hardened` both take.
fn child_value_bytes(l: &Layout) -> Vec<u8> {
    let mut out = Vec::new();
    for spec in &l.specs {
        for seg in spec.segments() {
            for child in seg.children() {
                out.extend_from_slice(&child.to_le_bytes());
            }
        }
    }
    // Never empty: both backends refuse a zero-length buffer, and a scope with no paths at
    // all -- a privkey-only sweep -- has no segments to upload.
    if out.is_empty() {
        out.extend_from_slice(&0u32.to_le_bytes());
    }
    out
}

/// Where each segment's run starts within that buffer, indexed `[spec][depth]`.
fn segment_offsets(l: &Layout) -> Vec<Vec<usize>> {
    let mut at = 0usize;
    l.specs
        .iter()
        .map(|spec| {
            spec.segments()
                .iter()
                .map(|seg| {
                    let start = at;
                    at += seg.len();
                    start
                })
                .collect()
        })
        .collect()
}

/// Hit records a launch can hold. See `Gpu::hit_capacity`.
const HIT_CAPACITY: usize = 1 << 16;

/// How big a launch should be.
#[derive(Clone, Copy, Debug)]
pub enum Batch {
    /// Exactly this many seeds, because `--gpu-batch` said so.
    Fixed(usize),
    /// Sized from what the device has room for and what is being swept.
    Auto {
        /// Bytes the filter will occupy on the device. Subtracted from the budget: on a
        /// discrete card it is a real copy in VRAM, and on unified memory it still counts
        /// against the working set.
        filter_bytes: u64,
        /// Seeds in the range this run will sweep, so a short range does not hand the GPU
        /// the whole thing in one claim.
        seeds_in_range: u64,
    },
}

/// Device bytes the launch buffers take for `seeds` seeds, **excluding the comb table**.
///
/// Mirrors `Gpu::allocate`, and is held to it by
/// `the_estimate_matches_what_is_actually_allocated`. It exists because the batch has to be
/// chosen *before* anything is allocated, which is exactly when the real figure is not yet
/// available.
///
/// The comb is excluded because its size is the thing being chosen at that point; add it
/// with `buffer_bytes_at_window`, which every sizing path uses.
pub fn buffer_bytes(l: &Layout, seeds: usize) -> usize {
    let leaves = seeds * l.leaves_per_point();
    let runs = leaves.div_ceil(INVERT_RUN);
    let per_level =
        (seeds * l.masters_per_point() + 2 * l.level_capacity(seeds)) * NODE_STRIDE;

    seeds * 32                                  // entropy
        + seeds * l.material_sizes.len() * 64   // bip39 seeds
        + per_level                             // masters and the two level buffers
        + leaves * NODE_STRIDE                  // leaf nodes
        + leaves * GEJ_STRIDE                   // curve scratch
        + leaves * 32 * 2                       // prefix products, zinv
        + runs * 32 * 3                         // run totals, inverses, second-level prefix
        + runs.div_ceil(INVERT_RUN) * 32 * 2    // the second level's own pair
        + 2048 * WORD_STRIDE                    // the wordlist, fixed
        + HIT_CAPACITY * HIT_STRIDE // the hit buffer, fixed
}

/// Device bytes a launch takes at a stated window, comb included.
///
/// **Everything that sizes anything goes through this, and none of it may consult the
/// window in force.** `crate::crypto::ec::gpu_window` is a `OnceLock` read through `get_or_init`,
/// so *observing* it settles it -- and the sizing runs before the window has been chosen.
/// An earlier version computed this as `buffer_bytes(..) - comb_gpu_bytes() + comb_bytes_at(w)`,
/// whose arithmetic is right and whose effect was to pin the window to the default on the
/// first call, silently, making the later `set_gpu_window` a no-op. A 5070 Ti that should
/// have picked 22 reported "16-bit comb window (auto)" and nothing else looked wrong.
fn buffer_bytes_at_window(l: &Layout, seeds: usize, window: usize) -> usize {
    buffer_bytes(l, seeds) + crate::crypto::ec::comb_bytes_at(window)
}

/// The launch the window is sized against.
///
/// Throughput is flat from about here upward (see `MAX_AUTO_BATCH`), so this is the
/// smallest launch worth protecting: a window wide enough to push the batch below it would
/// be trading a measured cliff for an unmeasured gain.
const WINDOW_REFERENCE_BATCH: usize = 16_384;

/// The comb window for this device, filter and run.
///
/// **The window is a memory-for-work trade and the memory is whatever the filter left**,
/// which is why it cannot be a constant: the same card with a 7.6 GB filter and with a
/// 300 MB one has completely different room, and the right window differs by two rungs.
/// Measured on an RTX 5070 Ti, W=16 gives 114,000 seeds/s and W=20 gives 123,000.
///
/// Sized against a reference launch rather than the real one, because the real one depends
/// on the window: `auto_batch` charges the comb as a fixed cost, so choosing the window
/// from the batch and the batch from the window is circular. Requiring room for
/// `WINDOW_REFERENCE_BATCH` seeds breaks it, and errs the right way -- a launch at the floor
/// is worth as much as one well above it.
///
/// Three things stop it widening, and they are all deliberate:
///
///   * **A run too short to amortise the table.** The build is single-threaded and doubles
///     with the window -- ~6s at W=20, ~22s at 22, ~80s at 24 -- so a caller that has
///     already settled on a launch below the reference batch is not doing a throughput run
///     and does not get a throughput-sized table. This is what keeps the test suite, which
///     opens devices at a batch of 64, from building a six-gigabyte comb.
///   * **Unified memory.** The table would come out of the same pool as the resident filter
///     and every CPU worker's working set, and the M1 Pro measurements behind `GPU_WINDOW`
///     flattened at 16 anyway. It stays at the default there.
///   * **A device that will not report its memory**, which is assumed small rather than
///     guessed at, exactly as `auto_batch` does.
///
/// `KEYFORGE_GPU_WINDOW` overrides all of it. Pure arithmetic on purpose, like `auto_batch`:
/// it is the piece most worth being able to check without a GPU.
pub fn auto_window(
    l: &Layout,
    device_memory: Option<u64>,
    unified: bool,
    batch: Batch,
    provisional_batch: usize,
) -> usize {
    if let Some(w) = crate::crypto::ec::gpu_window_override() {
        return w;
    }
    let default = crate::crypto::ec::DEFAULT_GPU_WINDOW;
    // A fixed batch keeps the default window, and this is a safety property rather than a
    // policy one: `Batch::Fixed` carries no filter size, so there is nothing to subtract
    // from the card before spending gigabytes on a table. Widening on a guess of zero would
    // pick the widest rung there is and then fail to place the filter beside it.
    // `--gpu-batch` is a manual override; `KEYFORGE_GPU_WINDOW` is the matching one here.
    let Batch::Auto { filter_bytes, .. } = batch else {
        return default;
    };
    if unified || provisional_batch < WINDOW_REFERENCE_BATCH {
        return default;
    }
    let Some(total) = device_memory else {
        return default;
    };

    let budget = (total.saturating_sub(filter_bytes) as f64 * BUDGET_SHARE) as u64;
    for &w in &crate::crypto::ec::WINDOW_LADDER {
        if w <= default {
            break;
        }
        if buffer_bytes_at_window(l, WINDOW_REFERENCE_BATCH, w) as u64 <= budget {
            return w;
        }
    }
    default
}

/// The largest launch auto-sizing will pick on a discrete card.
///
/// **Swept on an RTX 5070 Ti and flat, which settles a question this comment used to leave
/// open.** Default scope, 3,000,000 seeds against the real 7.6 GB filter, medians of three:
///
/// ```text
///   batch    buffers    seeds/s
///   16,384    2.2 GB    106,626
///   32,768    4.3 GB    107,092
///   49,152    6.4 GB    107,052
///   61,440    8.0 GB    (does not run -- see below)
/// ```
///
/// 0.4% across a 3x range, which is inside the 0.8% spread of the whole sweep. The knee is
/// below 16,384, not above 32,768: the earlier 2,048 -> 32,768 measurement (35,807 ->
/// 78,976 seeds/s) was climbing out of a launch too small to fill the card, and it had
/// finished climbing long before it got here. This constant stays at 32,768 because that is
/// the middle of the flat region, not because anything above it is better.
///
/// It also means **there is nothing to spend more device memory on**, which is why
/// `BUDGET_SHARE` is one number again and why tiling the leaf level to make room for larger
/// launches is not worth writing.
///
/// 61,440 fails, and the failure is worth understanding rather than avoiding: the launch
/// buffers are allocated before the filter is bound, so 8.0 GB of buffers leaves 7.23 GB on
/// a card that has 7.62 GB of filter to place. `--gpu-batch` is a fixed override and skips
/// the budget above entirely, so it does not back off -- it fails at startup with the
/// filter's own error. The auto path subtracts `filter_bytes` first and cannot get here.
const MAX_AUTO_BATCH: usize = 32_768;

/// The same, where the device shares the host's memory.
///
/// Two reasons to stop earlier, and the second is the one that matters. Throughput stops
/// improving: an M1 Pro measures flat from about 8,192 upward, where a discrete card is
/// still climbing. And the memory is not free -- it is the same pool holding the resident
/// filter and every CPU worker's working set, so the 4.3 GB a 32,768 launch would take is
/// 4.3 GB that a 16 GB machine holding a 7.6 GB filter does not have. The README's warning
/// about swapping costing far more than it saves applies to this allocation too.
const MAX_AUTO_BATCH_UNIFIED: usize = 8_192;

/// The smallest, so a tiny device or a very wide scope still makes progress.
const MIN_AUTO_BATCH: usize = 256;

/// Share of the memory left after the filter that the launch buffers may take.
///
/// Not all of it: the device is also holding the compiled module, the command queue's own
/// allocations, and on unified memory everything else the machine is doing. Overshooting
/// here is not a slow scan, it is an allocation failure at startup.
///
/// **This was briefly split into a larger share for discrete cards, on the theory that VRAM
/// left after the filter is idle and holding back two fifths of it buys nothing. The theory
/// was right and the conclusion was wrong**: bigger launches buy nothing either, so there is
/// nothing for the extra memory to be spent on. See the sweep recorded on `MAX_AUTO_BATCH`.
/// One share, and the reserve stays where a unified machine needs it.
const BUDGET_SHARE: f64 = 0.6;

/// The fewest launches a sweep should take, which bounds how much of a short range the GPU
/// can claim at once.
///
/// This is the guard on the pathology the README records: at 16,384 against a 40,000-seed
/// range the GPU took 82% of the work in two claims and the CPU threads finished early and
/// waited, dropping the combined rate below not using the GPU at all. That is a
/// batch-versus-range problem, not a batch problem -- the same 16,384 over a full sweep is
/// nothing -- so the fix belongs here, as a ratio, rather than in a smaller constant.
///
/// Sixteen rather than something larger because this also has to leave a short `bench` a
/// batch worth measuring: the figure it reports is only useful if the launch size is one a
/// real sweep would use. Sixteen claims is already far from the two that caused the
/// trouble, and against a full sweep this bound never binds at all.
const MIN_LAUNCHES: u64 = 16;

/// Choose a launch size for this scope, device and range.
///
/// Pure arithmetic on purpose: it is the piece most worth being able to check without a
/// GPU, and the tests below run everywhere.
pub fn auto_batch(
    l: &Layout,
    device_memory: Option<u64>,
    unified: bool,
    batch: Batch,
    window: usize,
) -> usize {
    let (filter_bytes, seeds_in_range) = match batch {
        Batch::Fixed(n) => return n.max(1),
        Batch::Auto {
            filter_bytes,
            seeds_in_range,
        } => (filter_bytes, seeds_in_range),
    };
    let ceiling = if unified {
        MAX_AUTO_BATCH_UNIFIED
    } else {
        MAX_AUTO_BATCH
    };

    // What memory allows. A device that will not say goes to the floor rather than
    // guessing high: a wrong guess upward fails the run at startup.
    let by_memory = match device_memory {
        Some(total) => {
            let budget = (total.saturating_sub(filter_bytes) as f64 * BUDGET_SHARE) as u64;
            // Per-seed cost, taken as the slope between two sizes so the fixed buffers --
            // the comb, the wordlist, the hit records -- are not charged per seed.
            let fixed = buffer_bytes_at_window(l, 0, window) as u64;
            let per_seed = (buffer_bytes_at_window(l, 1024, window) as u64 - fixed)
                .div_ceil(1024)
                .max(1);
            (budget.saturating_sub(fixed) / per_seed) as usize
        }
        None => MIN_AUTO_BATCH,
    };

    // What the range allows, so the GPU cannot swallow a short sweep in a couple of claims.
    let by_range = (seeds_in_range / MIN_LAUNCHES).max(1) as usize;

    by_memory
        .min(by_range)
        .clamp(MIN_AUTO_BATCH, ceiling)
        // Never more than the range itself, which is what a `bench` of a few thousand
        // seeds or a very short `scan` asks for.
        .min(seeds_in_range.max(1) as usize)
}

impl Gpu {
    pub fn open(
        scope: &crate::scan::derive::Scope,
        vuln: &dyn crate::vuln::Vulnerability,
        offsets: &[usize],
        batch: Batch,
    ) -> Result<Self> {
        trace("Gpu::open waiting for the launch lock");
        let _serial = ONE_LAUNCH_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        trace("opening a device");
        let mut backend = open()?;
        trace(&format!("device open: {}", backend.name()));

        // The batch is settled before the layout, because the layout's capacity *is* the
        // batch. Auto-sizing needs the device open to ask it about memory, and nothing
        // else -- the source is assembled from the scope, not the capacity.
        // The window first, then the batch, because the batch is charged for the table the
        // window chooses. A provisional batch at the default window breaks the circle: it
        // says whether this is a throughput run at all, which is the only thing the window
        // needs from it.
        let unit = Layout::new(scope, 1);
        let provisional = auto_batch(
            &unit,
            backend.device_memory(),
            backend.has_unified_memory(),
            batch,
            crate::crypto::ec::DEFAULT_GPU_WINDOW,
        );
        let window = auto_window(
            &unit,
            backend.device_memory(),
            backend.has_unified_memory(),
            batch,
            provisional,
        );
        let window = crate::crypto::ec::set_gpu_window(window);
        trace(&format!("comb window {window}"));

        let capacity = auto_batch(
            &unit,
            backend.device_memory(),
            backend.has_unified_memory(),
            batch,
            window,
        );
        let layout = Layout::new(scope, capacity);
        trace(&format!("launch capacity {capacity} seeds"));

        compile(
            &mut *backend,
            &source::assemble(&layout, vuln, offsets, source::dialect()),
        )?;

        trace("allocating buffers");
        let hit_capacity = HIT_CAPACITY;
        let buffers = Self::allocate(&mut *backend, &layout, hit_capacity)?;
        let mut gpu = Self {
            backend,
            layout,
            buffers,
            hit_capacity,
            filter_bits: 0,
            filter_blocked: 0,
        };
        trace("smoke dispatch");
        gpu.smoke()?;
        trace("Gpu::open done");
        Ok(gpu)
    }

    fn allocate(
        backend: &mut dyn Backend,
        l: &Layout,
        hit_capacity: usize,
    ) -> Result<Buffers> {
        let s = l.capacity;
        let leaves = l.leaves(l.capacity);
        let runs = leaves.div_ceil(INVERT_RUN);
        let level = l.level_capacity(l.capacity);

        Ok(Buffers {
            entropy: backend.buffer(s * 32)?,
            bip39_seeds: backend.buffer(s * l.material_sizes.len() * 64)?,
            masters: backend.buffer(s * l.masters_per_point().max(1) * NODE_STRIDE)?,
            level_a: backend.buffer(level.max(1) * NODE_STRIDE)?,
            level_b: backend.buffer(level.max(1) * NODE_STRIDE)?,
            leaf_nodes: backend.buffer(leaves * NODE_STRIDE)?,
            comb: backend.buffer_from(&crate::crypto::ec::comb_for_gpu())?,
            wordlist: backend.buffer_from(&wordlist())?,
            child_values: backend.buffer_from(&child_value_bytes(l))?,
            gej: backend.buffer(leaves * GEJ_STRIDE)?,
            prefix_products: backend.buffer(leaves * 32)?,
            run_totals: backend.buffer(runs * 32)?,
            run_inverses: backend.buffer(runs * 32)?,
            run_prefix2: backend.buffer(runs * 32)?,
            run_totals2: backend.buffer(runs.div_ceil(INVERT_RUN) * 32)?,
            run_inverses2: backend.buffer(runs.div_ceil(INVERT_RUN) * 32)?,
            zinv: backend.buffer(leaves * 32)?,
            // Replaced by the real filter in `bind_filter`; a launch before that would
            // probe an empty one and simply find nothing.
            filter: backend.buffer(8)?,
            hit_count: backend.buffer(4)?,
            hits: backend.buffer(hit_capacity * HIT_STRIDE)?,
        })
    }

    /// Expose the filter to the device, without copying it where the platform allows.
    ///
    /// On unified memory this is the whole reason `bloom::Words` is page-aligned: one set
    /// of pages serves the GPU and every CPU worker at once, which on a 16 GB machine
    /// holding a 7.6 GB filter is the difference between working and swapping.
    pub fn bind_filter(&mut self, filter: &crate::target::bloom::BloomFilter) -> Result<()> {
        // Under the same guard as a launch: this allocates on the device, and the whole
        // point of ONE_LAUNCH_AT_A_TIME is that a second device doing anything while one is
        // working returns wrong answers without reporting an error.
        let _serial = ONE_LAUNCH_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let (ptr, bytes) = filter.raw_words();
        // SAFETY: `bloom::Words` allocates page-aligned and rounds its length up to a
        // whole page, and the filter outlives this `Gpu` -- both are owned by `run_scan`
        // for the length of the sweep, with the filter declared first.
        self.buffers.filter = unsafe { self.backend.buffer_no_copy(ptr, bytes)? };
        self.filter_bits = filter.bit_count();
        // Bound together with the bits, because the two describe the same file: a launch
        // that had one without the other would probe the right array with the wrong
        // arithmetic and report nothing, which is what a filter holding nothing looks like.
        self.filter_blocked = u32::from(filter.layout() == crate::target::bloom::BitLayout::Blocked);
        Ok(())
    }

    pub fn name(&self) -> String {
        self.backend.name()
    }

    /// The kernel compiler, when the backend chose one. See `Backend::compiler`.
    pub fn compiler(&self) -> Option<String> {
        self.backend.compiler()
    }

    /// Where a launch's time went, per kernel, when profiling is on.
    pub fn report(&mut self) -> Option<String> {
        self.backend.report()
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Device memory the launch buffers occupy, before the filter is bound.
    ///
    /// Reported in the banner because it is the figure `--gpu-batch` is really bounded by,
    /// and the only one an operator cannot work out from the outside. It scales linearly
    /// with the launch size, so doubling the batch doubles this.
    pub fn scratch_bytes(&self) -> usize {
        self.backend.allocated_bytes()
    }

    /// Walk `seeds` seeds starting at `base`, returning everything the filter let through.
    ///
    /// `seeds` may be short of a full batch -- the last launch of a range usually is --
    /// and every kernel is dispatched over the actual count rather than the capacity.
    ///
    /// One launch walks one `draw`: the whole grid shares a distribution and an offset,
    /// so `k_entropy`'s branch is uniform and a claim spanning several draws is several
    /// launches. See `launch_claims`.
    pub fn run(&mut self, base: u128, seeds: usize, stream: usize) -> Result<Vec<Hit>> {
        anyhow::ensure!(
            seeds <= self.layout.capacity,
            "launch of {seeds} seeds exceeds the {} this device was sized for",
            self.layout.capacity
        );
        if seeds == 0 {
            return Ok(Vec::new());
        }
        trace("Gpu::run waiting for the launch lock");
        let _serial = ONE_LAUNCH_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        trace(&format!("launch of {seeds} points from {base} stream {stream}"));
        let l = self.layout.clone();
        // Copied rather than borrowed: `public_keys` takes `&mut self`, and a `BufferId`
        // is a `usize`, so there is nothing to gain from holding the borrow.
        let b = self.buffers.ids();

        let u32s = |v: &[u32]| -> Vec<[u8; 4]> { v.iter().map(|x| x.to_le_bytes()).collect() };
        let n = u32s(&[
            base as u32,
            (base >> 32) as u32,
            stream as u32,
            seeds as u32,
            (seeds * l.material_sizes.len()) as u32,
            (seeds * l.masters_per_point()) as u32,
            self.hit_capacity as u32,
            (seeds * l.raw_sizes.len()) as u32,
        ]);
        let (base_lo, base_hi, stream_a, seeds_a, pbkdf2_n, masters_n, cap_a, raw_n) = (
            &n[0], &n[1], &n[2], &n[3], &n[4], &n[5], &n[6], &n[7],
        );

        // Clear the hit counter for this launch.
        let zero = 0u32.to_le_bytes();
        self.backend.write(b.hit_count, &zero)?;

        self.backend.dispatch(
            "k_entropy",
            seeds,
            &[
                Arg::Buffer(b.entropy),
                Arg::Scalar(base_lo),
                Arg::Scalar(base_hi),
                Arg::Scalar(stream_a),
                Arg::Scalar(seeds_a),
            ],
        )?;

        if l.tree_routes.contains(&crate::scan::derive::Route::Bip39) {
            self.backend.dispatch(
                "k_pbkdf2",
                seeds * l.material_sizes.len(),
                &[
                    Arg::Buffer(b.entropy),
                    Arg::Buffer(b.bip39_seeds),
                    Arg::Buffer(b.wordlist),
                    Arg::Scalar(pbkdf2_n),
                ],
            )?;
        }

        if l.masters_per_point() > 0 {
            self.backend.dispatch(
                "k_master",
                seeds * l.masters_per_point(),
                &[
                    Arg::Buffer(b.entropy),
                    Arg::Buffer(b.bip39_seeds),
                    Arg::Buffer(b.masters),
                    Arg::Scalar(masters_n),
                ],
            )?;
        }

        // Every spec together, one round at a time -- see `Layout::rounds`.
        //
        // A hardened segment needs no public key, so it is one dispatch and no EC work at
        // all. A normal one needs its parents' keys: a k*G over the level, one batched
        // inversion, and a CKD. The specs walk in lockstep and share the level buffer, so
        // the k*G and the inversion happen **once per round** rather than once per spec
        // per segment -- eight `public_keys` per launch on the default scope became two.
        //
        // The two level buffers are swapped between rounds. A spec's last segment writes
        // straight into its region of the leaf array, so every leaf ends up where
        // `Layout::point_of` says it is; right-alignment puts every spec's last segment on
        // the last round, so they all reach the leaves together.
        let offsets = segment_offsets(&l);
        let rounds = l.rounds();
        for (index, round) in rounds.iter().enumerate() {
            let (parents, next) = if index % 2 == 0 {
                (b.level_a, b.level_b)
            } else {
                (b.level_b, b.level_a)
            };

            // A spec joining this round still has its parents in the master array. They
            // have to be in the shared buffer, at this spec's offset, before one
            // `public_keys` can cover the round.
            for step in round.steps.iter().filter(|s| s.joins) {
                let args = u32s(&[(step.width * seeds) as u32, (step.at * seeds) as u32]);
                self.backend.dispatch(
                    "k_copy_nodes",
                    step.width * seeds,
                    &[
                        Arg::Buffer(b.masters),
                        Arg::Buffer(parents),
                        Arg::Scalar(&args[0]),
                        Arg::Scalar(&args[1]),
                    ],
                )?;
            }

            // One batch for the whole round, over the prefix the normal steps occupy.
            if round.normal_width > 0 {
                self.public_keys(parents, round.normal_width * seeds)?;
            }

            for step in &round.steps {
                let spec = &l.specs[step.spec];
                let seg = &spec.segments()[step.segment];
                let out_base = if step.last {
                    let region = l
                        .regions
                        .iter()
                        .find(|r| r.spec == Some(step.spec))
                        .expect("a walked spec has a leaf region");
                    region.base_per_point * seeds
                } else {
                    // Where this spec's children sit in the next round's buffer, which is
                    // where it will read them from.
                    rounds[index + 1]
                        .steps
                        .iter()
                        .find(|s| s.spec == step.spec)
                        .expect("a spec that has not finished acts again next round")
                        .at
                        * seeds
                };
                let out = if step.last { b.leaf_nodes } else { next };
                let args = u32s(&[
                    (step.width * seeds) as u32,
                    seg.len() as u32,
                    offsets[step.spec][step.segment] as u32,
                    out_base as u32,
                    (step.at * seeds) as u32,
                ]);
                let (count, per_parent, values_at, out_base, in_base) =
                    (&args[0], &args[1], &args[2], &args[3], &args[4]);

                if step.hardened {
                    self.backend.dispatch(
                        "k_hardened_level",
                        step.width * seeds,
                        &[
                            Arg::Buffer(parents),
                            Arg::Buffer(out),
                            Arg::Scalar(count),
                            Arg::Scalar(per_parent),
                            Arg::Buffer(b.child_values),
                            Arg::Scalar(values_at),
                            Arg::Scalar(out_base),
                            Arg::Scalar(in_base),
                        ],
                    )?;
                } else {
                    self.backend.dispatch(
                        "k_ckd_normal",
                        step.width * seeds,
                        &[
                            Arg::Buffer(parents),
                            Arg::Buffer(b.gej),
                            Arg::Buffer(b.zinv),
                            Arg::Buffer(out),
                            Arg::Scalar(count),
                            Arg::Scalar(per_parent),
                            Arg::Buffer(b.child_values),
                            Arg::Scalar(values_at),
                            Arg::Scalar(out_base),
                            Arg::Scalar(in_base),
                        ],
                    )?;
                }
            }
        }

        // A spec with no segments at all -- `m`, the master node itself -- has no round to
        // take: its masters are its leaves and only need moving into place.
        for (spec_index, spec) in l.specs.iter().enumerate() {
            if !spec.segments().is_empty() {
                continue;
            }
            let Some(region) = l.regions.iter().find(|r| r.spec == Some(spec_index)) else {
                continue;
            };
            let n = seeds * l.masters_per_point();
            let args = u32s(&[n as u32, (region.base_per_point * seeds) as u32]);
            self.backend.dispatch(
                "k_copy_nodes",
                n,
                &[
                    Arg::Buffer(b.masters),
                    Arg::Buffer(b.leaf_nodes),
                    Arg::Scalar(&args[0]),
                    Arg::Scalar(&args[1]),
                ],
            )?;
        }

        // Keys with no derivation join at the leaf level and share its inversion, written
        // into their own region of the leaf array.
        if !l.raw_sizes.is_empty() {
            let raw_region = l
                .regions
                .iter()
                .find(|r| r.spec.is_none())
                .expect("a raw region exists when raw sizes do");
            let raw_base = u32s(&[(raw_region.base_per_point * seeds) as u32]);
            self.backend.dispatch(
                "k_raw",
                seeds * l.raw_sizes.len(),
                &[
                    Arg::Buffer(b.entropy),
                    Arg::Buffer(b.leaf_nodes),
                    Arg::Scalar(raw_n),
                    Arg::Scalar(&raw_base[0]),
                ],
            )?;
        }

        let leaves = seeds * l.leaves_per_point();
        let leaves_bytes = (leaves as u32).to_le_bytes();
        let leaves_n = &leaves_bytes;
        self.public_keys(b.leaf_nodes, leaves)?;

        let bits = self.filter_bits.to_le_bytes();
        let blocked = self.filter_blocked.to_le_bytes();
        self.backend.dispatch(
            "k_leaf",
            leaves,
            &[
                Arg::Buffer(b.gej),
                Arg::Buffer(b.zinv),
                Arg::Buffer(b.filter),
                Arg::Scalar(&bits),
                Arg::Buffer(b.hit_count),
                Arg::Buffer(b.hits),
                Arg::Scalar(leaves_n),
                Arg::Scalar(cap_a),
                Arg::Scalar(seeds_a),
                Arg::Scalar(&blocked),
            ],
        )?;
        self.backend.sync()?;
        trace("launch complete");

        self.collect_hits()
    }

    /// k*G for a whole level, then one batched inversion, leaving `gej` and `zinv` ready
    /// for whatever consumes the affine points.
    fn public_keys(&mut self, nodes: BufferId, count: usize) -> Result<()> {
        let b = self.buffers;
        let runs = count.div_ceil(INVERT_RUN);
        let runs2 = runs.div_ceil(INVERT_RUN);
        let (n, r, r2) = (
            (count as u32).to_le_bytes(),
            (runs as u32).to_le_bytes(),
            (runs2 as u32).to_le_bytes(),
        );

        self.backend.dispatch(
            "k_kmul",
            count,
            &[
                Arg::Buffer(nodes),
                Arg::Buffer(b.comb),
                Arg::Buffer(b.gej),
                Arg::Scalar(&n),
            ],
        )?;
        self.backend.dispatch(
            "k_invert_a",
            runs,
            &[
                Arg::Buffer(b.gej),
                Arg::Buffer(b.prefix_products),
                Arg::Buffer(b.run_totals),
                Arg::Scalar(&n),
            ],
        )?;
        // Second level: reduce the run totals again before anything serial touches them.
        self.backend.dispatch(
            "k_invert_fe_a",
            runs2,
            &[
                Arg::Buffer(b.run_totals),
                Arg::Buffer(b.run_prefix2),
                Arg::Buffer(b.run_totals2),
                Arg::Scalar(&r),
            ],
        )?;
        self.backend.dispatch(
            "k_invert_b",
            1,
            &[
                Arg::Buffer(b.run_totals2),
                Arg::Buffer(b.run_inverses2),
                Arg::Scalar(&r2),
            ],
        )?;
        self.backend.dispatch(
            "k_invert_fe_c",
            runs2,
            &[
                Arg::Buffer(b.run_totals),
                Arg::Buffer(b.run_prefix2),
                Arg::Buffer(b.run_inverses2),
                Arg::Buffer(b.run_inverses),
                Arg::Scalar(&r),
            ],
        )?;
        self.backend.dispatch(
            "k_invert_c",
            runs,
            &[
                Arg::Buffer(b.gej),
                Arg::Buffer(b.prefix_products),
                Arg::Buffer(b.run_inverses),
                Arg::Buffer(b.zinv),
                Arg::Scalar(&n),
            ],
        )?;
        Ok(())
    }

    /// The entropy buffer as the last launch left it.
    ///
    /// Only the differential test uses this, to localise a mismatch to the first stage
    /// rather than to "somewhere in the pipeline".
    #[cfg(all(test, feature = "gpu"))]
    pub fn read_entropy(&mut self, seeds: usize) -> Result<Vec<u8>> {
        let mut out = vec![0u8; seeds * 32];
        self.backend.read(self.buffers.entropy, &mut out)?;
        Ok(out)
    }

    /// The BIP39 seed buffer as the last launch left it.
    #[cfg(all(test, feature = "gpu"))]
    pub fn read_bip39_seeds(&mut self, seeds: usize) -> Result<Vec<u8>> {
        let mut out = vec![0u8; seeds * self.layout.material_sizes.len() * 64];
        self.backend.read(self.buffers.bip39_seeds, &mut out)?;
        Ok(out)
    }

    fn collect_hits(&mut self) -> Result<Vec<Hit>> {
        let mut count = [0u8; 4];
        self.backend.read(self.buffers.hit_count, &mut count)?;
        let count = u32::from_le_bytes(count) as usize;
        if count > self.hit_capacity {
            anyhow::bail!(
                "a launch produced {count} filter survivors but the buffer holds \
                 {}. That is far past the false-positive rate any real filter has, so \
                 something is wrong with the filter or the kernels rather than with this \
                 limit.",
                self.hit_capacity
            );
        }
        let mut raw = vec![0u8; count * HIT_STRIDE];
        if count > 0 {
            self.backend.read(self.buffers.hits, &mut raw)?;
        }
        Ok(raw
            .chunks_exact(HIT_STRIDE)
            .map(|r| Hit {
                point: u32::from_le_bytes(r[0..4].try_into().unwrap()),
                leaf: u32::from_le_bytes(r[4..8].try_into().unwrap()),
                form: r[8] as u32,
                hash: r[12..32].try_into().unwrap(),
            })
            .collect())
    }

    /// Dispatch the smoke kernel and check it read the scope it was compiled with.
    ///
    /// This is not ceremony: it is the one cheap check that the whole chain -- source
    /// assembly, runtime compile, dispatch, readback -- is intact, and that the `#define`s
    /// the kernels index arrays with are the ones this `Scope` asked for. Everything else
    /// in the GPU path assumes all four of those work.
    fn smoke(&mut self) -> Result<()> {
        let out = self.backend.buffer(4)?;
        self.backend
            .dispatch("smoke", 1, &[Arg::Buffer(out)])?;
        self.backend.sync()?;

        let mut got = [0u8; 4];
        self.backend.read(out, &mut got)?;

        let l = &self.layout;
        let want: u32 = l.material_sizes.iter().map(|s| *s as u32).sum::<u32>()
            + l.regions.iter().map(|r| r.per_point as u32).sum::<u32>()
            + l.hashes_per_point() as u32 * 1_000_000;
        let got = u32::from_le_bytes(got);
        if got != want {
            anyhow::bail!(
                "the smoke kernel returned {got}, expected {want}. The kernels compiled \
                 but did not see the scope they were built for, so nothing below this \
                 can be trusted."
            );
        }
        Ok(())
    }
}

/// Print a timestamped stage to stderr when KEYFORGE_GPU_TRACE is set.
///
/// Opening a device is four steps -- find the libraries, assemble the source, compile it,
/// run a smoke dispatch -- and when one of them does not return, which one matters more
/// than anything else. Without this the only visible symptom is "the test has been running
/// for over 60 seconds", which is equally consistent with a slow compiler, a wedged
/// dispatch, and a kernel looping forever.
pub fn trace(stage: &str) {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("KEYFORGE_GPU_TRACE").is_some()) {
        return;
    }
    let start = START.get_or_init(std::time::Instant::now);
    eprintln!("[gpu +{:>7.2}s] {stage}", start.elapsed().as_secs_f64());
}

/// Compile a translation unit, turning a missing kernel compiler into `NoDevice`.
///
/// Guarded for the same reason `open` is: `cudarc` dlopens NVRTC on first use and panics
/// out of a lazy static if the library is absent, which no amount of matching on a `Result`
/// will catch. A missing compiler is reported as "no usable device" so a machine without
/// one skips the GPU tests with a single line instead of failing every one of them with
/// the same stack trace.
///
/// Every caller goes through here. An earlier version guarded only `Gpu::open` and left
/// the parity harness calling `Backend::compile` directly, which meant half the suite
/// still failed with the raw panic.
pub fn compile(backend: &mut dyn Backend, source: &str) -> Result<()> {
    trace(&format!("compiling {} bytes of kernel source", source.len()));
    let compiled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        backend.compile(source)
    }));
    match compiled {
        Ok(result) => {
            trace("compile returned");
            result
        }
        Err(panic) => {
            let detail = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("(no message)");
            Err(anyhow::Error::new(NoDevice(format!(
                "the kernel compiler could not be loaded: {detail}\n\n\
                 On NVIDIA this is libnvrtc, which ships with the CUDA *toolkit* -- the \
                 driver alone is not enough, and installs only libcuda. Install the \
                 toolkit or just its compiler (`cuda-nvrtc-12-x` on Debian and Ubuntu, \
                 `nvidia-cuda-toolkit` for the lot), or add its directory to \
                 LD_LIBRARY_PATH if it is already present:\n    \
                 find / -name 'libnvrtc.so*' 2>/dev/null\n    \
                 export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
            ))))
        }
    }
}

/// Open the best available device, or explain why there is none.
///
/// The error is written for someone who just typed `--gpu` and expected it to work, so it
/// names the feature to rebuild with rather than reporting an absence.
pub fn open() -> Result<Box<dyn Backend>> {
    // Backend construction is wrapped because it is not all our code: `cudarc` dlopens the
    // driver, and a missing or mismatched libcuda surfaces as a panic from inside the
    // loader rather than as an error we could return. Unwinding out of here would abort
    // every test that merely asked whether a GPU exists, which is exactly what a machine
    // without one does.
    #[cfg(any(feature = "metal", feature = "cuda"))]
    fn guarded<T: Backend + 'static>(
        what: &str,
        open: impl FnOnce() -> Result<T> + std::panic::UnwindSafe,
    ) -> Result<Box<dyn Backend>> {
        match std::panic::catch_unwind(open) {
            Ok(Ok(backend)) => Ok(Box::new(backend)),
            Ok(Err(e)) => Err(e),
            Err(panic) => {
                let detail = panic
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("(no message)");
                Err(anyhow::Error::new(NoDevice(format!(
                    "the {what} runtime panicked while opening a device: {detail}. \
                     That usually means its driver library is missing or does not match \
                     the version this build expects."
                ))))
            }
        }
    }

    #[cfg(feature = "metal")]
    {
        guarded("Metal", metal::Metal::open)
    }

    #[cfg(all(feature = "cuda", not(feature = "metal")))]
    {
        guarded("CUDA", cuda::Cuda::open)
    }

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    {
        // `NoDevice`, not a plain error: a build with no backend has no device by
        // definition, so the GPU tests skip rather than fail on a default `cargo test`.
        Err(anyhow::Error::new(NoDevice(
            "this build has no GPU backend compiled in.\n\
             Rebuild with one:\n    \
             cargo build --release --features metal    (Apple Silicon and Intel Macs)\n    \
             cargo build --release --features cuda     (NVIDIA, driver 580 or newer)\n\n\
             `nvidia-smi` reports the driver version."
                .to_string(),
        )))
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use crate::scan::derive::Scope;

    /// The kernels compile, dispatch, and see the scope they were built for.
    ///
    /// Runtime compilation buys a lot -- no GPU toolchain at build time, and the scope
    /// baked in as constants -- at the price of syntax errors that a build cannot catch.
    /// This is what buys that back: `cargo test --features metal` fails on a kernel that
    /// does not compile, in the same place every other mistake surfaces.
    ///
    /// Skips rather than fails where there is no device, so a headless CI box running the
    /// suite does not report a machine's shape as a defect. Watch for the message if you
    /// expected this to run.
    #[test]
    fn the_kernel_source_compiles_and_sees_its_scope() {
        // A default scope and a narrowed one: the constants differ, and a kernel that
        // only compiles for one of them is a kernel that has hard-coded something.
        for scope in [
            Scope::default(),
            // A narrowed scope, and one whose path shape the old fixed pipeline could
            // not express at all: hardened levels below normal ones, and a spec with no
            // segments. A kernel set that only compiles for the default has hard-coded
            // something.
            Scope {
                material_sizes: vec![32],
                routes: vec![crate::scan::derive::Route::Bip39],
                paths: vec![crate::wallet::path::PathSpec::parse("m/0'/1/{0..2}").unwrap()],
                forms: vec![crate::wallet::address::HashForm::Compressed],
            },
            Scope {
                material_sizes: vec![16],
                routes: vec![crate::scan::derive::Route::Bip32Seed],
                paths: vec![
                    crate::wallet::path::PathSpec::parse("m").unwrap(),
                    crate::wallet::path::PathSpec::parse("m/{0,1}/{2,3}'").unwrap(),
                ],
                forms: vec![crate::wallet::address::HashForm::Compressed],
            },
        ] {
            match Gpu::open(&scope, &crate::vuln::mt19937::MilkSad, &[0], Batch::Fixed(64)) {
                Ok(gpu) => println!("compiled and smoke-tested on {}", gpu.name()),
                Err(e) if is_unavailable(&e) => {
                    println!("skipping: {e}");
                    return;
                }
                Err(e) => panic!("{e}"),
            }
        }

        // Every plugin that ships a kernel, at its own default scope.
        //
        // Compiling one is not the same as compiling another: the scope a plugin narrows
        // to decides which kernels survive the preprocessor, and `low-int` and
        // `repeated-byte` are the degenerate case -- privkey-only, so there is no BIP39,
        // no tree and no path, several constant arrays are empty and several kernels are
        // compiled out entirely. Naming plugins here individually is what would let the
        // next one be added without ever being built.
        for v in crate::vuln::registry() {
            if v.kernel().is_none() {
                continue;
            }
            let mut scope = Scope::default();
            v.defaults().apply(&mut scope);
            match Gpu::open(&scope, *v, &[0], Batch::Fixed(64)) {
                Ok(gpu) => println!("compiled {} and smoke-tested on {}", v.id(), gpu.name()),
                Err(e) if is_unavailable(&e) => {
                    println!("skipping: {e}");
                    return;
                }
                Err(e) => panic!("{}: {e}", v.id()),
            }
        }
    }

    /// The estimate auto-sizing chooses against has to match what then gets allocated,
    /// or every budget it computes is against the wrong number.
    ///
    /// Allowed to run under: the backends round each allocation up to a page, and there
    /// are twenty of them. Not allowed to run over, and not allowed to drift far under --
    /// which is what would happen if a buffer were added to `allocate` and not here.
    #[test]
    fn the_estimate_matches_what_is_actually_allocated() {
        let scope = Scope::default();
        let gpu = match Gpu::open(&scope, &crate::vuln::mt19937::MilkSad, &[0], Batch::Fixed(2048)) {
            Ok(gpu) => gpu,
            Err(e) if is_unavailable(&e) => {
                println!("skipping: {e}");
                return;
            }
            Err(e) => panic!("{e}"),
        };
        // At the window actually in force: `buffer_bytes` excludes the comb, and the
        // allocation very much includes it.
        let predicted = buffer_bytes_at_window(gpu.layout(), 2048, crate::crypto::ec::gpu_window());
        let actual = gpu.scratch_bytes();
        assert!(
            predicted <= actual && actual <= predicted + predicted / 20,
            "estimated {predicted} bytes, allocated {actual}"
        );
    }
}

#[cfg(test)]
mod sizing_tests {
    use super::*;
    use crate::scan::derive::Scope;

    fn layout() -> Layout {
        Layout::new(&Scope::default(), 1)
    }

    /// The whole point of sizing the window: the same card with a different filter has
    /// different room, and gets a different table.
    #[test]
    fn a_smaller_filter_buys_a_wider_comb() {
        let l = layout();
        let w = |filter: u64| {
            auto_window(
                &l,
                Some(16 << 30),
                false,
                Batch::Auto { filter_bytes: filter, seeds_in_range: 1 << 32 },
                32_768,
            )
        };
        let big = w(7_600_000_000);
        let small = w(300_000_000);
        assert!(
            small > big,
            "a 300 MB filter should afford a wider window than a 7.6 GB one, got {small} \
             and {big}"
        );
        // And what it picks has to actually fit beside the filter and a floor-sized launch.
        for filter in [7_600_000_000u64, 3_000_000_000, 300_000_000] {
            let chosen = w(filter);
            let need = buffer_bytes_at_window(&l, WINDOW_REFERENCE_BATCH, chosen) as u64;
            let budget = ((16u64 << 30) - filter) as f64 * BUDGET_SHARE;
            assert!(
                need as f64 <= budget,
                "window {chosen} for a {filter}-byte filter needs {need} against {budget:.0}"
            );
        }
    }

    /// The three things that hold it at the default, each for its own reason.
    #[test]
    fn the_comb_window_does_not_widen_where_it_should_not() {
        let l = layout();
        let d = crate::crypto::ec::DEFAULT_GPU_WINDOW;
        // Unified memory: the table would come out of the pool the filter lives in.
        assert_eq!(auto_window(&l, Some(16 << 30), true, Batch::Auto { filter_bytes: 300_000_000, seeds_in_range: 1 << 32 }, 32_768), d);
        // A run already settled below the reference batch is not a throughput run.
        assert_eq!(auto_window(&l, Some(16 << 30), false, Batch::Auto { filter_bytes: 300_000_000, seeds_in_range: 1 << 32 }, 64), d);
        // A fixed batch carries no filter size, so there is nothing safe to widen against.
        assert_eq!(
            auto_window(&l, Some(16 << 30), false, Batch::Fixed(32_768), 32_768),
            d
        );
        // A device that will not report its memory is assumed small.
        assert_eq!(auto_window(&l, None, false, Batch::Auto { filter_bytes: 300_000_000, seeds_in_range: 1 << 32 }, 32_768), d);
        // A filter that fills the card leaves nothing to widen into.
        assert_eq!(auto_window(&l, Some(16 << 30), false, Batch::Auto { filter_bytes: 16_000_000_000, seeds_in_range: 1 << 32 }, 32_768), d);
    }

    /// **The sizing path must never consult the window in force.**
    ///
    /// `ec::gpu_window` is a `OnceLock` read through `get_or_init`, so *observing* it
    /// settles it -- and every sizing call happens before the window has been chosen. An
    /// earlier version of `buffer_bytes_at_window` computed itself as
    /// `buffer_bytes(..) - comb_gpu_bytes() + comb_bytes_at(w)`. The arithmetic was right
    /// and the effect was to pin the window to the default on the first call, silently,
    /// which made the later `set_gpu_window` a no-op: a 5070 Ti that should have chosen 22
    /// printed "16-bit comb window (auto)" and nothing else looked wrong.
    ///
    /// Pinned as an arithmetic property so it holds whatever order the tests run in.
    #[test]
    fn sizing_does_not_depend_on_the_window_in_force() {
        let l = layout();
        // `buffer_bytes` excludes the comb, so it cannot vary with the window at all.
        let base = buffer_bytes(&l, 4096);
        for &w in &crate::crypto::ec::WINDOW_LADDER {
            assert_eq!(
                buffer_bytes_at_window(&l, 4096, w),
                base + crate::crypto::ec::comb_bytes_at(w),
                "window {w} must be charged arithmetically, not by asking what is in force"
            );
        }
        // And the rungs have to differ, or there is nothing for the sizing to choose.
        assert!(buffer_bytes_at_window(&l, 4096, 22) > buffer_bytes_at_window(&l, 4096, 16));
    }

    /// It may only pick a rung that buys a row. 21, 23 and 25 double the table for nothing.
    #[test]
    fn the_window_ladder_only_holds_rungs_that_buy_a_row() {
        let rows = |w: usize| 256usize.div_ceil(w) + 1;
        let mut seen = Vec::new();
        for &w in &crate::crypto::ec::WINDOW_LADDER {
            assert!(
                !seen.contains(&rows(w)),
                "window {w} has {} rows, which the ladder already reaches more cheaply",
                rows(w)
            );
            seen.push(rows(w));
        }
        assert!(
            crate::crypto::ec::WINDOW_LADDER.contains(&crate::crypto::ec::DEFAULT_GPU_WINDOW),
            "the ladder has to be able to return the default"
        );
    }

    #[test]
    fn a_large_card_is_given_the_largest_batch_there_is_evidence_for() {
        let l = layout();
        let n = auto_batch(
            &l,
            Some(16 << 30),
            false,
            Batch::Auto {
                filter_bytes: 7_600_000_000,
                seeds_in_range: 1 << 32,
            },
            crate::crypto::ec::DEFAULT_GPU_WINDOW,
        );
        assert_eq!(n, MAX_AUTO_BATCH);
        // And what that costs has to be inside the budget it was chosen from, or the
        // scan allocates past the card on the very first launch.
        let budget = ((16u64 << 30) - 7_600_000_000) as f64 * BUDGET_SHARE;
        assert!(
            (buffer_bytes(&l, n) as f64) < budget,
            "{} bytes against a {budget:.0} budget",
            buffer_bytes(&l, n)
        );
    }

    /// A filter that fills the card leaves little for the buffers, and the batch has to
    /// come down rather than the run failing to allocate.
    #[test]
    fn a_filter_that_crowds_the_device_shrinks_the_batch() {
        let l = layout();
        let roomy = auto_batch(
            &l,
            Some(16 << 30),
            false,
            Batch::Auto {
                filter_bytes: 0,
                seeds_in_range: 1 << 32,
            },
            crate::crypto::ec::DEFAULT_GPU_WINDOW,
        );
        let tight = auto_batch(
            &l,
            Some(16 << 30),
            false,
            Batch::Auto {
                filter_bytes: 15 << 30,
                seeds_in_range: 1 << 32,
            },
            crate::crypto::ec::DEFAULT_GPU_WINDOW,
        );
        assert!(tight < roomy, "{tight} should be below {roomy}");
        assert!(tight >= MIN_AUTO_BATCH);
    }

    /// The pathology the old constant was guarding against: a launch that is a large
    /// fraction of the range lets the GPU take the whole sweep in a couple of claims
    /// while the CPU threads sit idle. Sizing has to see the range, not just the device.
    #[test]
    fn a_short_range_is_not_swallowed_in_a_couple_of_claims() {
        let l = layout();
        let n = auto_batch(
            &l,
            Some(64 << 30),
            false,
            Batch::Auto {
                filter_bytes: 0,
                seeds_in_range: 40_000,
            },
            crate::crypto::ec::DEFAULT_GPU_WINDOW,
        );
        assert!(
            (n as u64) <= 40_000 / MIN_LAUNCHES,
            "{n} would claim more than a {}th of the range",
            MIN_LAUNCHES
        );
    }

    /// A range shorter than the floor asks for fewer seeds than the floor allows, and
    /// must not be rounded up into allocating for seeds nobody will walk.
    #[test]
    fn a_range_shorter_than_the_floor_is_not_rounded_up() {
        let l = layout();
        for range in [1u64, 10, 100] {
            let n = auto_batch(
                &l,
                Some(64 << 30),
                false,
                Batch::Auto {
                    filter_bytes: 0,
                    seeds_in_range: range,
                },
            crate::crypto::ec::DEFAULT_GPU_WINDOW,
        );
            assert_eq!(n as u64, range, "range of {range}");
        }
    }

    /// A device that will not say how much memory it has must be assumed small. Guessing
    /// high fails the run at startup, which is worse than running slower than it could.
    #[test]
    fn an_unknown_device_gets_the_floor() {
        assert_eq!(
            auto_batch(
                &layout(),
                None,
                false,
                Batch::Auto {
                    filter_bytes: 0,
                    seeds_in_range: 1 << 32,
                },
            crate::crypto::ec::DEFAULT_GPU_WINDOW,
        ),
            MIN_AUTO_BATCH
        );
    }

    /// Unified memory stops earlier than a discrete card, given identical inputs.
    ///
    /// Not a tuning detail: on a discrete card the VRAM is otherwise idle, while on
    /// unified memory the same gigabytes are holding the resident filter and the CPU
    /// workers' working set. Spending 4.3 GB there to gain nothing measurable is how a
    /// 16 GB machine ends up swapping, which costs far more than the launch saves.
    #[test]
    fn unified_memory_is_given_a_smaller_batch_than_a_discrete_card() {
        let l = layout();
        let ask = Batch::Auto {
            filter_bytes: 0,
            seeds_in_range: 1 << 32,
        };
        let discrete = auto_batch(&l, Some(64 << 30), false, ask, crate::crypto::ec::DEFAULT_GPU_WINDOW);
        let unified = auto_batch(&l, Some(64 << 30), true, ask, crate::crypto::ec::DEFAULT_GPU_WINDOW);
        assert_eq!(discrete, MAX_AUTO_BATCH);
        assert_eq!(unified, MAX_AUTO_BATCH_UNIFIED);
        assert!(unified < discrete);
    }

    /// `--gpu-batch` is an override, so it is honoured whatever the device or range says.
    #[test]
    fn an_explicit_batch_overrides_every_bound() {
        assert_eq!(
            auto_batch(&layout(), None, false, Batch::Fixed(100_000), crate::crypto::ec::DEFAULT_GPU_WINDOW),
            100_000
        );
        assert_eq!(
            auto_batch(&layout(), Some(1 << 30), false, Batch::Fixed(1), crate::crypto::ec::DEFAULT_GPU_WINDOW),
            1
        );
    }

    /// A wider scope costs more per seed, so the same device has to take fewer of them.
    #[test]
    fn a_wider_scope_gets_a_smaller_batch() {
        let narrow = Layout::new(
            &Scope {
                material_sizes: vec![32],
                routes: vec![crate::scan::derive::Route::Bip39],
                paths: vec![crate::wallet::path::PathSpec::parse("m/0").unwrap()],
                forms: vec![crate::wallet::address::HashForm::Compressed],
            },
            1,
        );
        let device = Some(4u64 << 30);
        let ask = Batch::Auto {
            filter_bytes: 0,
            seeds_in_range: 1 << 32,
        };
        assert!(
            auto_batch(&layout(), device, false, ask, crate::crypto::ec::DEFAULT_GPU_WINDOW) < auto_batch(&narrow, device, false, ask, crate::crypto::ec::DEFAULT_GPU_WINDOW),
            "the default scope should not get as many seeds as a one-address scope"
        );
    }
}
