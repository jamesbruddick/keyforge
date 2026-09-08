//! A fixed-base secp256k1 multiplier built for throughput rather than secrecy.
//!
//! The scanner derives hundreds of public keys per seed and does nothing else with
//! the curve, so `k*G` is the entire elliptic-curve workload. libsecp256k1 spends most of
//! its time there defending a secret that does not exist here: every scalar descends
//! from a published 32-bit timestamp by a published algorithm. Its generator multiply
//! sweeps the whole precomputed table with conditional moves so the access pattern
//! reveals nothing, and it normalises each result with its own modular inverse.
//!
//! Dropping both assumptions is worth several times the throughput on this workload:
//!
//!   * The comb table is indexed directly. Signed `WINDOW`-bit digits over a table of
//!     `2^(WINDOW-1)` points per row turn a 256-bit multiply into one point addition
//!     per digit and no doublings, reading one 64-byte point per row instead of
//!     sweeping the row under conditional moves.
//!   * Scalars are multiplied a chunk at a time rather than one by one, so the
//!     additions at a given digit position are independent of each other and can share
//!     a single field inversion. That is what makes plain affine addition affordable:
//!     2 multiplies and a squaring, against the eleven of the mixed Jacobian addition
//!     a lone scalar is stuck with. See `Batch::chunk`.
//!
//! **This is not a general-purpose curve implementation.** It is variable-time
//! throughout -- the digit loop branches on the scalar, and `add` branches on point
//! equality -- so using it on key material anyone is meant not to learn would leak
//! that material through timing. It is correct, not safe.
//!
//! Correctness is held against libsecp256k1 itself: the tests compare hundreds of
//! thousands of public keys, the batch path against the single path, and the table
//! against the curve equation.

use crate::crypto::field::{self, Fe};
use std::sync::LazyLock;

/// Bits per comb digit, and the one knob that trades table size against additions.
///
/// A scalar costs one point addition per nonzero digit, so widening the window cuts
/// the curve work directly: 13 bits is 20 digits where 8 is 32. It pays in table size,
/// which doubles with every bit. The table is read at an unpredictable offset once per
/// digit, so what matters is whether it stays in cache -- and every worker thread
/// shares one read-only copy rather than holding its own.
///
/// Measured on an M1 Pro (12 MB L2 across the performance cores, 24 MB system cache),
/// default scope, median of five runs at ten threads:
///
/// ```text
///   W      table    seeds/s
///   12    2.6 MB      2,207
///   13    4.8 MB      2,234
///   14    9.4 MB      2,286
///   15   18.0 MB      2,305
/// ```
///
/// It keeps improving and keeps flattening: 12 to 14 is 3.6%, 14 to 15 is 0.8% for
/// twice the memory. 13 is the balance -- within a couple of percent of the peak at
/// half the footprint of 14, which leaves the cache to the rest of the working set and
/// stays reasonable on a machine with a smaller last-level cache. Building the table
/// costs well under a second at any of these sizes, so startup does not enter into it.
///
/// Worth retuning per machine, and worth measuring rather than reasoning about: an
/// earlier version of this table was taken from single runs and reported a collapse at
/// 14 that turned out to be system noise. Take medians.
///
/// **The default is now 16, and 13 was too cautious.** The curve above stopped at 15
/// because that was as far as it had been measured, not because it had turned over.
/// Re-measured on the same M1 Pro at eight threads, medians of three:
///
/// ```text
///   W      table    seeds/s
///   13    4.8 MB      2,711
///   14    9.4 MB      2,745
///   15   18.0 MB      2,744
///   16   35.7 MB      2,819
/// ```
///
/// 16 is 4% ahead of 13 on a 24 MB last-level cache, and the GPU's own window landed on
/// 16 independently once it was measured apart from this one. Two curves taken on
/// different processors agreeing on the same answer is a better basis for the default
/// than the smaller table being intuitively safer.
///
/// 17 would be free of charge in rows -- `ceil(256/W) + 1` is 17 at both 16 and 17 -- so
/// it doubles the table for nothing, and 18 buys one row for four times the memory. 16 is
/// where this stops.
///
/// The cost is startup: 17 x 32,768 entries is ~557,000 point additions and ~90 MB of
/// transient allocation, which takes a scan's process from 0.06s to 0.45s before the
/// first seed. Irrelevant against a sweep measured in days, and it is the one thing that
/// makes `verify` -- a one-shot triage command -- noticeably slower than it was.
///
/// **A machine with a small last-level cache may still want less.** The table is read at
/// an unpredictable offset once per row, so at 35.7 MB against an 8 MB L3 most of those
/// reads reach memory; whether that costs more than the four rows it saves depends on how
/// much memory-level parallelism the batch keeps in flight, and that has not been measured
/// on such a machine. Sweep it there rather than assuming, with `KEYFORGE_EC_WINDOW`:
///
/// ```sh
/// for w in 13 14 15 16; do
///     KEYFORGE_EC_WINDOW=$w cargo build --release -q
///     ./target/release/keyforge scan --vuln milksad -f addresses.bf --end 4000
/// done
/// ```
///
/// Table size is `rows * 2^(W-1) * 64` bytes and doubles with every bit: 4.8 MB at 13,
/// 35.7 MB at 16. It is read-only and shared by every worker thread, so what matters is
/// whether it stays in the last-level cache against everything else the scan is touching
/// -- which on a scan means a bloom filter far larger than any cache. Take medians, and
/// prefer the smaller window when two are within a percent of each other.
const WINDOW: usize = match option_env!("KEYFORGE_EC_WINDOW") {
    Some(text) => parse_window(text),
    None => 16,
};

/// `KEYFORGE_EC_WINDOW` as a number, at compile time.
///
/// Hand-rolled because `usize::from_str_radix` is not a `const fn`. The bounds are what
/// the rest of this file assumes rather than taste: below 2 there is no digit to speak of
/// and `1 << (W - 1)` underflows, and above 20 the table passes a gigabyte.
const fn parse_window(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut value = 0usize;
    let mut i = 0;
    while i < bytes.len() {
        assert!(
            bytes[i] >= b'0' && bytes[i] <= b'9',
            "KEYFORGE_EC_WINDOW must be a number"
        );
        value = value * 10 + (bytes[i] - b'0') as usize;
        i += 1;
    }
    assert!(
        value >= 2 && value <= 20,
        "KEYFORGE_EC_WINDOW must be between 2 and 20"
    );
    value
}

/// The same knob for the GPU, which wants a **much** wider window than the CPU does.
///
/// The two were one constant until it was measured. They should not have been: the trade
/// is table size against additions, and the two processors do not price either the same
/// way. A GPU has thousands of independent lanes issuing gathers, so a table that misses
/// cache costs latency it already has plenty of other work to hide -- while an addition is
/// ~16 field multiplies it cannot hide at all.
///
/// Measured on an M1 Pro, `--gpu only`, default scope, 20,000 seeds at a 2,048 batch,
/// medians of three:
///
/// ```text
///   W    rows    table    k_kmul    seeds/s
///   11     25    1.6 MB    3.97s      2,496
///   12     23    3.0 MB    3.62s      2,608
///   13     21    5.5 MB    3.26s      2,646
///   14     20   10.5 MB    2.98s      2,809
///   15     19   19.9 MB    2.85s      2,849
///   16     17   35.7 MB    2.75s      2,945
/// ```
///
/// Monotonic the whole way, which is the finding: **the table size never bites.** 13 to 16
/// is 11% on the whole sweep for a constant. The CPU's own table stays at 13 -- at 16 it
/// would be 35.7 MB against a 24 MB system cache, and the CPU measurements above `WINDOW`
/// were already flattening at 15.
///
/// 16 rather than more because it is where the row count stops falling *cheaply*:
/// `ceil(256/W) + 1` is 17 rows at both 16 and 17, so 17 doubles the table for nothing.
/// 18 does buy a row, for four times the memory, and on the machine this was measured on
/// that was the end of the argument -- an M1 Pro's 134 MB comes out of the same 16 GB the
/// resident filter and every worker thread are living in.
///
/// **That argument is about unified memory, and it does not transfer to a discrete card.**
/// An RTX 5070 Ti has gigabytes of VRAM doing nothing once the filter and the launch
/// buffers are placed, and the rows keep falling for a while yet:
///
/// **Measured on an RTX 5070 Ti: W=16 gives 114,000 seeds/s and W=20 gives 123,000, +7.9%.**
/// Rows fell 17 -> 14, or 17.6%, and the wall clock fell 7.3% -- so `k_kmul` is about 42%
/// of the launch and, within it, cost tracks the row count almost exactly. That is the
/// useful finding: a row is a gather *and* ~11 field multiplies, and reducing rows cuts both
/// where reducing instructions alone cut one. Two rounds of PTX work took 46% off the
/// instruction count of a point addition and 15% off `k_kmul`; three rows took 18% off it.
///
/// Only some windows buy a row, and the ones that do not are pure cost:
///
/// ```text
///   W    rows    table       vs W=16
///   16     17      35.7 MB      -
///   17     17      71.3 MB     nothing -- twice the table for no row
///   18     16     134.2 MB    -5.9%
///   19     15     251.7 MB   -11.8%
///   20     14     469.8 MB   -17.6%   measured, +7.9% on the sweep
///   21     14     939.5 MB    nothing
///   22     13    1745.0 MB   -23.5%
///   23     13    3490.0 MB    nothing
///   24     12    6442.5 MB   -29.4%
///   25     12   12885.0 MB    nothing
/// ```
///
/// So the ladder is 18, 19, 20, 22, 24, and each rung roughly doubles the table. **What
/// stops it is device memory, not the gathers.** On a 16 GB card holding the 7.6 GB filter
/// and a 32,768-seed launch, W=22 fits with room to spare and W=24 does not -- 6.4 GB of
/// comb beside 7.6 GB of filter leaves nothing for the buffers. `auto_batch` charges the
/// table as a fixed cost, so an over-large window does not fail, it silently shrinks the
/// launch; watch the seeds-per-launch figure in the banner if a wide window measures badly.
///
/// The cap is 24 because 25 buys no row and 26 wants 23.6 GB, which is past any consumer
/// card holding a filter as well.
///
/// **This is chosen for you, per device and per filter.** See `gpu::auto_window`: the
/// window is a memory-for-work trade and the memory is whatever the filter left, so the
/// same card with a 7.6 GB filter and with a 300 MB one wants windows two rungs apart. A
/// 16 GB card holding the 7.6 GB filter lands on 22; a smaller filter, or a 24 GB card,
/// reaches 24. Unified memory stays at the default, where the table would come out of the
/// pool the filter is living in.
///
/// `KEYFORGE_GPU_WINDOW` overrides it, and is read at run time rather than compile time --
/// the kernel is built by NVRTC at startup and the table is built here, so nothing needs
/// rebuilding to sweep one:
///
/// ```sh
/// for w in 18 19 20 22 24; do
///     KEYFORGE_GPU_WINDOW=$w ./target/release/keyforge scan --vuln milksad \
///         -f addresses.bf --gpu only --end 3000000
/// done
/// ```
///
/// 16 remains the *fallback* -- what a unified device, a short run, or a card that will not
/// report its memory gets -- because it is the only window with measurements on such a
/// machine behind it.
///
/// The cost is startup, and it doubles with the table: 17 x 32,768 entries is ~557,000
/// point additions against the CPU table's 86,000, W=20 is 7.3 million (~6s), W=22 is 27.3
/// million (~22s) and W=24 is 100 million (~80s). Paid once -- the table is built lazily and
/// cached -- against a sweep measured in hours, and only when a GPU is actually in use. It
/// is single-threaded, which is what makes the wider windows feel slow to start; that is
/// worth fixing before the startup cost is worth caring about, not after.
/// Windows that actually buy a row, widest first.
///
/// Everything between them is pure cost: `ceil(256/W) + 1` is 14 rows at both 20 and 21,
/// and 13 at both 22 and 23, so those rungs double the table and change nothing. This is
/// the ladder `gpu::auto_window` walks.
pub const WINDOW_LADDER: [usize; 6] = [24, 22, 20, 19, 18, 16];

/// The window when nothing has chosen one: what a small device, a short run, or a caller
/// that never asked gets.
pub const DEFAULT_GPU_WINDOW: usize = 16;

/// Bytes the comb table occupies at a given window. Pure arithmetic, so the sizing that
/// picks a window can ask about one it has not built.
pub const fn comb_bytes_at(window: usize) -> usize {
    let (rows, row_len) = shape_of(window);
    rows * row_len * 64
}

/// The window this process will build its GPU comb at, settled exactly once.
///
/// A `OnceLock` and not a `LazyLock` because the value depends on the device: it cannot be
/// computed from the environment alone, and it must be fixed before anything reads it --
/// `gpu::source` bakes `EC_WINDOW` into the kernel and `buffer_bytes` charges the table
/// against the launch budget, so a window that changed between those two would put a table
/// on the device that the kernel does not index and the batch was not sized for.
///
/// `Gpu::open` calls `set_gpu_window` before it sizes anything. Everything else reads
/// through `gpu_window`, which falls back to the default rather than panicking: the parity
/// harness and the table tests legitimately run without a device having chosen for them.
static GPU_WINDOW: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// `KEYFORGE_GPU_WINDOW`, if it is set. Overrides whatever the device would have chosen.
pub fn gpu_window_override() -> Option<usize> {
    let text = std::env::var("KEYFORGE_GPU_WINDOW").ok()?;
    let w: usize = text
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("KEYFORGE_GPU_WINDOW must be a number, got `{text}`"));
    // Below 2 there is no digit and `1 << (W - 1)` underflows. The upper bound is 24 rather
    // than the CPU's 20: 25 buys no row over 24, and 26 wants 23.6 GB. The CPU keeps its own
    // limit because its table shares a last-level cache measured in tens of megabytes, where
    // a discrete card has gigabytes of VRAM doing nothing.
    assert!(
        (2..=24).contains(&w),
        "KEYFORGE_GPU_WINDOW must be between 2 and 24, got {w}"
    );
    Some(w)
}

/// Fix the window for this process. The first call wins; later ones are ignored.
///
/// Ignored rather than rejected because a process may open several devices -- the test
/// suite opens one per test -- and the table is built once and shared. A second opinion
/// arriving after the table exists cannot be honoured, and pretending otherwise would be
/// worse than keeping the first.
pub fn set_gpu_window(window: usize) -> usize {
    *GPU_WINDOW.get_or_init(|| window)
}

/// The window in force, or the default if nothing has chosen one.
pub fn gpu_window() -> usize {
    *GPU_WINDOW.get_or_init(|| gpu_window_override().unwrap_or(DEFAULT_GPU_WINDOW))
}
/// Digits are base-`2^WINDOW` and signed, so a row covers `1..=BASE/2` and negation
/// covers the rest -- halving the table for the cost of a borrow between digits.
const BASE: u64 = 1 << WINDOW;
const ROW_LEN: usize = (BASE / 2) as usize;

/// One row per digit of a 256-bit scalar, plus one for the carry the signed recoding
/// can push off the top.
const ROWS: usize = 256usize.div_ceil(WINDOW) + 1;

/// `start` marks a row index and `ABANDONED` takes `u8::MAX` out of that range, so the
/// row count has to stay clear of it. True for every window this file accepts -- the
/// narrowest, 2, gives 129 rows -- but it is an invariant of two constants that are set
/// in different places, which is the kind that stops being true quietly.
const _: () = assert!(ROWS < u8::MAX as usize, "ROWS must not collide with ABANDONED");

/// `(rows, row_len)` for a window, which is the whole of a comb's shape.
const fn shape_of(window: usize) -> (usize, usize) {
    (256usize.div_ceil(window) + 1, 1 << (window - 1))
}

/// The generator, from SEC 2.
const G: Ge = Ge {
    x: Fe([
        0x59f2_815b_16f8_1798,
        0x029b_fcdb_2dce_28d9,
        0x55a0_6295_ce87_0b07,
        0x79be_667e_f9dc_bbac,
    ]),
    y: Fe([
        0x9c47_d08f_fb10_d4b8,
        0xfd17_b448_a685_5419,
        0x5da4_fbfc_0e11_08a8,
        0x483a_da77_26a3_c465,
    ]),
};

/// An affine curve point.
///
/// Exactly 64 bytes, which is not an accident: the comb is tens of thousands of these
/// read at an unpredictable offset once per digit, and a lane's accumulator is
/// rewritten just as often. At any larger size an entry straddles two cache lines
/// about as often as not, and the table grows by the same fraction.
///
/// That is why infinity is `(0, 0)` rather than a flag. `x = 0` gives `y^2 = 7`, whose
/// roots are not zero, so the point is not on the curve and cannot arise as a real
/// value -- it is free to mean something else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Ge {
    pub x: Fe,
    pub y: Fe,
}

const GE_INFINITY: Ge = Ge {
    x: field::ZERO,
    y: field::ZERO,
};

/// A Jacobian curve point: (x, y) = (X/Z^2, Y/Z^3).
#[derive(Clone, Copy, Debug)]
pub struct Gej {
    x: Fe,
    y: Fe,
    z: Fe,
    infinity: bool,
}

const INFINITY: Gej = Gej {
    x: field::ZERO,
    y: field::ZERO,
    z: field::ZERO,
    infinity: true,
};

impl Ge {
    #[inline]
    fn negate(&self) -> Ge {
        Ge {
            x: self.x,
            y: self.y.negate(),
        }
    }

    #[inline]
    fn is_infinity(&self) -> bool {
        self.x.is_zero() && self.y.is_zero()
    }

    /// SEC1 compressed encoding: a parity byte and x.
    #[inline]
    pub fn serialize(&self) -> [u8; 33] {
        debug_assert!(!self.is_infinity());
        let mut out = [0u8; 33];
        out[0] = if self.y.is_odd() { 0x03 } else { 0x02 };
        out[1..].copy_from_slice(&self.x.to_bytes());
        out
    }

    /// SEC1 uncompressed encoding: a 0x04 tag, then x and y.
    #[inline]
    pub fn serialize_uncompressed(&self) -> [u8; 65] {
        debug_assert!(!self.is_infinity());
        let mut out = [0u8; 65];
        out[0] = 0x04;
        out[1..33].copy_from_slice(&self.x.to_bytes());
        out[33..].copy_from_slice(&self.y.to_bytes());
        out
    }

    /// Does this point satisfy y^2 = x^3 + 7? Only the tests ask.
    #[cfg(test)]
    fn is_on_curve(&self) -> bool {
        !self.is_infinity() && self.y.sqr() == self.x.sqr().mul(&self.x).add(&field::B)
    }
}

impl Gej {
    #[inline]
    fn from_ge(p: &Ge) -> Gej {
        Gej {
            x: p.x,
            y: p.y,
            z: field::ONE,
            infinity: p.is_infinity(),
        }
    }

    /// Point doubling, `dbl-2009-l` from the EFD, valid because the curve has a = 0.
    fn double(&self) -> Gej {
        if self.infinity || self.y.is_zero() {
            return INFINITY;
        }
        let a = self.x.sqr();
        let b = self.y.sqr();
        let c = b.sqr();
        let d = self.x.add(&b).sqr().sub(&a).sub(&c).mul2();
        let e = a.mul3();
        let f = e.sqr();

        let x = f.sub(&d.mul2());
        let y = e.mul(&d.sub(&x)).sub(&c.mul8());
        let z = self.y.mul(&self.z).mul2();
        Gej {
            x,
            y,
            z,
            infinity: false,
        }
    }

    /// Mixed addition, `madd-2007-bl` from the EFD: Jacobian plus affine.
    ///
    /// The degenerate cases are handled by branching, which is exactly what a
    /// constant-time implementation may not do. See the module note.
    fn add_ge(&self, other: &Ge) -> Gej {
        if other.is_infinity() {
            return *self;
        }
        if self.infinity {
            return Gej::from_ge(other);
        }

        let z1z1 = self.z.sqr();
        let u2 = other.x.mul(&z1z1);
        let s2 = other.y.mul(&self.z).mul(&z1z1);
        let h = u2.sub(&self.x);
        let r = s2.sub(&self.y).mul2();

        if h.is_zero() {
            // Same x: either the same point, which needs the doubling formula, or a
            // point and its negation, which sum to infinity.
            return if r.is_zero() { self.double() } else { INFINITY };
        }

        let hh = h.sqr();
        let i = hh.mul4();
        let j = h.mul(&i);
        let v = self.x.mul(&i);

        let x = r.sqr().sub(&j).sub(&v.mul2());
        let y = r.mul(&v.sub(&x)).sub(&self.y.mul(&j).mul2());
        let z = self.z.add(&h).sqr().sub(&z1z1).sub(&hh);
        Gej {
            x,
            y,
            z,
            infinity: false,
        }
    }
}

/// The comb: row `j` holds `d * BASE^j * G` for `d` in `1..=ROW_LEN`, laid out
/// row-major.
struct Comb(Vec<Ge>);

impl Comb {
    #[inline]
    fn get(&self, row: usize, digit: usize) -> Ge {
        self.0[row * ROW_LEN + digit - 1]
    }
}

/// Built once per process and shared by every worker thread, so its size is paid once
/// rather than per thread. See `WINDOW` for the size it comes to.
static COMB: LazyLock<Comb> = LazyLock::new(|| Comb(build_comb(WINDOW)));

/// The comb for an arbitrary window: row `j` holds `d * 2^(window*j) * G` for `d` in
/// `1..=row_len`, laid out row-major.
///
/// Parameterised because the CPU and the GPU want different windows -- see `GPU_WINDOW`.
/// One builder rather than two so `the_comb_table_is_exhaustively_correct` can check both
/// tables with the same code, which is what stops the second window from being a second
/// construction that could quietly disagree.
fn build_comb(window: usize) -> Vec<Ge> {
    let (rows, row_len) = shape_of(window);
    // Fill in Jacobian coordinates first so the whole table costs one inversion.
    let mut jacobian = vec![INFINITY; rows * row_len];
    let mut base = G;
    for row in 0..rows {
        let mut acc = Gej::from_ge(&base);
        for d in 0..row_len {
            jacobian[row * row_len + d] = acc;
            acc = acc.add_ge(&base);
        }
        // The next row's base is this one times 2^window.
        if row + 1 < rows {
            let mut next = Gej::from_ge(&base);
            for _ in 0..window {
                next = next.double();
            }
            base = to_affine(std::slice::from_mut(&mut next))[0];
        }
    }

    to_affine(&jacobian)
}

/// Invert every element of `values` behind a single field inversion.
///
/// Montgomery's trick: accumulate the running product, invert that once, then walk
/// back peeling off one factor at a time. An inversion is ~255 squarings and dwarfs
/// everything around it, so sharing one across `n` values turns the per-value cost
/// into about three multiplications. No element may be zero.
fn batch_invert(values: &[Fe], prefix: &mut Vec<Fe>, out: &mut Vec<Fe>) {
    debug_assert!(values.iter().all(|v| !v.is_zero()));
    prefix.clear();
    let mut running = field::ONE;
    for v in values {
        prefix.push(running);
        running = running.mul(v);
    }

    let mut inv = running.inv();
    out.clear();
    out.resize(values.len(), field::ZERO);
    for i in (0..values.len()).rev() {
        out[i] = inv.mul(&prefix[i]);
        inv = inv.mul(&values[i]);
    }
}

/// Convert Jacobian points to affine behind a single inversion. Cold: table
/// construction, the degenerate-case fallback, and the tests.
fn to_affine(points: &[Gej]) -> Vec<Ge> {
    // Infinities get a placeholder of one rather than their zero z, which would
    // otherwise destroy every other point's inverse. Their output is discarded.
    let zs: Vec<Fe> = points
        .iter()
        .map(|p| if p.infinity { field::ONE } else { p.z })
        .collect();
    let (mut prefix, mut inv) = (Vec::new(), Vec::new());
    batch_invert(&zs, &mut prefix, &mut inv);

    points
        .iter()
        .zip(&inv)
        .map(|(p, z_inv)| {
            if p.infinity {
                return GE_INFINITY;
            }
            let z_inv2 = z_inv.sqr();
            Ge {
                x: p.x.mul(&z_inv2),
                y: p.y.mul(&z_inv2).mul(z_inv),
            }
        })
        .collect()
}

/// How many scalars share one field inversion per digit position.
///
/// The whole chunk is live at once -- an accumulator, a digit and an inverse per lane
/// -- so this trades L1 residency against how thinly the inversion is spread. At 256
/// the scratch is around 50 KB, comfortably inside a 128 KB L1, and the inversion is
/// already down to well under one multiplication per lane; going wider buys almost
/// nothing and starts to spill.
const LANES: usize = 256;

/// Scratch space for turning a batch of scalars into public keys.
///
/// Held across calls by each worker thread so a batch costs no allocation. Every
/// buffer is indexed by lane within the current chunk.
#[derive(Default)]
pub struct Batch {
    /// Digits transposed: `digits[row * lanes + lane]`, because the loop walks one
    /// digit position across every lane at a time.
    ///
    /// **`i32`, and the width is load-bearing.** This was `i16`, which holds a signed
    /// digit only up to `WINDOW = 15`: the range is `-2^(W-1) ..= 2^(W-1)`, so at 16 the
    /// digit `+32768` does not fit and `digit as i16` -- a wrapping cast, silent in
    /// release -- turns it into `-32768`. The result is a wrong public key with no
    /// diagnostic anywhere, and it is reachable by doing exactly what the note on
    /// `WINDOW` invites: retuning the window upward on a machine with more cache.
    /// `batched_keys_match_single_keys` catches it and nothing else does, because the
    /// single-scalar path and the comb table itself are both fine.
    digits: Vec<i32>,
    /// The affine accumulator per lane -- the running `k*G` as far as it has got.
    acc: Vec<Ge>,
    /// The first digit position at which a lane has anything to add. `ABANDONED`
    /// marks a lane that has dropped out to the fallback below.
    start: Vec<u8>,
    /// The lanes adding at the digit position being processed, and what each adds.
    ///
    /// Carrying the addend over from the gather looks wasteful -- 64 bytes written and
    /// read back per lane, when the second pass could re-read the table from a digit it
    /// has to load anyway. It is not: a row is `ROW_LEN` entries, a quarter of a
    /// megabyte at the current window, so the entries a chunk touched are long out of
    /// L1 by the time the second pass wants them. Measured, re-reading costs 7%.
    active: Vec<u32>,
    addends: Vec<Ge>,
    /// `x_addend - x_acc` per active lane, and its inverse; the two together are the
    /// slope of every one of this position's additions.
    dx: Vec<Fe>,
    inv: Vec<Fe>,
    prefix: Vec<Fe>,
    /// Lanes that met a degenerate addition and are being recomputed the slow way.
    abandoned: Vec<u32>,
    jacobian: Vec<Gej>,
}

/// `start` value marking a lane the digit loop must not touch again.
const ABANDONED: u8 = u8::MAX;

impl Batch {
    /// Public keys for `scalars`, appended to `out` after clearing it.
    ///
    /// Scalars must be valid secret keys -- in `1..n` -- which every caller here
    /// guarantees by construction, so no result is the point at infinity.
    pub fn public_keys(&mut self, scalars: &[[u8; 32]], out: &mut Vec<Ge>) {
        out.clear();
        out.reserve(scalars.len());
        for chunk in scalars.chunks(LANES) {
            self.chunk(chunk, out);
        }
    }

    /// One chunk of at most `LANES` scalars, stepped through the comb together.
    ///
    /// This is the shape the whole module exists for. A single `k*G` is a chain of
    /// point additions where each waits on the one before it, and each of those is
    /// itself a chain of long-latency field multiplications -- a core running one
    /// scalar spends most of its issue slots idle. Running a chunk of scalars through
    /// the same digit position at once makes every lane independent of every other,
    /// and it makes affine addition affordable: the inversion each one would need
    /// alone becomes a single inversion shared by the position.
    ///
    /// Affine addition is 2 multiplications and a squaring, plus about three more for
    /// its share of the inversion. Mixed Jacobian addition, which is what a lone
    /// scalar has to use, is eleven.
    fn chunk(&mut self, scalars: &[[u8; 32]], out: &mut Vec<Ge>) {
        let comb = &*COMB;
        let lanes = scalars.len();

        self.digits.clear();
        self.digits.resize(ROWS * lanes, 0);
        self.start.clear();
        self.acc.clear();
        self.abandoned.clear();

        for (lane, scalar) in scalars.iter().enumerate() {
            let digits = signed_digits(scalar);
            for (row, &digit) in digits.iter().enumerate() {
                self.digits[row * lanes + lane] = digit;
            }
            // A scalar in 1..n has at least one nonzero digit, and the point it
            // selects seeds the accumulator -- affine coordinates cannot represent
            // the infinity an empty sum would start from.
            let first = digits.iter().position(|&d| d != 0).expect("scalar is zero");
            self.start.push(first as u8);
            self.acc.push(entry(comb, first, digits[first]));
        }

        for row in 0..ROWS {
            self.active.clear();
            self.addends.clear();
            self.dx.clear();

            for lane in 0..lanes {
                if row <= self.start[lane] as usize {
                    continue;
                }
                let digit = self.digits[row * lanes + lane];
                if digit == 0 {
                    continue;
                }
                let point = entry(comb, row, digit);
                let dx = point.x.sub(&self.acc[lane].x);
                if dx.is_zero() {
                    // The accumulator has landed on this row's entry or its negation,
                    // so the slope is 0/0 and the affine formula says nothing. Both
                    // cases need the group law's special forms, and a zero here would
                    // also destroy every other lane's inverse. It is a ~2^-128 event
                    // over scalars nobody chose adversarially, so rather than carry a
                    // second formula through the hot loop the lane leaves it and is
                    // recomputed by the Jacobian path below, which handles both.
                    self.abandoned.push(lane as u32);
                    self.start[lane] = ABANDONED;
                    continue;
                }
                self.active.push(lane as u32);
                self.addends.push(point);
                self.dx.push(dx);
            }

            if self.active.is_empty() {
                continue;
            }
            // Montgomery's running products are built here, in a pass of their own,
            // rather than accumulated during the gather above. Folding them in looks
            // free -- the gather has each difference in a register already -- but the
            // product is a serial dependency, and threading it through a loop whose
            // iterations are otherwise independent costs 7%.
            batch_invert(&self.dx, &mut self.prefix, &mut self.inv);

            for (slot, &lane) in self.active.iter().enumerate() {
                let lane = lane as usize;
                let (p, q) = (self.acc[lane], self.addends[slot]);
                // lambda = (y2 - y1) / (x2 - x1); x3 = lambda^2 - x1 - x2;
                // y3 = lambda * (x1 - x3) - y1.
                let lambda = q.y.sub(&p.y).mul(&self.inv[slot]);
                let x = lambda.sqr().sub(&p.x).sub(&q.x);
                let y = lambda.mul(&p.x.sub(&x)).sub(&p.y);
                self.acc[lane] = Ge { x, y };
            }
        }

        // The lanes that met a degenerate addition, done again from scratch. Never
        // taken in practice, so it allocates rather than complicating the loop above.
        for &lane in &self.abandoned {
            self.jacobian.clear();
            self.jacobian.push(mul_gen(&scalars[lane as usize]));
            self.acc[lane as usize] = to_affine(&self.jacobian)[0];
        }

        out.extend_from_slice(&self.acc);
    }
}

/// The comb entry a signed digit selects, negated when the digit is.
#[inline]
fn entry(comb: &Comb, row: usize, digit: i32) -> Ge {
    let point = comb.get(row, digit.unsigned_abs() as usize);
    if digit < 0 { point.negate() } else { point }
}

/// Recode a big-endian scalar into signed base-`2^WINDOW` digits.
///
/// A plain split would need `BASE` table entries per row; borrowing from the next
/// digit whenever one exceeds `BASE/2` halves that, because `-d` is a free negation of
/// `d`'s point. The borrow can run off the top, which is the extra row in `ROWS`.
#[inline]
fn signed_digits(scalar: &[u8; 32]) -> [i32; ROWS] {
    // Little-endian limbs, so digit j is the WINDOW-bit field at bit j*WINDOW.
    let mut limbs = [0u64; 4];
    for (i, limb) in limbs.iter_mut().enumerate() {
        *limb = u64::from_be_bytes(scalar[24 - i * 8..32 - i * 8].try_into().unwrap());
    }

    let mut digits = [0i32; ROWS];
    let mut carry = 0i32;
    for (j, digit) in digits.iter_mut().take(ROWS - 1).enumerate() {
        // Read the field out of a 128-bit window so one straddling a limb boundary
        // needs no special case. The digits past bit 256 read as zero.
        let (limb, offset) = (j * WINDOW / 64, j * WINDOW % 64);
        let wide =
            (limbs[limb] as u128) | ((limbs.get(limb + 1).copied().unwrap_or(0) as u128) << 64);
        let d = ((wide >> offset) as u64 & (BASE - 1)) as i32 + carry;

        if d > ROW_LEN as i32 {
            *digit = d - BASE as i32;
            carry = 1;
        } else {
            *digit = d;
            carry = 0;
        }
    }
    digits[ROWS - 1] = carry;
    digits
}

/// `scalar * G` on its own, in Jacobian coordinates.
///
/// The scan does not come this way -- `Batch` is several times faster per key and is
/// what every caller in the hot path uses. This is here for the two places a single
/// scalar is genuinely all there is: building the comb table, and recovering the rare
/// lane that meets a degenerate addition.
fn mul_gen(scalar: &[u8; 32]) -> Gej {
    let comb = &*COMB;
    let mut acc = INFINITY;
    for (row, &digit) in signed_digits(scalar).iter().enumerate() {
        if digit == 0 {
            continue;
        }
        let point = comb.get(row, digit.unsigned_abs() as usize);
        acc = acc.add_ge(&if digit < 0 { point.negate() } else { point });
    }
    acc
}

/// The GPU comb's shape: `(GPU_WINDOW, rows, row_len)`.
///
/// The GPU compiles its own copy of the digit recoding, so it needs these as constants
/// rather than inferring them. Exported from here rather than duplicated so the table and
/// the recoding that indexes it can only ever be retuned together.
pub fn comb_shape() -> (usize, usize, usize) {
    let window = gpu_window();
    let (rows, row_len) = shape_of(window);
    (window, rows, row_len)
}

/// How many bytes `comb_for_gpu` will return, without building it.
///
/// `buffer_bytes` needs the size to choose a launch, and called `comb_for_gpu().len()` to
/// get it -- rebuilding the whole table, three times per `auto_batch`, to read a length.
/// That was half a second of startup at W=16 and would be twenty at W=20.
pub fn comb_gpu_bytes() -> usize {
    comb_bytes_at(gpu_window())
}

/// The GPU's comb table in the GPU's limb order, ready to upload.
///
/// Same construction as the CPU's and the same layout, at a different window and with a
/// different splitting: this file keeps a field element as four 64-bit limbs and
/// `kernels/field.h` keeps it as eight 32-bit ones. Going through `Fe::to_bytes` rather
/// than reinterpreting the memory makes that conversion explicit and endian-independent.
///
/// It goes through the same `build_comb` the CPU's table does, so
/// `the_comb_table_is_exhaustively_correct` covers both windows -- one construction,
/// checked twice, rather than a second one that could quietly disagree.
pub fn comb_for_gpu() -> &'static [u8] {
    static TABLE: LazyLock<Vec<u8>> = LazyLock::new(build_comb_for_gpu);
    &TABLE
}

fn build_comb_for_gpu() -> Vec<u8> {
    let (window, rows, row_len) = comb_shape();
    let comb = build_comb(window);
    let mut out = Vec::with_capacity(rows * row_len * 64);
    for point in &comb {
        for fe in [&point.x, &point.y] {
            // `to_bytes` is big-endian; the kernel's limb 0 is the least significant.
            let be = fe.to_bytes();
            for i in 0..8 {
                let limb = u32::from_be_bytes(be[28 - i * 4..32 - i * 4].try_into().unwrap());
                out.extend_from_slice(&limb.to_le_bytes());
            }
        }
    }
    out
}

/// The public key for a single scalar.
///
/// One public key, with an inversion all to itself.
///
/// The scan proper never comes through here: it always has a level's worth of scalars to
/// share an inversion with, which is the whole reason `Batch` exists. This is for the
/// callers that genuinely have one key and nothing to batch it with -- the tests, which
/// check the tree a node at a time, and `derive::leaf_private_key`, which re-derives a
/// single leaf after a filter match. Both are cold paths where an inversion per key
/// costs nothing worth measuring.
pub fn public_key(scalar: &[u8; 32]) -> Ge {
    to_affine(std::slice::from_ref(&mul_gen(scalar)))[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;
    use secp256k1::{PublicKey, Secp256k1, SecretKey};

    fn reference(scalar: &[u8; 32]) -> [u8; 33] {
        let secp = Secp256k1::signing_only();
        let key = SecretKey::from_byte_array(*scalar).unwrap();
        PublicKey::from_secret_key(&secp, &key).serialize()
    }

    /// The scalar that produces the comb's entry for `d` at `row`.
    ///
    /// The entry is `d * BASE^row * G`, a group element, so the scalar behind it is
    /// `d * BASE^row` reduced modulo the group order -- and that always fits 256 bits
    /// even when the unreduced product does not. Reducing rather than giving up on the
    /// overflowing entries is what keeps the table check exhaustive at any `WINDOW`.
    fn base_pow(d: u64, row: usize) -> [u8; 32] {
        base_pow_at(d, row, WINDOW)
    }

    /// The same for an arbitrary window, so the CPU and GPU tables share one reference.
    fn base_pow_at(d: u64, row: usize, window: usize) -> [u8; 32] {
        // n, the order of the group.
        let n = num_bigint::BigUint::from_bytes_be(&[
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xfe, 0xba, 0xae, 0xdc, 0xe6, 0xaf, 0x48, 0xa0, 0x3b, 0xbf, 0xd2, 0x5e, 0x8c,
            0xd0, 0x36, 0x41, 0x41,
        ]);
        let v = (num_bigint::BigUint::from(d) << (window * row)) % n;

        let bytes = v.to_bytes_be();
        let mut scalar = [0u8; 32];
        scalar[32 - bytes.len()..].copy_from_slice(&bytes);
        scalar
    }

    fn random_scalars(count: usize) -> Vec<[u8; 32]> {
        let mut rng = rand::rng();
        let mut out = Vec::with_capacity(count);
        while out.len() < count {
            let bytes: [u8; 32] = rng.random();
            if SecretKey::from_byte_array(bytes).is_ok() {
                out.push(bytes);
            }
        }
        out
    }

    /// The generator itself, against the constant every implementation publishes.
    #[test]
    fn the_generator_is_the_published_point() {
        let mut one = [0u8; 32];
        one[31] = 1;
        assert_eq!(
            hex::encode(public_key(&one).serialize_uncompressed()),
            concat!(
                "04",
                "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
                "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8",
            )
        );
    }

    /// Every table entry must be the point it claims to be, and must be on the curve.
    ///
    /// A single wrong entry would corrupt only the scalars whose digits select it,
    /// which random testing could easily miss, so the table is checked exhaustively:
    /// row j entry d has to equal d * 256^j * G.
    #[test]
    fn the_comb_table_is_exhaustively_correct() {
        let comb = &*COMB;
        for row in 0..ROWS {
            for d in 0..ROW_LEN {
                let point = comb.get(row, d + 1);
                assert!(point.is_on_curve(), "row {row} entry {d} is off the curve");

                let scalar = base_pow(d as u64 + 1, row);
                assert_eq!(
                    point.serialize(),
                    reference(&scalar),
                    "row {row} entry {d} is not {}*{BASE}^{row}*G",
                    d + 1
                );
            }
        }
    }

    /// And so is the GPU's, which is a *different* table since `GPU_WINDOW` split from
    /// `WINDOW`. The builder is shared, so this is checking the shape arithmetic and the
    /// wider row bases rather than the point addition -- but a comb is exactly the kind of
    /// thing where one wrong entry corrupts only the scalars that select it, so it is
    /// checked entry by entry like the other one.
    ///
    /// Ignored by default: 17 x 32,768 entries against a reference implementation is
    /// minutes, not seconds. Run it when `GPU_WINDOW` changes.
    ///
    ///     cargo test --release -- --ignored the_gpu_comb_table
    #[test]
    #[ignore = "exhaustive over 557,000 entries; run when GPU_WINDOW changes"]
    fn the_gpu_comb_table_is_exhaustively_correct() {
        let (rows, row_len) = shape_of(gpu_window());
        let comb = build_comb(gpu_window());
        for row in 0..rows {
            for d in 0..row_len {
                let point = comb[row * row_len + d];
                assert!(point.is_on_curve(), "row {row} entry {d} is off the curve");

                let scalar = base_pow_at(d as u64 + 1, row, gpu_window());
                assert_eq!(
                    point.serialize(),
                    reference(&scalar),
                    "row {row} entry {d} is not {}*2^({}*{row})*G",
                    d + 1,
                    gpu_window()
                );
            }
        }
    }

    /// The digit recoding must be a faithful re-encoding of the scalar: the digits,
    /// weighted by 256^j, have to add back up to the original value.
    #[test]
    fn signed_digits_reconstruct_the_scalar() {
        for scalar in random_scalars(2000) {
            let digits = signed_digits(&scalar);
            let mut acc = num_bigint::BigInt::from(0);
            for (j, &d) in digits.iter().enumerate() {
                acc += num_bigint::BigInt::from(d) << (WINDOW * j);
            }
            assert_eq!(
                acc,
                num_bigint::BigInt::from_bytes_be(num_bigint::Sign::Plus, &scalar)
            );
            let limit = ROW_LEN as i32;
            assert!(digits.iter().all(|d| (-limit..=limit).contains(d)));
            assert!(digits[ROWS - 1] == 0 || digits[ROWS - 1] == 1);
        }
    }

    /// The headline claim: this multiplier agrees with libsecp256k1.
    ///
    /// Run with `--ignored` for a far larger sample; the default count is chosen to
    /// keep `cargo test` quick while still covering every row of the table many times
    /// over.
    #[test]
    fn agrees_with_libsecp256k1() {
        for scalar in random_scalars(20_000) {
            assert_eq!(
                public_key(&scalar).serialize(),
                reference(&scalar),
                "mismatch for scalar {}",
                hex::encode(scalar)
            );
        }
    }

    #[test]
    #[ignore = "slow; run with --ignored for a deeper sweep"]
    fn agrees_with_libsecp256k1_over_a_large_sample() {
        for scalar in random_scalars(500_000) {
            assert_eq!(public_key(&scalar).serialize(), reference(&scalar));
        }
    }

    /// The edge scalars: 1, 2, and the ones that make the recoding carry all the way
    /// up -- 2^k - 1 and n - 1, which is the largest valid scalar there is.
    #[test]
    fn agrees_at_the_scalar_edges() {
        let mut edges: Vec<[u8; 32]> = Vec::new();
        for small in [1u8, 2, 3, 127, 128, 129, 255] {
            let mut s = [0u8; 32];
            s[31] = small;
            edges.push(s);
        }
        edges.push([0xff; 32]);
        // n - 1, the order of the group minus one.
        edges.push([
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xfe, 0xba, 0xae, 0xdc, 0xe6, 0xaf, 0x48, 0xa0, 0x3b, 0xbf, 0xd2, 0x5e, 0x8c,
            0xd0, 0x36, 0x41, 0x40,
        ]);

        for scalar in edges {
            if SecretKey::from_byte_array(scalar).is_err() {
                continue;
            }
            assert_eq!(
                public_key(&scalar).serialize(),
                reference(&scalar),
                "mismatch at edge scalar {}",
                hex::encode(scalar)
            );
        }
    }

    /// Batching must not change any answer. Montgomery's trick is easy to get wrong
    /// in a way that only shows up at particular batch sizes or positions, so several
    /// sizes are checked, including the ones with an odd tail.
    ///
    /// The sizes around `LANES` are the ones that matter most: a batch is cut into
    /// chunks of that many lanes, and an off-by-one at the seam would leave a short
    /// final chunk wrong while every round size passed.
    #[test]
    fn batched_keys_match_single_keys() {
        for size in [1usize, 2, 3, 5, 30, 91, 128, 255, 256, 257, 511, 600] {
            let scalars = random_scalars(size);
            let mut batched = Vec::new();
            Batch::default().public_keys(&scalars, &mut batched);
            assert_eq!(batched.len(), size);
            for (point, scalar) in batched.iter().zip(&scalars) {
                assert_eq!(point.serialize(), reference(scalar));
                assert_eq!(point.serialize(), public_key(scalar).serialize());
            }
        }
    }

    /// The uncompressed encoding, which the scanner probes as its own hash form and
    /// which no other test would exercise.
    #[test]
    fn uncompressed_encoding_matches_the_reference() {
        let secp = Secp256k1::signing_only();
        for scalar in random_scalars(500) {
            let key = SecretKey::from_byte_array(scalar).unwrap();
            assert_eq!(
                public_key(&scalar).serialize_uncompressed(),
                PublicKey::from_secret_key(&secp, &key).serialize_uncompressed()
            );
        }
    }

    /// The group law's degenerate cases, which the digit loop never reaches but which
    /// `build_comb` and any future caller can.
    #[test]
    fn addition_handles_infinity_and_negation() {
        let g = Gej::from_ge(&G);
        assert!(INFINITY.add_ge(&G).z == field::ONE);
        assert!(g.add_ge(&G.negate()).infinity, "P + (-P) must be infinity");

        // P + P has to take the doubling branch and agree with `double`.
        let doubled = to_affine(std::slice::from_ref(&g.double()))[0];
        let added = to_affine(std::slice::from_ref(&g.add_ge(&G)))[0];
        assert_eq!(doubled, added);

        let mut two = [0u8; 32];
        two[31] = 2;
        assert_eq!(doubled.serialize(), reference(&two));
    }
}
