//! Read-only reader for the `.bf` bloom filters `keyscan bf-gen` builds.
//!
//! That is the only producer this scanner accepts, and the extension is what says so
//! -- see [`FILTER_EXT`]. The file format is ecloop's, `keyscan` having inherited it
//! (the magic is still "ECBF"), and so is the scattered probe schedule, which here is a
//! faithful port of `blf_has` in `ecloop/lib/utils.c`. But a file being *readable* is
//! not the same as its being right for this sweep, which is the distinction the
//! extension draws.
//!
//! A filter now says two things about itself that it used to leave to convention, and
//! both are read rather than assumed -- see [`BitLayout`] and [`BloomFilter::kinds`]:
//!
//! * **How its bits are laid out.** The scattered schedule spreads twenty probes across
//!   the whole array; the blocked one confines them to a single 512-bit block, so a
//!   lookup costs one cache line rather than one miss per probe, at about 47% more bits
//!   per entry. `bf-gen` writes the filter a scan probes blocked and the verification
//!   filter beside it scattered, each the way its own access pattern wants, so a sweep
//!   reads one of each and neither schedule is the default.
//! * **What went into it.** Which address forms `bf-gen` was given, as a bitmask. It is
//!   the only thing in the file that answers the question the `.bf` name was standing in
//!   for, and [`crate::target::Target::unmatchable_forms`] is what asks it.
//!
//! A version-1 file records neither and is read as scattered, holding what `bf-gen` took
//! before it recorded anything -- which covers every form this sweep derives, so such a
//! filter sweeps exactly as it always did.
//!
//! The two implementations are held together by a test that shells out to
//! `keyscan bf-check`. Any drift here silently turns every lookup into a miss, so the
//! format is validated on open and both schedules are exercised against real data rather
//! than trusted.
//!
//! Filters are large (the working one is 7.6 GB) and the whole of one is read into
//! an owned buffer on open, shared read-only across every thread. Probes land
//! uniformly across the bit space, so anything short of fully resident is not
//! "mostly cached" in any useful sense -- the hit rate equals the resident fraction,
//! and a miss costs a page fault where a hit costs a few nanoseconds. Owning the
//! bytes is what keeps the whole filter resident; a mapping of the same file is at
//! the page cache's mercy.

use anyhow::{Context, Result, bail};
use std::alloc::Layout;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::ptr::NonNull;

/// The extension a filter has to carry, matching what `keyscan bf-gen` writes.
///
/// The check is not about the bytes: an ecloop `.blf` has the same header and the same
/// scattered probe schedule, and would load and answer lookups perfectly well. It is about
/// what went *into* the filter. This sweep tests three hash forms per key, and one of them
/// -- P2SH-P2WPKH, `hash160(0x0014 || hash160(pubkey))` -- is a script hash, which a filter
/// built for a key scanner has no reason to hold.
///
/// A version-2 header answers that question outright -- see [`BloomFilter::kinds`] -- and
/// where it does, [`crate::target::Target::unmatchable_forms`] is the check that matters.
/// The name is what covers a version-1 file, which records nothing about its contents, and
/// it costs a string comparison before the file is opened. The failure it guards against is
/// silent and expensive either way: every P2SH-P2WPKH probe misses, a third of the sweep's
/// work matches nothing it should, and the run still ends reporting a clean sweep of the
/// keyspace.
pub const FILTER_EXT: &str = "bf";

/// FourCC "ECBF", little-endian. Inherited from ecloop along with the format.
const MAGIC: u32 = 0x4543_4246;

/// The version `keyscan bf-gen` writes now: the version-1 header plus a record of what went
/// into the filter and how its bits are arranged.
const VERSION: u32 = 2;

/// The version before that, which is still read. It is the same bit array behind a shorter
/// header, and a filter takes hours to build; refusing one for the sake of eight bytes of
/// provenance that have a known answer would be throwing that away. See [`LEGACY_KINDS`].
const VERSION_LEGACY: u32 = 1;

/// u32 magic + u32 version + u64 word count.
const HEADER_LEN_V1: usize = 16;

/// The above, plus the kinds mask and the bit-layout code.
const HEADER_LEN: usize = 24;

/// The four shift amounts ecloop folds the hash160 words with. Together with the
/// five rotations below they give k = 20 probes.
const SHIFTS: [u32; 4] = [24, 28, 36, 40];

/// The hash160 is folded into this many overlapping 64-bit words, each of which
/// is probed once per shift.
const FOLDED_WORDS: usize = 5;

/// Probes per lookup, which ecloop fixes rather than deriving from the filter's
/// size. It is the exponent in the false-positive rate below.
pub const PROBES: usize = SHIFTS.len() * FOLDED_WORDS;

/// The largest chunk `contains_batch` accepts, and therefore how many DRAM misses it
/// can have in flight at once.
///
/// It wants to be well past the handful of misses a core can sustain -- the point is
/// to keep the memory system busy -- while staying small enough that the survivor
/// buffer is a few hundred bytes of stack and the derivation's matching chunk of
/// locations stays in L1.
pub const PROBE_CHUNK: usize = 256;

/// Bits in one block of a [`BitLayout::Blocked`] filter: 64 bytes, one cache line.
///
/// The whole point of the layout is that every probe for a hash lands inside this span, so a
/// lookup is one cache line and one TLB entry however many probes it makes. Mirrors
/// `BLOCK_BITS` in `keyscan`'s `src/bloom.rs`, and `BLOOM_BLOCK_BITS` in `kernels/bloom.h`;
/// all three name the same bits of the same file.
pub const BLOCK_BITS: u64 = 512;

/// Nine-bit positions per mixed 64-bit word. Seven fit, with a bit to spare.
const POSITIONS_PER_WORD: usize = 7;

/// Constants that make the three position mixes independent of each other and of the block
/// mix. `keyscan`'s, and arbitrary only in the sense that nothing depends on which they are.
const BLOCK_SALT: [u64; 3] = [
    0xa076_1d64_78bd_642f,
    0xe703_7ed1_a0b4_28db,
    0x8ebc_6af0_9c88_c6e3,
];

/// How a filter's bits are arranged, and so how a hash picks the ones it probes.
///
/// Recorded per file rather than assumed, because the two coexist in a pair `keyscan bf-gen`
/// writes: the primary is blocked, and the verification filter beside it is scattered. A
/// version-1 file predates blocking entirely and is scattered by construction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BitLayout {
    /// [`PROBES`] bit positions spread across the whole array, ecloop's original schedule.
    /// Every probe is an independent random access.
    Scattered,
    /// One block chosen per hash, and every probe inside it -- so a lookup costs one cache
    /// miss whatever the probes do, at the price of more bits per entry.
    Blocked,
}

impl BitLayout {
    /// The layout a header code names, or `None` for one this build does not know.
    ///
    /// Unknown is refused rather than guessed at: a layout is the arithmetic that decides
    /// which bits a hash probes, and reading a file with the wrong one finds nothing while
    /// looking exactly like a filter that holds nothing.
    fn from_code(code: u32) -> Option<BitLayout> {
        match code {
            0 => Some(BitLayout::Scattered),
            1 => Some(BitLayout::Blocked),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            BitLayout::Scattered => "scattered",
            BitLayout::Blocked => "blocked",
        }
    }
}

/// An address form a filter entry can have come from, as `keyscan bf-gen` numbers them.
///
/// A version-2 header carries these as a bitmask, which is the only thing in the file that
/// says what went into it. Mirrors `Kind` in `keyscan`'s `src/address.rs`: the bit numbers
/// are the file format and cannot be renumbered on this side.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// `1...`: the hash160 of a public key, compressed or not.
    P2pkh,
    /// `3...`: the hash160 of a redeem script. P2SH-P2WPKH is one of those scripts, and a
    /// `3...` string does not say which it wraps, so a filter takes them all.
    P2sh,
    /// `bc1q...` with a 20-byte program, which is the same hash160 a compressed P2PKH holds.
    P2wpkh,
    /// `bc1q...` with a 32-byte program: the SHA-256 of a witness script, which no key
    /// derivation reaches. `bf-gen` counts these and leaves them out.
    P2wsh,
    /// `bc1p...`: the leading 20 bytes of a 32-byte taproot output key.
    P2tr,
    /// Witness versions 2 to 16, carried on the same truncation rule.
    Witness,
    /// 40 hex characters: a tag given directly, of whatever it was a tag of.
    Hex,
}

impl Kind {
    /// Every form, in the order their bits are numbered.
    pub const ALL: [Kind; 7] = [
        Kind::P2pkh,
        Kind::P2sh,
        Kind::P2wpkh,
        Kind::P2wsh,
        Kind::P2tr,
        Kind::Witness,
        Kind::Hex,
    ];

    /// This form's bit in a filter header's kinds mask.
    pub const fn bit(self) -> u32 {
        1 << match self {
            Kind::P2pkh => 0,
            Kind::P2sh => 1,
            Kind::P2wpkh => 2,
            Kind::P2wsh => 3,
            Kind::P2tr => 4,
            Kind::Witness => 5,
            Kind::Hex => 6,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Kind::P2pkh => "p2pkh",
            Kind::P2sh => "p2sh",
            Kind::P2wpkh => "p2wpkh",
            Kind::P2wsh => "p2wsh",
            Kind::P2tr => "p2tr",
            Kind::Witness => "witness",
            Kind::Hex => "hash160",
        }
    }
}

/// What a version-1 filter holds, since it does not say.
///
/// Everything `bf-gen` took before it recorded anything: a hash160 written out, a `1...`, a
/// `3...`, and the 20-byte `bc1q...` that carries the same hash160 as a `1...`. Taproot was
/// not among them, which is why this is a known answer and not a guess -- and every form this
/// sweep derives is in it, so a version-1 filter is swept exactly as it always was.
pub const LEGACY_KINDS: u32 =
    Kind::Hex.bit() | Kind::P2pkh.bit() | Kind::P2sh.bit() | Kind::P2wpkh.bit();

/// The forms in a mask, named, in the order [`Kind::ALL`] numbers them.
///
/// Empty for a mask that records nothing, which is the one thing a real filter's mask never
/// is: `bf-gen` writes what it took, and a version-1 file is read as [`LEGACY_KINDS`].
pub fn kind_names(kinds: u32) -> Vec<&'static str> {
    Kind::ALL
        .iter()
        .filter(|k| kinds & k.bit() != 0)
        .map(|k| k.name())
        .collect()
}

/// The `PROBES` bit indices a scattered lookup reads, in schedule order.
///
/// The scan never calls this -- `contains` folds the same arithmetic into its early-exit
/// loop and never materialises the list. It exists for the GPU's differential test, which
/// compares the two implementations' *schedules* rather than their verdicts: both say
/// "absent" about almost every hash, so agreeing on that says nothing, while agreeing on
/// twenty indices says the fold and the shift schedule are identical.
#[cfg(test)]
pub fn probe_indices(hash160: &[u8; 20]) -> [u64; PROBES] {
    let folded = BloomFilter::fold(hash160);
    let mut out = [0u64; PROBES];
    let mut at = 0;
    for shift in SHIFTS {
        for i in 0..FOLDED_WORDS {
            out[at] = BloomFilter::index(&folded, shift, i);
            at += 1;
        }
    }
    out
}

/// The `PROBES` bit indices a blocked lookup reads, for a filter of `bits` bits.
///
/// The device's half of the same differential test. The block count is an input because it
/// is what the block index is reduced against: two filters of different sizes probe
/// different bits for the same hash, which is the one thing about this schedule that is not
/// a property of the hash alone.
#[cfg(test)]
pub fn block_probe_indices(hash160: &[u8; 20], bits: u64) -> [u64; PROBES] {
    let (origin, positions) = block_probes(hash160, bits / BLOCK_BITS);
    positions.map(|p| origin + p as u64)
}

/// The false-positive rate implied by a filter that is `fill` full, in the layout it holds
/// its bits in.
///
/// Derived from the filter in hand rather than from the rate it was *sized* for, because the
/// two differ whenever a filter holds more or fewer entries than intended -- and quoting the
/// design rate would understate the junk an over-full filter produces, which is exactly the
/// case an operator needs to be told about.
///
/// The two layouts do not share a formula. A scattered lookup's probes are independent across
/// the whole array, so it false-positives when all [`PROBES`] of them land on set bits: the
/// fill to that power. A blocked lookup is confined to one block, so its rate turns on how
/// loaded *that* block is -- and the blocks that received more than their share of entries
/// dominate the answer, which the array's average fill cannot express. So the bits an entry
/// occupies are recovered from the fill first, through the same model `keyscan bf-gen` sized
/// the file with, and the rate follows from those.
pub fn false_positive_rate(fill: f64, layout: BitLayout) -> f64 {
    match layout {
        BitLayout::Scattered => fill.powi(PROBES as i32),
        // The two ends, where the model is exact and the bisection below is not: with no bits
        // set nothing can pass, and with every bit set everything does. Neither is bracketed
        // by a finite number of bits per entry, so both are answered before it.
        BitLayout::Blocked if fill <= 0.0 => 0.0,
        BitLayout::Blocked if fill >= 1.0 => 1.0,
        BitLayout::Blocked => blocked_rate(blocked_bits_at_fill(fill)),
    }
}

/// The false-positive rate of a blocked filter holding `b` bits per entry.
///
/// A block receives a Poisson number of entries about a mean of `BLOCK_BITS / b`; one holding
/// `j` of them has each bit set with probability `1 - (1 - k/B)^j`, and admits a stranger when
/// all `k` of its probes land on one. The rate is that averaged over `j`. `keyscan`'s model,
/// transcribed: it is what the file was sized by, so reading a rate off a fill with any other
/// would answer a different question than the one the filter was built to.
fn blocked_rate(bits_per_entry: f64) -> f64 {
    let (k, block) = (PROBES as f64, BLOCK_BITS as f64);
    let lambda = block / bits_per_entry;
    let miss = (1.0 - 1.0 / block).powf(k);
    let (mut total, mut log_pmf) = (0.0f64, -lambda);
    for j in 0..4000u32 {
        if j > 0 {
            log_pmf += lambda.ln() - (j as f64).ln();
        }
        let p = log_pmf.exp();
        if p < 1e-22 && (j as f64) > lambda {
            break;
        }
        total += p * (1.0 - miss.powi(j as i32)).powf(k);
    }
    total
}

/// The share of bits a blocked filter holding `b` bits per entry has set.
///
/// Averaged over the same Poisson spread, which for the fill -- unlike the rate -- has a
/// closed form: `E[x^j] = e^(lambda(x-1))`.
fn blocked_fill(bits_per_entry: f64) -> f64 {
    let lambda = BLOCK_BITS as f64 / bits_per_entry;
    let set = 1.0 - (1.0 - 1.0 / BLOCK_BITS as f64).powf(PROBES as f64);
    1.0 - (-lambda * set).exp()
}

/// The bits per entry a blocked filter at this fill is carrying: the inverse of
/// [`blocked_fill`], by bisection on that same function so the two cannot drift.
fn blocked_bits_at_fill(fill: f64) -> f64 {
    let (mut lo, mut hi) = (1.0f64, 4096.0f64);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        // Fill falls as the bits per entry rise.
        if blocked_fill(mid) > fill {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The SplitMix64 finalizer: two multiplies and three shifts, after which every input bit
/// has reached every output bit. `keyscan`'s `mix64`, and the kernels' `bloom_mix64`.
#[inline]
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Where a hash probes in a blocked filter: the first bit of its block, and the
/// [`PROBES`] positions inside it.
///
/// A tag carries 160 bits and this needs over two hundred -- a block index plus twenty
/// nine-bit positions -- so something has to expand it, and the expansion is not free to be
/// the obvious one. `keyscan` records why at length; the short of it is that double hashing
/// (`base + i*step`) leaves only 256 odd steps in a 512-bit block, two entries sharing a step
/// then cover overlapping arithmetic progressions rather than independent positions, and a
/// filter built for one in ten million measured **three in ten thousand**. So the positions
/// come from four independent mixes instead: one names the block, and the other three carry
/// seven, seven and six positions each, none straddling a word boundary so the kernels can
/// take the same slices.
///
/// This side only reproduces it. Every constant here is part of the file `bf-gen` wrote, and
/// a difference in any of them reads as a filter that holds nothing.
#[inline]
fn block_probes(hash160: &[u8; 20], blocks: u64) -> (u64, [u32; PROBES]) {
    let w = words(hash160);
    // All five words into one value first, so every bit of the tag reaches every mix.
    let fold = ((w[0] << 32) | w[1])
        ^ (((w[2] << 32) | w[3]).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        ^ (w[4].wrapping_mul(0xff51_afd7_ed55_8ccd));

    let m = [
        mix64(fold ^ BLOCK_SALT[0]),
        mix64(fold ^ BLOCK_SALT[1]),
        mix64(fold ^ BLOCK_SALT[2]),
    ];

    // Multiply-and-shift rather than a remainder: `(k * n) >> 32` for a uniform 32-bit `k` is
    // uniform over `0..n`, and costs no division -- which the GPU side has no instruction for
    // at all.
    let block = ((mix64(fold) >> 32) * blocks) >> 32;

    let mask = BLOCK_BITS as u32 - 1;
    let positions = std::array::from_fn(|i| {
        ((m[i / POSITIONS_PER_WORD] >> (9 * (i % POSITIONS_PER_WORD))) as u32) & mask
    });
    (block * BLOCK_BITS, positions)
}

/// A hash160 as the five big-endian 32-bit words every schedule here is written over.
#[inline]
fn words(hash160: &[u8; 20]) -> [u64; FOLDED_WORDS] {
    std::array::from_fn(|i| {
        u32::from_be_bytes(hash160[i * 4..i * 4 + 4].try_into().unwrap()) as u64
    })
}

/// Re-mix a hash160 the way `keyscan bf-gen` does before it goes into the verification
/// filter, so the same entry lands on bits unrelated to the ones the primary set.
///
/// This is what makes the second filter a second *opinion*. Two filters over the same
/// entries with the same schedule would agree on every false positive and rule out
/// nothing; re-salted, the two are independent tests and their rates multiply.
///
/// A bijection -- the five big-endian words reversed, each XORed with a different constant
/// -- so two distinct hashes stay distinct. A transform that merged any pair would add
/// false positives of its own, which is the opposite of the job.
///
/// The constants and the word order are `keyscan`'s and cannot drift from them: this side
/// only reads a file the other side wrote, and a resalt that disagreed would look exactly
/// like a filter that holds nothing -- every candidate rejected, every real find lost, and
/// no error anywhere. See `resalting_matches_keyscan` for the vector that pins it.
pub fn resalt(hash160: &[u8; 20]) -> [u8; 20] {
    const SALT: [u32; 5] = [
        0x9e37_79b9,
        0x85eb_ca6b,
        0xc2b2_ae35,
        0x27d4_eb2f,
        0x1656_67b1,
    ];
    let mut out = [0u8; 20];
    for (i, salt) in SALT.iter().enumerate() {
        let src = (4 - i) * 4;
        let word = u32::from_be_bytes([
            hash160[src],
            hash160[src + 1],
            hash160[src + 2],
            hash160[src + 3],
        ]) ^ salt;
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// The page size to align the filter's allocation to.
///
/// Queried rather than assumed: it is 4 KB on most Linux and 16 KB on Apple Silicon,
/// and the GPU path cares about the real value. 16 KB is the safe fallback where it
/// cannot be asked for -- over-aligning wastes at most one page of a multi-gigabyte
/// allocation, while under-aligning is what `newBufferWithBytesNoCopy` rejects.
fn page_size() -> usize {
    #[cfg(unix)]
    {
        // SAFETY: sysconf takes no pointers and cannot fail meaningfully here; a
        // non-positive answer would mean a broken libc, and the fallback covers it.
        let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if n > 0 {
            return n as usize;
        }
    }
    16384
}

/// A page-aligned owned array of `u64`.
///
/// `Vec<u64>` would be the obvious choice and is what this used to be. It is not enough
/// because Metal's `newBufferWithBytesNoCopy` -- the thing that lets the GPU read the
/// filter with no copy and no second 7.6 GB of residency -- requires *both* the pointer
/// and the length to be page-aligned, and `Vec` guarantees neither.
///
/// This is unconditional rather than hidden behind a GPU cfg. Alignment costs nothing on
/// the CPU side (the allocator rounds a multi-gigabyte request up to whole pages anyway),
/// it makes `MADV_HUGEPAGE` cover whole pages on Linux instead of a ragged range, and one
/// allocation path is worth more than the bytes a cfg would save.
struct Words {
    ptr: NonNull<u64>,
    /// Length in `u64`s, which is what the probe indexes by.
    len: usize,
    layout: Layout,
}

// SAFETY: the allocation is written once during `open` and is immutable for the rest of
// its life, so handing `&Words` to every scanning thread is a shared read of frozen
// memory. The raw pointer is what blocks the automatic impls; nothing else here does.
unsafe impl Send for Words {}
unsafe impl Sync for Words {}

impl Words {
    /// Allocate `len` zeroed `u64`s, page-aligned and rounded up to a whole page.
    fn zeroed(len: usize) -> Result<Self> {
        let page = page_size();
        // Checked, not wrapped: `len` comes from a file header, and in release a wrapped
        // multiply here would allocate a page while `len` still claimed exabytes -- which
        // `Deref` would then hand out as a slice.
        let bytes = len
            .checked_mul(8)
            .and_then(|b| b.checked_next_multiple_of(page))
            .with_context(|| {
                format!("a {len}-word filter is larger than this machine can address")
            })?
            .max(page);
        let layout = Layout::from_size_align(bytes, page)
            .with_context(|| format!("a {bytes}-byte page-aligned layout"))?;
        // SAFETY: `bytes` is non-zero (the `.max(page)` above), so this is a valid
        // request. Zeroed means the pages arrive untouched from the OS, which is what
        // keeps a multi-gigabyte allocation free until the read below makes it resident.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr as *mut u64)
            .with_context(|| format!("allocating {bytes} bytes for the filter"))?;
        Ok(Self { ptr, len, layout })
    }

    /// The whole allocation as bytes, for reading the file into and for handing to a GPU.
    ///
    /// This is the padded length, not `len * 8`: the tail beyond the declared word count
    /// is zero and never probed, and the GPU wants the page-aligned length.
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: we own `layout.size()` initialised bytes, and `u8` has no alignment or
        // validity constraints a `u64` allocation could violate.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut u8, self.layout.size()) }
    }
}

impl std::ops::Deref for Words {
    type Target = [u64];

    fn deref(&self) -> &[u64] {
        // SAFETY: `ptr` is a live, aligned allocation of at least `len` u64s, fully
        // initialised by `alloc_zeroed` and then by the file read.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for Words {
    fn drop(&mut self) {
        // SAFETY: same pointer and layout the allocation was made with.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, self.layout) };
    }
}

impl std::fmt::Debug for Words {
    /// The contents are gigabytes of bit noise; the size is the only useful thing.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Words({} words)", self.len)
    }
}

#[derive(Debug)]
pub struct BloomFilter {
    /// The whole bit space, in memory, in native word order.
    words: Words,
    /// `words.len() * 64`, precomputed because it divides every probe index.
    bits: u64,
    /// How the bits are arranged, and so which of the two schedules a lookup runs.
    ///
    /// From the header, never assumed: `keyscan bf-gen` writes a blocked primary and a
    /// scattered companion, so a sweep reads one of each and the wrong schedule on either
    /// finds nothing while looking exactly like a filter that holds nothing.
    layout: BitLayout,
    /// Which address forms went in, as [`Kind::bit`] flags. See [`BloomFilter::kinds`].
    kinds: u32,
}

impl BloomFilter {
    /// Read the whole filter into memory.
    ///
    /// This allocates the size of the file, which for a real filter is gigabytes.
    /// That is the point: see the module comment.
    pub fn open(path: &Path) -> Result<Self> {
        // Before the open, so a mistyped or wrong-provenance path costs nothing and says
        // one thing. See `FILTER_EXT` for why the name is load-bearing.
        if path.extension().is_none_or(|e| e != FILTER_EXT) {
            bail!(
                "{} is not a .{FILTER_EXT} filter. This sweep reads the filters \
                 `keyscan bf-gen` builds, which are the ones that carry P2SH script hashes \
                 -- without them the P2SH-P2WPKH third of every seed matches nothing, and \
                 the sweep finishes looking complete. Build one with `keyscan bf-gen \
                 addresses.txt -o addresses.{FILTER_EXT}`.",
                path.display()
            );
        }
        let mut file =
            File::open(path).with_context(|| format!("opening bloom filter {}", path.display()))?;
        let len = file.metadata()?.len();
        if len < HEADER_LEN_V1 as u64 {
            bail!("{} is too short to be a .{FILTER_EXT} file", path.display());
        }

        let mut header = [0u8; HEADER_LEN_V1];
        file.read_exact(&mut header)
            .with_context(|| format!("reading the header of {}", path.display()))?;
        let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
        let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
        let words = u64::from_le_bytes(header[8..16].try_into().unwrap());

        if magic != MAGIC {
            bail!(
                "{} is not a bloom filter (magic {magic:#010x}, expected {MAGIC:#010x}); \
                 build one with `keyscan bf-gen`",
                path.display()
            );
        }
        // Version 1 is the same bit array behind a shorter header. What it does not carry
        // has one possible answer -- it was written before there was any layout but the
        // scattered one, and before `bf-gen` recorded what it took -- so it is read rather
        // than refused. See `VERSION_LEGACY` and `LEGACY_KINDS`.
        let header_len = match version {
            VERSION_LEGACY => HEADER_LEN_V1,
            VERSION => HEADER_LEN,
            _ => bail!(
                "{} is bloom filter version {version}, this build understands \
                 {VERSION_LEGACY} and {VERSION}. Rebuild it with `keyscan bf-gen`.",
                path.display()
            ),
        };

        let (kinds, layout) = if version == VERSION_LEGACY {
            (LEGACY_KINDS, BitLayout::Scattered)
        } else {
            if len < HEADER_LEN as u64 {
                bail!(
                    "{} declares a version-{VERSION} header but is too short to hold one",
                    path.display()
                );
            }
            let mut rest = [0u8; HEADER_LEN - HEADER_LEN_V1];
            file.read_exact(&mut rest)
                .with_context(|| format!("reading the header of {}", path.display()))?;
            let kinds = u32::from_le_bytes(rest[0..4].try_into().unwrap());
            let code = u32::from_le_bytes(rest[4..8].try_into().unwrap());
            // Refused rather than guessed at: the layout decides which bits a hash probes,
            // and reading a file with the wrong one is a sweep that reports nothing and
            // says nothing about why.
            let layout = BitLayout::from_code(code).with_context(|| {
                format!(
                    "{} lays its bits out as {code}, which this build does not know. \
                     Rebuild it with `keyscan bf-gen`, or update this scanner.",
                    path.display()
                )
            })?;
            (kinds, layout)
        };

        // A filter with no bits at all divides by zero on the first probe rather than
        // reporting anything, and it could never match in any case.
        if words == 0 {
            bail!(
                "{} declares a filter of zero words, so it holds no addresses and nothing \
                 could ever match it. Rebuild it with `keyscan bf-gen`.",
                path.display()
            );
        }
        // Checked: the declared count is untrusted, and a wrapped multiply in release would
        // let a 16-byte file claim exabytes of bits and pass the length check below.
        let expected = words
            .checked_mul(8)
            .and_then(|payload| payload.checked_add(header_len as u64))
            .with_context(|| {
                format!(
                    "{} declares {words} words, which is not a real size",
                    path.display()
                )
            })?;
        if len != expected {
            bail!(
                "{} is truncated: header declares {words} words ({expected} bytes) but the file is {len} bytes",
                path.display()
            );
        }
        let word_count = usize::try_from(words).with_context(|| {
            format!(
                "{} declares {words} words, more than this machine can address",
                path.display()
            )
        })?;

        // `words` is trusted only after the length check above agreed with it, so
        // this allocation is the size of a file that exists. It comes from the
        // allocator zeroed, i.e. as untouched anonymous pages, and the read below is
        // what makes it resident.
        let mut bits = Words::zeroed(word_count)?;
        // The declared words, not the padded length: the page-alignment tail stays zero
        // and is never probed, because `self.bits` is derived from `words`.
        let raw = &mut bits.as_bytes_mut()[..word_count * 8];
        file.read_exact(raw)
            .with_context(|| format!("reading bloom filter {}", path.display()))?;
        if cfg!(target_endian = "big") {
            // SAFETY: the bytes just read are a live, initialised, aligned run of
            // `words` u64s; u64 has no padding or invalid bit patterns.
            let native =
                unsafe { std::slice::from_raw_parts_mut(raw.as_mut_ptr() as *mut u64, word_count) };
            for word in native {
                *word = word.to_le();
            }
        }
        Self::advise_hugepages(&bits);

        // A blocked filter's probe names a block outright and never reduces, so every block
        // its arithmetic can pick has to exist -- which `bf-gen` guarantees by rounding the
        // array up to whole blocks, and which is checked here because the consequence of a
        // ragged tail is a read past the end of the array.
        let bit_count = words * 64;
        if layout == BitLayout::Blocked && !bit_count.is_multiple_of(BLOCK_BITS) {
            bail!(
                "{} is blocked but holds {bit_count} bits, which is not a whole number of \
                 {BLOCK_BITS}-bit blocks. Rebuild it with `keyscan bf-gen`.",
                path.display()
            );
        }

        Ok(Self {
            words: bits,
            bits: bit_count,
            layout,
            kinds,
        })
    }

    /// How this filter's bits are arranged, and so which schedule a lookup runs.
    pub fn layout(&self) -> BitLayout {
        self.layout
    }

    /// Which address forms went into this filter, as [`Kind::bit`] flags.
    ///
    /// The only thing in the file that says what it holds. A sweep derives three hash forms
    /// and this is what says whether the filter was ever given any of them -- see
    /// [`crate::target::Target::unmatchable_forms`], which is the check it exists for.
    pub fn kinds(&self) -> u32 {
        self.kinds
    }

    /// Blocks in this filter. Meaningless for a scattered one, and nothing asks.
    #[inline]
    fn blocks(&self) -> u64 {
        self.bits / BLOCK_BITS
    }

    /// The bit space's length, which every probe index is reduced against.
    pub fn bit_count(&self) -> u64 {
        self.bits
    }

    /// The raw allocation, for handing to a GPU without copying it.
    ///
    /// Page-aligned in both pointer and length -- see `Words` -- which is exactly what
    /// `newBufferWithBytesNoCopy` requires and what a `Vec<u64>` could not promise. The
    /// length returned is the padded one; the tail past `bit_count` is zero and unreachable
    /// because every probe is reduced modulo `bits` first.
    pub fn raw_words(&self) -> (*const u8, usize) {
        (self.words.ptr.as_ptr() as *const u8, self.words.layout.size())
    }

    /// Probes land at unpredictable offsets across gigabytes, so ask for huge pages
    /// where the OS offers the choice -- at this size the TLB miss on every probe is
    /// a measurable share of the cost.
    ///
    /// `MADV_HUGEPAGE` is a Linux extension, and on the other platforms the decision
    /// is not the caller's to make. macOS has no equivalent advice: its superpage
    /// flags (`VM_FLAGS_SUPERPAGE_SIZE_2MB` and `..._ANY`, passed to `mmap`) are
    /// rejected outright with `EINVAL` on Apple Silicon, tested rather than assumed.
    /// Its 16 KB base page does at least cost a quarter of the TLB pressure Linux's
    /// 4 KB page would, and `contains_batch` is what covers the rest, by overlapping
    /// the page walks instead of serialising them.
    #[cfg(target_os = "linux")]
    fn advise_hugepages(bits: &[u64]) {
        // SAFETY: ptr/len describe our own live allocation. madvise is advisory; a
        // failure (e.g. no THP support) only costs performance, so it is ignored.
        unsafe {
            libc::madvise(
                bits.as_ptr() as *mut libc::c_void,
                std::mem::size_of_val(bits),
                libc::MADV_HUGEPAGE,
            );
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn advise_hugepages(_bits: &[u64]) {}

    #[inline]
    fn probe(&self, index: u64) -> bool {
        let index = index % self.bits;
        // SAFETY: `index < self.bits == self.words.len() * 64` after the reduction
        // above, so the word index is in bounds. The unchecked read removes a bounds
        // check from the innermost loop.
        let word = unsafe { *self.words.get_unchecked((index / 64) as usize) };
        word & (1u64 << (index % 64)) != 0
    }

    /// The five overlapping 64-bit words ecloop folds a hash160 into.
    ///
    /// It treats the hash as five big-endian u32s and pairs them cyclically. See
    /// blf_has() in ecloop/lib/utils.c.
    #[inline]
    fn fold(hash160: &[u8; 20]) -> [u64; FOLDED_WORDS] {
        let w = words(hash160);
        [
            w[0] << 32 | w[1],
            w[2] << 32 | w[3],
            w[4] << 32 | w[0],
            w[1] << 32 | w[2],
            w[3] << 32 | w[4],
        ]
    }

    /// Read a bit the schedule has already placed inside the array.
    ///
    /// A blocked probe names its bit outright -- the block exists, `open` having checked the
    /// array is a whole number of them, and the position is inside it -- so there is nothing
    /// left to reduce, and reducing anyway would wrap an index that is already in range.
    #[inline]
    fn probe_at(&self, index: u64) -> bool {
        // SAFETY: `index < self.bits`, the block being one of `self.bits / BLOCK_BITS` and
        // the position inside it: `origin + pos <= (blocks - 1) * BLOCK_BITS + BLOCK_BITS - 1`.
        let word = unsafe { *self.words.get_unchecked((index / 64) as usize) };
        word & (1u64 << (index % 64)) != 0
    }

    /// The bit a lookup for this hash reads first, and everything the rest of it needs.
    ///
    /// The two layouts differ in what "first" costs to work out but not in what it buys: one
    /// memory access that rules the hash out `1 - fill` of the time. Blocked probes then all
    /// land in the line that access pulled in, which is the whole point of the layout.
    #[inline]
    fn first_probe(&self, hash160: &[u8; 20]) -> bool {
        match self.layout {
            BitLayout::Scattered => self.probe(Self::index(&Self::fold(hash160), SHIFTS[0], 0)),
            BitLayout::Blocked => {
                let (origin, positions) = block_probes(hash160, self.blocks());
                self.probe_at(origin + positions[0] as u64)
            }
        }
    }

    /// The bit index of probe `(shift, i)` for a folded hash.
    #[inline]
    fn index(a: &[u64; FOLDED_WORDS], shift: u32, i: usize) -> u64 {
        a[i] << shift | a[(i + 1) % FOLDED_WORDS] >> shift
    }

    /// Is this hash160 possibly in the set?
    ///
    /// False positives occur at the rate the filter was built for; false negatives
    /// never do.
    ///
    /// Returns on the first clear bit, so a miss -- the overwhelmingly common case --
    /// typically costs about two memory probes rather than twenty. On a blocked filter the
    /// early exit buys much less, all twenty probes being one cache line, and it is kept for
    /// the arithmetic rather than the memory.
    #[inline]
    pub fn contains(&self, hash160: &[u8; 20]) -> bool {
        match self.layout {
            BitLayout::Scattered => {
                let a = Self::fold(hash160);
                for shift in SHIFTS {
                    for i in 0..FOLDED_WORDS {
                        if !self.probe(Self::index(&a, shift, i)) {
                            return false;
                        }
                    }
                }
                true
            }
            BitLayout::Blocked => {
                let (origin, positions) = block_probes(hash160, self.blocks());
                positions
                    .iter()
                    .all(|&p| self.probe_at(origin + p as u64))
            }
        }
    }

    /// Test a whole chunk of hash160s, appending the indices of the survivors to
    /// `hits`.
    ///
    /// This exists for one reason, and it is not tidiness. The filter is gigabytes,
    /// probes land uniformly across it, and a lookup ends at its first clear bit
    /// roughly `1 - fill` of the time -- so the cost of a sweep is dominated by one
    /// DRAM access per hash160, and at 7.6 GB that access is really a TLB miss and a
    /// page walk on top of the DRAM latency. Issued one at a time, as
    /// `contains`-per-hash does, each of those is a full serialised round trip with
    /// the machine's memory parallelism sitting idle.
    ///
    /// So the first probe of every hash in the chunk is taken in one pass, with no
    /// branch between them. They are independent loads, the out-of-order window
    /// covers many of them at once, and the misses overlap instead of queueing. Only
    /// the few that survive pay for the remaining 19 probes.
    ///
    /// A blocked filter has the same shape for the same reason -- one access per hash decides
    /// almost all of them -- and gains from it doubly: the remaining nineteen probes read the
    /// line the first one has already pulled in.
    ///
    /// This is a pure batching of `contains` and agrees with it hash for hash.
    pub fn contains_batch(&self, hashes: &[[u8; 20]], hits: &mut Vec<u32>) {
        assert!(hashes.len() <= PROBE_CHUNK, "chunk longer than PROBE_CHUNK");

        // Pass one: the discriminating probe, for every hash, back to back. Written
        // as a fold into a buffer rather than a filter so nothing here can branch on
        // a value that has not arrived yet.
        let mut survived = [false; PROBE_CHUNK];
        for (slot, hash) in survived.iter_mut().zip(hashes) {
            *slot = self.first_probe(hash);
        }

        // Pass two: the other 19, for the ~16% that got here.
        for (i, hash) in hashes.iter().enumerate() {
            if survived[i] && self.contains(hash) {
                hits.push(i as u32);
            }
        }
    }

    /// Bytes of RAM the filter occupies, which is also its size on disk less the
    /// header.
    pub fn size_bytes(&self) -> usize {
        std::mem::size_of_val(&self.words[..])
    }

    /// Fraction of bits set, sampled rather than counted in full. Reported at
    /// startup because it is the one cheap sanity check that the filter holds what
    /// the operator thinks it holds.
    ///
    /// A filter loaded to the capacity `keyscan bf-gen` sized it for sits near **a third**,
    /// not the half an optimally-parameterised filter would: k is fixed at 20, and the sizing
    /// solves for the bits that k needs rather than for the k the bits would prefer. A blocked
    /// primary at p=1e-7 takes 49.6 bits an entry and fills to **0.327**; a scattered
    /// verification filter at p=1e-10 takes 52.6 and fills to **0.316**. Well under either
    /// means the filter holds fewer entries than it was sized for, and ~0.0 means an empty or
    /// wrong file. [`BloomFilter::false_positive_rate`] is what turns the number into the one
    /// that matters.
    pub fn sampled_fill_ratio(&self, samples: usize) -> f64 {
        let stride = (self.words.len() / samples.max(1)).max(1);
        let mut set = 0u64;
        let mut seen = 0u64;
        for word in self.words.iter().step_by(stride) {
            set += word.count_ones() as u64;
            seen += 64;
        }
        if seen == 0 {
            0.0
        } else {
            set as f64 / seen as f64
        }
    }

    /// The rate at which this filter reports a hash160 it was never given, read out of the
    /// bits actually set and the layout they are set in.
    ///
    /// The layout is not a detail here: at the same fill the two schedules admit strangers at
    /// very different rates, and a blocked filter read as a scattered one is quoted a rate
    /// orders of magnitude off. See [`false_positive_rate`].
    pub fn false_positive_rate(&self, samples: usize) -> f64 {
        false_positive_rate(self.sampled_fill_ratio(samples), self.layout)
    }
}

/// Building and finding filters to test against, shared with [`crate::target`]'s tests.
///
/// The writer lives here rather than there because the format is this module's, and a
/// second transcription of it in another file is a second thing to keep in step with
/// `keyscan bf-gen`.
#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use std::path::PathBuf;

    /// The real-filter tests need the operator's own 7.6 GB file and a `keyscan`
    /// binary, neither of which belongs in the repo or in a checked-in path. Every
    /// such input is named by an env var and nothing is guessed: a test whose inputs
    /// are unset skips rather than fails, so a plain `cargo test` still passes.
    /// Watch for the skip message if you expected these to run.
    ///
    /// ```sh
    /// KEYFORGE_KEYSCAN=../allkeys-keyscan/target/release/keyscan \
    /// KEYFORGE_FILTER=../allkeys-keyscan/addresses.bf cargo test
    /// ```
    pub(crate) fn env_path(var: &str) -> Option<PathBuf> {
        std::env::var_os(var).map(PathBuf::from)
    }

    /// A textbook transcription of the `blf_add` this schedule descends from, used to
    /// build a filter in-process so the probes can be tested without a 7.6 GB file.
    pub(crate) fn reference_add(bits: &mut [u64], hash: &[u8; 20]) {
        let w: Vec<u64> = hash
            .chunks_exact(4)
            .map(|c| u32::from_be_bytes(c.try_into().unwrap()) as u64)
            .collect();
        let a = [
            w[0] << 32 | w[1],
            w[2] << 32 | w[3],
            w[4] << 32 | w[0],
            w[1] << 32 | w[2],
            w[3] << 32 | w[4],
        ];
        let total = bits.len() as u64 * 64;
        for shift in SHIFTS {
            for i in 0..5 {
                let idx = (a[i] << shift | a[(i + 1) % 5] >> shift) % total;
                bits[(idx / 64) as usize] |= 1u64 << (idx % 64);
            }
        }
    }

    /// `reference_add` for a blocked filter: the four mixes, the block they pick, and the
    /// twenty positions inside it.
    ///
    /// Written through `block_probes` rather than transcribed a second time. There is nothing
    /// to hold it against: the scattered schedule descends from ecloop and a textbook version
    /// of it is worth writing out, while this one exists only in the file `keyscan bf-gen`
    /// wrote, and the test that pins it is the one that runs against a real filter.
    pub(crate) fn reference_add_blocked(bits: &mut [u64], hash: &[u8; 20]) {
        let blocks = bits.len() as u64 * 64 / BLOCK_BITS;
        let (origin, positions) = block_probes(hash, blocks);
        for p in positions {
            let idx = origin + p as u64;
            bits[(idx / 64) as usize] |= 1u64 << (idx % 64);
        }
    }

    /// A scattered version-2 filter holding what `bf-gen` took before it recorded anything,
    /// which is what most of these tests want.
    pub(crate) fn write_filter(path: &Path, bits: &[u64]) {
        write_filter_as(path, bits, BitLayout::Scattered, LEGACY_KINDS);
    }

    /// A version-2 filter in a chosen layout, holding a chosen set of forms.
    pub(crate) fn write_filter_as(path: &Path, bits: &[u64], layout: BitLayout, kinds: u32) {
        let code = match layout {
            BitLayout::Scattered => 0u32,
            BitLayout::Blocked => 1u32,
        };
        let mut out = Vec::with_capacity(HEADER_LEN + bits.len() * 8);
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(bits.len() as u64).to_le_bytes());
        out.extend_from_slice(&kinds.to_le_bytes());
        out.extend_from_slice(&code.to_le_bytes());
        for w in bits {
            out.extend_from_slice(&w.to_le_bytes());
        }
        std::fs::write(path, out).unwrap();
    }

    /// A version-1 filter: the same bit array behind the shorter header, with nothing said
    /// about layout or contents. Filters in the field are still written this way.
    pub(crate) fn write_filter_v1(path: &Path, bits: &[u64]) {
        let mut out = Vec::with_capacity(HEADER_LEN_V1 + bits.len() * 8);
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&VERSION_LEGACY.to_le_bytes());
        out.extend_from_slice(&(bits.len() as u64).to_le_bytes());
        for w in bits {
            out.extend_from_slice(&w.to_le_bytes());
        }
        std::fs::write(path, out).unwrap();
    }

    pub(crate) fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("keyforge-tests");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;
    use std::process::Command;

    /// A hash the filter was given must come back, and one it was not must not -- in either
    /// layout, since a sweep reads one of each.
    #[test]
    fn membership_round_trips_through_the_file_format() {
        let members: Vec<[u8; 20]> = (0u8..64)
            .map(|i| {
                let mut h = [0u8; 20];
                for (j, b) in h.iter_mut().enumerate() {
                    *b = i.wrapping_mul(37).wrapping_add(j as u8 * 11);
                }
                h
            })
            .collect();

        for layout in [BitLayout::Scattered, BitLayout::Blocked] {
            let mut bits = vec![0u64; 4096];
            for m in &members {
                match layout {
                    BitLayout::Scattered => reference_add(&mut bits, m),
                    BitLayout::Blocked => reference_add_blocked(&mut bits, m),
                }
            }

            let path = scratch(&format!("roundtrip-{}.bf", layout.name()));
            write_filter_as(&path, &bits, layout, LEGACY_KINDS);
            let filter = BloomFilter::open(&path).unwrap();
            assert_eq!(filter.layout(), layout, "the header records the layout");

            for m in &members {
                assert!(
                    filter.contains(m),
                    "member {} must be present in a {} filter",
                    hex::encode(m),
                    layout.name()
                );
            }

            // With 64 members in 256 Kib of bits, false positives should be vanishing.
            let absent = (0..2000).filter(|i| {
                let mut h = [0u8; 20];
                h[..4].copy_from_slice(&(0xdead_0000u32 + i).to_be_bytes());
                filter.contains(&h)
            });
            assert_eq!(
                absent.count(),
                0,
                "unexpected false positives in a sparse {} filter",
                layout.name()
            );
        }
    }

    /// A version-1 filter has no layout and no kinds mask, and both have one right answer.
    /// It is the format every filter in the field was built in, so it has to keep sweeping
    /// exactly as it did.
    #[test]
    fn a_version_1_filter_reads_as_scattered_and_as_what_bf_gen_took_then() {
        let members: Vec<[u8; 20]> = (0..64).map(|i: u32| {
            let mut h = [0u8; 20];
            h[..4].copy_from_slice(&i.wrapping_mul(2_654_435_761).to_be_bytes());
            h
        }).collect();
        let mut bits = vec![0u64; 4096];
        for m in &members {
            reference_add(&mut bits, m);
        }
        let path = scratch("legacy-v1.bf");
        write_filter_v1(&path, &bits);

        let filter = BloomFilter::open(&path).unwrap();
        assert_eq!(filter.layout(), BitLayout::Scattered);
        assert_eq!(filter.kinds(), LEGACY_KINDS);
        for m in &members {
            assert!(filter.contains(m), "a v1 filter must read back what went into it");
        }
    }

    /// A layout this build does not know is refused rather than read as one it does.
    ///
    /// The layout decides which bits a hash probes. Reading a file with the wrong schedule
    /// finds nothing, which is indistinguishable from a filter that holds nothing -- so the
    /// unknown code has to stop the run rather than be defaulted away.
    #[test]
    fn refuses_a_bit_layout_it_does_not_know() {
        let mut bits = vec![0u64; 512];
        reference_add(&mut bits, &[7u8; 20]);
        let path = scratch("future-layout.bf");
        write_filter(&path, &bits);

        // Byte 20 of the header is the layout code; 2 is a layout no build has yet.
        let mut raw = std::fs::read(&path).unwrap();
        raw[20] = 2;
        std::fs::write(&path, &raw).unwrap();

        let err = BloomFilter::open(&path).unwrap_err().to_string();
        assert!(
            err.contains("lays its bits out as 2") && err.contains("keyscan bf-gen"),
            "the refusal has to name the code and where a readable filter comes from: {err}"
        );
    }

    /// A blocked filter's probe names a block outright and never reduces, so an array that is
    /// not a whole number of blocks would let it read past the end. `bf-gen` rounds up and
    /// this is what checks that it did.
    #[test]
    fn refuses_a_blocked_filter_that_is_not_whole_blocks() {
        let mut bits = vec![0u64; 4096];
        reference_add_blocked(&mut bits, &[3u8; 20]);
        bits.push(0); // 4097 words: 8 words short of a whole block
        let path = scratch("ragged-blocks.bf");
        write_filter_as(&path, &bits, BitLayout::Blocked, LEGACY_KINDS);

        let err = BloomFilter::open(&path).unwrap_err().to_string();
        assert!(err.contains("whole number of 512-bit blocks"), "{err}");
    }

    /// Every probe of a blocked lookup has to land in one 512-bit block, which is the entire
    /// claim the layout makes and the only thing that makes it worth its extra bits.
    #[test]
    fn every_blocked_probe_lands_in_one_block() {
        for i in 0..2000u32 {
            let mut h = [0u8; 20];
            h[..4].copy_from_slice(&i.to_be_bytes());
            h[8..12].copy_from_slice(&i.wrapping_mul(2_654_435_761).to_be_bytes());
            let blocks = 1_000_003; // deliberately not a power of two
            let idx = block_probe_indices(&h, blocks * BLOCK_BITS);
            let block = idx[0] / BLOCK_BITS;
            assert!(block < blocks, "block {block} is outside the array");
            for probe in idx {
                assert_eq!(probe / BLOCK_BITS, block, "probe {probe} left the block");
            }
        }
    }

    /// `contains_batch` is a pure batching of `contains`, so `contains` is its whole
    /// specification and the two must never disagree.
    ///
    /// The filter is deliberately loaded until it false-positives freely: that is what
    /// exercises the second pass, which a sparse filter would leave almost unvisited.
    /// Every chunk length up to the maximum is checked, because the two passes index
    /// the same buffer by different routes and an off-by-one between them would
    /// otherwise only show at one particular length.
    #[test]
    fn batched_lookups_agree_with_single_lookups() {
        let hash = |i: u32| {
            let mut h = [0u8; 20];
            for (j, b) in h.iter_mut().enumerate() {
                *b = (i.wrapping_mul(2_654_435_761).rotate_left(j as u32 * 5)) as u8;
            }
            h
        };

        for layout in [BitLayout::Scattered, BitLayout::Blocked] {
            let mut bits = vec![0u64; 512];
            for i in 0..400 {
                match layout {
                    BitLayout::Scattered => reference_add(&mut bits, &hash(i)),
                    BitLayout::Blocked => reference_add_blocked(&mut bits, &hash(i)),
                }
            }

            let path = scratch(&format!("batched-{}.bf", layout.name()));
            write_filter_as(&path, &bits, layout, LEGACY_KINDS);
            let filter = BloomFilter::open(&path).unwrap();

            // Members and non-members interleaved, so survivors land at both ends of a
            // chunk and in its middle.
            let hashes: Vec<[u8; 20]> = (0..PROBE_CHUNK as u32)
                .map(|i| if i % 3 == 0 { hash(i) } else { hash(i + 10_000) })
                .collect();
            assert!(
                hashes.iter().filter(|h| filter.contains(h)).count() > PROBE_CHUNK / 4,
                "the fixture must produce plenty of survivors, or pass two goes untested"
            );

            let mut hits = Vec::new();
            for len in 0..=PROBE_CHUNK {
                let chunk = &hashes[..len];
                hits.clear();
                filter.contains_batch(chunk, &mut hits);

                let expected: Vec<u32> = (0..len as u32)
                    .filter(|&i| filter.contains(&chunk[i as usize]))
                    .collect();
                assert_eq!(
                    hits,
                    expected,
                    "chunk of {len} disagrees with contains() in a {} filter",
                    layout.name()
                );
            }
        }
    }

    /// The rate a sweep's startup banner quotes, at the fill each layout reaches when it
    /// holds exactly what `keyscan bf-gen` sized it for.
    ///
    /// The two schedules do not share a formula, and the gap is not a rounding difference: a
    /// blocked filter read as a scattered one would be quoted a rate orders of magnitude out,
    /// in the optimistic direction, which is the direction that matters.
    #[test]
    fn the_false_positive_rate_follows_the_fill_and_the_layout() {
        assert_eq!(PROBES, 20);

        // A scattered verification filter at 52.6 bits an entry -- p=1e-10 -- fills to 0.316.
        let verify = false_positive_rate(0.3162, BitLayout::Scattered);
        assert!(
            (5e-11..2e-10).contains(&verify),
            "expected ~1e-10 at the verification filter's fill, got {verify:e}"
        );
        // A blocked primary at 49.6 bits an entry -- p=1e-7 -- fills to 0.327.
        let primary = false_positive_rate(0.3269, BitLayout::Blocked);
        assert!(
            (5e-8..2e-7).contains(&primary),
            "expected ~1e-7 at the primary's fill, got {primary:e}"
        );
        // Which is the point of reading the layout: the same bits, taken as scattered, would
        // be quoted a rate the sweep could not live up to.
        assert!(
            false_positive_rate(0.3269, BitLayout::Scattered) < primary / 100.0,
            "blocking costs rate at a given fill, and the two formulas must show it"
        );

        for layout in [BitLayout::Scattered, BitLayout::Blocked] {
            // An empty filter never false-positives; a saturated one always does.
            assert_eq!(false_positive_rate(0.0, layout), 0.0);
            assert_eq!(false_positive_rate(1.0, layout), 1.0);
            // And it has to be monotonic, or the warning it drives points the wrong way.
            assert!(
                false_positive_rate(0.5, layout) > false_positive_rate(0.3, layout),
                "{} rate must rise with the fill",
                layout.name()
            );
        }
    }

    /// The re-salt has to be `keyscan`'s, bit for bit.
    ///
    /// It is not an arithmetic this side is free to choose: `bf-gen` applied it before
    /// writing the verification filter, so a different one here probes bits nothing ever
    /// set. Every candidate would be rejected, the sweep would end with an empty
    /// `matches.txt`, and nothing would have gone wrong that anyone could see. The vector
    /// is the genesis coinbase hash160 -- the first entry of the real filter's address
    /// list -- transformed by hand from `keyscan`'s constants and word order.
    #[test]
    fn resalting_matches_keyscan() {
        let mut hash = [0u8; 20];
        hex::decode_to_slice("62e907b15cbf27d5425399ebf6f0fb50ebb88f18", &mut hash).unwrap();
        assert_eq!(
            hex::encode(resalt(&hash)),
            "758ff6a1731b313b80e137de7b6bccfa74bf6000"
        );

        // Injective, or it would merge distinct entries and add false positives of its
        // own -- and it has to move the bits, or the second filter would be the first.
        let mut seen = std::collections::HashSet::new();
        for i in 0..50_000u32 {
            let mut h = [0u8; 20];
            h[..4].copy_from_slice(&i.to_be_bytes());
            h[8..12].copy_from_slice(&i.wrapping_mul(2_654_435_761).to_be_bytes());
            assert_ne!(resalt(&h), h, "the re-salt left {i} where it was");
            assert!(seen.insert(resalt(&h)), "the re-salt collided at {i}");
        }
    }

    #[test]
    fn rejects_files_that_are_not_bloom_filters() {
        let path = scratch("bogus.bf");
        std::fs::write(&path, vec![0u8; 64]).unwrap();
        let err = BloomFilter::open(&path).unwrap_err().to_string();
        assert!(err.contains("is not a bloom filter"), "{err}");
    }

    #[test]
    fn rejects_truncated_files() {
        let mut bits = vec![0u64; 128];
        reference_add(&mut bits, &[7u8; 20]);
        let path = scratch("truncated.bf");
        write_filter(&path, &bits);
        let full = std::fs::read(&path).unwrap();
        std::fs::write(&path, &full[..full.len() - 8]).unwrap();
        let err = BloomFilter::open(&path).unwrap_err().to_string();
        assert!(err.contains("truncated"), "{err}");
    }

    /// A header is untrusted input, and the two values that are not merely wrong but
    /// unsound have to be refused on open rather than reached later.
    ///
    /// Zero words divides by zero on the first probe -- every index is reduced modulo the
    /// bit count -- and a word count whose byte length overflows would pass the truncation
    /// check against a tiny file and then be handed out as a slice of that length. Both are
    /// a corrupt or hostile filter taking the scan down, so both stop here.
    #[test]
    fn rejects_headers_that_no_real_filter_could_have() {
        let header = |words: u64| {
            let mut out = Vec::new();
            out.extend_from_slice(&MAGIC.to_le_bytes());
            out.extend_from_slice(&VERSION.to_le_bytes());
            out.extend_from_slice(&words.to_le_bytes());
            out.extend_from_slice(&LEGACY_KINDS.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes()); // scattered
            out
        };

        let path = scratch("empty-header.bf");
        std::fs::write(&path, header(0)).unwrap();
        let err = BloomFilter::open(&path).unwrap_err().to_string();
        assert!(err.contains("zero words"), "{err}");

        // 2^61 words is 2^64 bytes, which wraps to zero: the payload length then agrees
        // with a header-sized file and every later size is computed from a lie.
        let path = scratch("overflowing-header.bf");
        std::fs::write(&path, header(1 << 61)).unwrap();
        let err = BloomFilter::open(&path).unwrap_err().to_string();
        assert!(!err.is_empty(), "an overflowing word count must be refused");

        // A version-2 file that stops where a version-1 header would have ended: the fields
        // that decide how to read every bit of it are simply not there.
        let path = scratch("short-v2-header.bf");
        std::fs::write(&path, &header(4)[..HEADER_LEN_V1]).unwrap();
        let err = BloomFilter::open(&path).unwrap_err().to_string();
        assert!(err.contains("too short to hold one"), "{err}");
    }

    /// A filter this scanner cannot vouch for is refused by name, before it is opened.
    ///
    /// The bytes are a perfectly good filter -- that is the point. An ecloop `.blf` has
    /// the same magic, the same schedule and would answer every lookup; what it does not
    /// promise is the P2SH script hashes a third of this sweep probes for. Nothing in the
    /// file distinguishes the two, so the name is the whole check, and it has to fire on
    /// a file that would otherwise load cleanly.
    #[test]
    fn a_filter_that_is_not_dot_bf_is_refused_however_valid_its_bytes() {
        let mut bits = vec![0u64; 512];
        let member = [7u8; 20];
        reference_add(&mut bits, &member);

        let refused = scratch("wrong-extension.blf");
        write_filter(&refused, &bits);
        let err = BloomFilter::open(&refused).unwrap_err().to_string();
        assert!(
            err.contains(".bf") && err.contains("keyscan bf-gen"),
            "the refusal has to name the extension and where to get one: {err}"
        );

        // The same bytes under the right name load and answer correctly, so the test is
        // about provenance and not about a file that was broken anyway.
        let accepted = scratch("right-extension.bf");
        write_filter(&accepted, &bits);
        let filter = BloomFilter::open(&accepted).expect("a .bf filter loads");
        assert!(filter.contains(&member), "and reads back what went into it");

        // A path with no extension at all is the other half of `is_none_or`.
        let bare = scratch("no-extension");
        write_filter(&bare, &bits);
        assert!(
            BloomFilter::open(&bare).is_err(),
            "a filter with no extension is refused too"
        );
    }

    /// The real cross-check: agree with `keyscan bf-check` on the operator's own
    /// 7.6 GB filter, for hashes that are in it and hashes that are not.
    ///
    /// Held against `keyscan` rather than against a second implementation of the same
    /// arithmetic, because `keyscan bf-gen` is what wrote the file: a schedule that
    /// agreed with a third party but not with the builder would still read every
    /// filter this scanner is given as empty.
    #[test]
    fn agrees_with_keyscan_bf_check_on_the_real_filter() {
        let (Some(real_filter), Some(keyscan)) =
            (env_path("KEYFORGE_FILTER"), env_path("KEYFORGE_KEYSCAN"))
        else {
            eprintln!("skipping: set KEYFORGE_FILTER and KEYFORGE_KEYSCAN to run this");
            return;
        };
        if !real_filter.exists() || !keyscan.exists() {
            eprintln!("skipping: real filter or keyscan binary not present");
            return;
        }

        let filter = BloomFilter::open(&real_filter).unwrap();

        // A spread of hashes; most will be absent, and any that keyscan reports as
        // present must be present for us too (and vice versa).
        let mut probes: Vec<String> = Vec::new();
        for i in 0u32..64 {
            let mut h = [0u8; 20];
            h[..4].copy_from_slice(&i.to_be_bytes());
            h[4..8].copy_from_slice(&(i * 2654435761).to_be_bytes());
            probes.push(hex::encode(h));
        }
        // One hash the filter certainly holds, so the comparison exercises the full
        // 20-probe path rather than twenty early exits -- which on a blocked filter is also
        // the only case that reads more than the one word every miss stops at.
        //
        // Taken from the list the filter was built from where one is named, so this runs
        // against any `keyscan bf-gen` pair rather than only the operator's own. Otherwise
        // Satoshi's genesis coinbase hash160 (1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa), which is
        // the first entry of the address list the working filter was built from.
        let known_member = env_path("KEYFORGE_FILTER_SOURCE")
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|text| {
                text.lines()
                    .map(str::trim)
                    .find(|l| l.len() == 40 && l.bytes().all(|b| b.is_ascii_hexdigit()))
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "62e907b15cbf27d5425399ebf6f0fb50ebb88f18".to_string());
        probes.push(known_member);

        let out = Command::new(&keyscan)
            .arg("bf-check")
            .arg("-f")
            .arg(&real_filter)
            .args(&probes)
            .output()
            .expect("running keyscan bf-check");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            !stdout.trim().is_empty(),
            "keyscan produced no output: {out:?}"
        );

        let mut compared = 0;
        let mut hits = 0;
        for line in stdout.lines() {
            // `bf-check` prints exactly "<40 hex> FOUND" or "<40 hex> NOT FOUND" per
            // lookup; its opening block goes to stderr, and any line here that is not
            // one of those two shapes is skipped rather than guessed at.
            let Some((hash, verdict)) = line.split_once(' ') else {
                continue;
            };
            let expected = match verdict.trim() {
                "FOUND" => true,
                "NOT FOUND" => false,
                _ => continue,
            };
            let mut bytes = [0u8; 20];
            hex::decode_to_slice(hash, &mut bytes).unwrap();
            assert_eq!(filter.contains(&bytes), expected, "disagreed on {hash}");
            compared += 1;
            hits += expected as usize;
        }

        assert_eq!(compared, probes.len(), "did not compare every probe");
        assert!(hits >= 1, "the genesis hash160 should have been a hit");
    }

    /// No false negatives: every hash160 the filter was built from must be found.
    /// Sampled from the operator's own input list, so this checks the probe schedule
    /// against real data rather than synthetic hashes.
    #[test]
    fn finds_every_hash_the_real_filter_was_built_from() {
        use std::io::{BufRead, BufReader};

        let (Some(real_filter), Some(source)) = (
            env_path("KEYFORGE_FILTER"),
            env_path("KEYFORGE_FILTER_SOURCE"),
        ) else {
            eprintln!("skipping: set KEYFORGE_FILTER and KEYFORGE_FILTER_SOURCE to run this");
            return;
        };
        if !real_filter.exists() || !source.exists() {
            eprintln!("skipping: real filter or source hash list not present");
            return;
        }

        let filter = BloomFilter::open(&real_filter).unwrap();
        let reader = BufReader::new(File::open(&source).unwrap());
        let mut checked = 0;
        for line in reader.lines().take(200_000) {
            let line = line.unwrap();
            let line = line.trim();
            if line.len() != 40 {
                continue;
            }
            let mut bytes = [0u8; 20];
            hex::decode_to_slice(line, &mut bytes).unwrap();
            assert!(filter.contains(&bytes), "false negative on {line}");
            checked += 1;
        }
        assert!(
            checked > 100_000,
            "expected to check many hashes, got {checked}"
        );
    }
}
