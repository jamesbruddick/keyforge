//! Independent SHA-512 streams compressed in lockstep.
//!
//! PBKDF2 is serial within a stream: iteration 2048 cannot start until 2047 has
//! finished. That leaves the hardware badly underused, because the work of a round has
//! several cycles of latency and the unit can accept new work every cycle. The scanner
//! always has independent streams to fill the gap with -- `crate::scan::derive` walks a batch
//! of seeds together and each seed contributes one PBKDF2 job per entropy size, which is
//! twelve streams at the default scope.
//!
//! **How they are filled depends entirely on what the target can do**, and the two cases
//! are not variations on each other:
//!
//!   * **A core with SHA-512 instructions** (ARMv8.2 `sha3`) needs only enough streams to
//!     hide the round latency. Two interleaved measured 9.7 million compressions a second
//!     against one stream's 5.4 on an M1 Pro -- 1.8x for no extra work. A third adds
//!     nothing; the unit is saturated. That is `compress_x2_sha3`.
//!   * **A core without them** has no SHA-512 unit to saturate and must do the rounds in
//!     general-purpose arithmetic. There the win is not latency hiding but *width*: the
//!     round is a handful of adds, xors, rotates and and-nots on 64-bit words, and a
//!     vector unit will do four or eight lanes of each in one instruction. That is
//!     `compress_wide`, and it is why this module is generic over the lane count.
//!
//! x86-64 is the second case and stays there: Zen 4 and Ice Lake have SHA-NI for SHA-256
//! but no SHA-512 instructions at all, so every round is scalar integer work. Before this
//! module was widened, the two-lane path on x86 was two sequential calls to `sha2`'s
//! one-lane compression -- a scheduling idea that did nothing, because two calls a
//! thousand instructions long do not overlap in any reorder window.
//!
//! **Which path runs is decided at run time, not at build time.** An ordinary portable
//! `cargo build --release` gets the vector path on any CPU that has the vector unit for
//! it, and the same binary still runs on one that does not -- there is no build flag to
//! remember and no binary that dies with an illegal instruction on the wrong machine.
//! `#[target_feature]` is what makes that possible: the wrappers around `compress_wide`
//! are each compiled with their extensions enabled regardless of the build's baseline,
//! and `wide_support` picks between them once. It is the same arrangement the aarch64
//! SHA-512 check has always used here.
//!
//! Nothing here is `unsafe` except those wrappers and the aarch64 intrinsics that were
//! always here, and no path changes results: `wide_lanes_agree_with_one` compares every
//! width against `sha2`'s own compression, which is the whole specification.

use sha2::block_api::compress512;

/// SHA-512 round constants (FIPS 180-4).
static K64: [u64; 80] = [
    0x428a2f98d728ae22,
    0x7137449123ef65cd,
    0xb5c0fbcfec4d3b2f,
    0xe9b5dba58189dbbc,
    0x3956c25bf348b538,
    0x59f111f1b605d019,
    0x923f82a4af194f9b,
    0xab1c5ed5da6d8118,
    0xd807aa98a3030242,
    0x12835b0145706fbe,
    0x243185be4ee4b28c,
    0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f,
    0x80deb1fe3b1696b1,
    0x9bdc06a725c71235,
    0xc19bf174cf692694,
    0xe49b69c19ef14ad2,
    0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5,
    0x240ca1cc77ac9c65,
    0x2de92c6f592b0275,
    0x4a7484aa6ea6e483,
    0x5cb0a9dcbd41fbd4,
    0x76f988da831153b5,
    0x983e5152ee66dfab,
    0xa831c66d2db43210,
    0xb00327c898fb213f,
    0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2,
    0xd5a79147930aa725,
    0x06ca6351e003826f,
    0x142929670a0e6e70,
    0x27b70a8546d22ffc,
    0x2e1b21385c26c926,
    0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df,
    0x650a73548baf63de,
    0x766a0abb3c77b2a8,
    0x81c2c92e47edaee6,
    0x92722c851482353b,
    0xa2bfe8a14cf10364,
    0xa81a664bbc423001,
    0xc24b8b70d0f89791,
    0xc76c51a30654be30,
    0xd192e819d6ef5218,
    0xd69906245565a910,
    0xf40e35855771202a,
    0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8,
    0x1e376c085141ab53,
    0x2748774cdf8eeb99,
    0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63,
    0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373,
    0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc,
    0x78a5636f43172f60,
    0x84c87814a1f0ab72,
    0x8cc702081a6439ec,
    0x90befffa23631e28,
    0xa4506cebde82bde9,
    0xbef9a3f7b2c67915,
    0xc67178f2e372532b,
    0xca273eceea26619c,
    0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e,
    0xf57d4f7fee6ed178,
    0x06f067aa72176fba,
    0x0a637dc5a2c898a6,
    0x113f9804bef90dae,
    0x1b710b35131c471b,
    0x28db77f523047d84,
    0x32caab7b40c72493,
    0x3c9ebe0a15c9bebc,
    0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6,
    0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec,
    0x6c44198c4a475817,
];

/// How many streams are compressed at once.
///
/// A scheduling choice, not a capability claim: `compress` decides at run time what to do
/// with a group, and a machine with no vector unit worth using simply compresses the eight
/// one at a time exactly as it did before. So this is fixed per architecture rather than
/// read off the build's target features, and **no build flag is needed to get the fast
/// path** -- see `Wide`.
///
/// Two on aarch64 because a core with SHA-512 instructions wants exactly enough streams to
/// hide the round latency and no more; eight on x86-64 because that is the one width the
/// vectoriser handles. Four is the obvious width for AVX2 -- a `[u64; 4]` is exactly one
/// ymm register -- and LLVM does not vectorise it at all. Compiled for `znver4`,
/// `compress_wide::<4>` comes out as 13,423 scalar instructions with 3,186 stack
/// references and not one vector register, which is far worse than the `sha2` call it
/// would replace. The same code at eight lanes vectorises cleanly on both:
///
/// ```text
///   target feature   width   insns   per block   vector regs
///   avx512f              8   3,246         405   3,014 zmm, 652 vprolq
///   avx2                 8  12,731       1,591   6,564 ymm
///   avx512f              4  13,423       3,355   none -- scalar
/// ```
///
/// So the useful widths are eight and one, and there is no four in between. Check the
/// table above with:
///
/// ```sh
/// rustc --target x86_64-unknown-linux-gnu -C opt-level=3 -C target-cpu=znver4 \
///     --emit=asm src/sha512.rs
/// ```
pub const LANES: usize = if cfg!(target_arch = "x86_64") { 8 } else { 2 };

/// Which vector width this CPU actually has, resolved once.
///
/// **Detected at run time rather than compiled in**, for the same reason the aarch64
/// SHA-512 check beside it is: a binary should be fast on the machine it finds itself on
/// without having been told about it at build time, and `-C target-cpu=native` produces a
/// binary that dies with an illegal instruction anywhere else. The two `compress_wide`
/// wrappers below carry `#[target_feature]`, so each is *compiled* with its vector
/// extensions enabled whatever the build's baseline, and only the one this CPU supports is
/// ever called.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Wide {
    Avx512,
    Avx2,
    /// No vector unit worth the transposition; `sha2`'s one-lane compression wins.
    None,
}

#[cfg(target_arch = "x86_64")]
fn wide_support() -> Wide {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHED: AtomicU8 = AtomicU8::new(u8::MAX);

    let cached = CACHED.load(Ordering::Relaxed);
    if cached != u8::MAX {
        return match cached {
            0 => Wide::Avx512,
            1 => Wide::Avx2,
            _ => Wide::None,
        };
    }
    // AVX-512F is what carries `vprolq`, the single-instruction rotate that makes the
    // 512-bit form worth more than twice the 256-bit one rather than exactly twice.
    let detected = if std::arch::is_x86_feature_detected!("avx512f") {
        Wide::Avx512
    } else if std::arch::is_x86_feature_detected!("avx2") {
        Wide::Avx2
    } else {
        Wide::None
    };
    CACHED.store(detected as u8, Ordering::Relaxed);
    detected
}

/// `compress_wide` compiled for AVX-512.
///
/// # Safety
/// The caller must have established that this CPU has `avx512f`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn compress_wide_avx512<const N: usize>(
    states: &mut [[u64; 8]; N],
    blocks: &[[u8; 128]; N],
) {
    compress_wide(states, blocks);
}

/// `compress_wide` compiled for AVX2.
///
/// # Safety
/// The caller must have established that this CPU has `avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn compress_wide_avx2<const N: usize>(states: &mut [[u64; 8]; N], blocks: &[[u8; 128]; N]) {
    compress_wide(states, blocks);
}

/// Whether a lockstep group costs the same however many of its lanes carry real work.
///
/// True when `compress` will dispatch to a path that does all `N` lanes in one pass -- the
/// aarch64 intrinsics or either vector width -- and false when it will fall through to `N`
/// sequential `sha2` calls, where an unused lane is simply a wasted compression chain.
///
/// `crate::crypto::pbkdf2` pads a short final group, which is free in the first case and is
/// straightforwardly more work in the second: twelve streams padded to sixteen is a third
/// more PBKDF2 than doing the last four singly.
pub fn padding_is_free() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        has_sha512_instructions()
    }
    #[cfg(target_arch = "x86_64")]
    {
        wide_support() != Wide::None
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        false
    }
}

/// Compress one block into each of `N` independent states.
///
/// `N` is a const generic rather than `LANES` because the caller cannot always fill a
/// full group: at a narrowed scope a batch may have four streams where `LANES` is eight.
/// `crate::crypto::pbkdf2` steps the width down rather than dropping to one at a time.
pub fn compress<const N: usize>(states: &mut [[u64; 8]; N], blocks: &[[u8; 128]; N]) {
    // A core with SHA-512 instructions wants pairs and nothing wider; see the module
    // comment. Whole pairs go through the intrinsics and an odd one out goes scalar.
    #[cfg(target_arch = "aarch64")]
    if has_sha512_instructions() {
        let mut state_pairs = states.chunks_exact_mut(2);
        let mut block_pairs = blocks.chunks_exact(2);
        for (s, b) in (&mut state_pairs).zip(&mut block_pairs) {
            // SAFETY: guarded by the runtime feature check the function name describes.
            unsafe { compress_x2_sha3(s.try_into().unwrap(), b.try_into().unwrap()) };
        }
        for (state, block) in state_pairs.into_remainder().iter_mut().zip(block_pairs.remainder())
        {
            compress512(state, std::slice::from_ref(block));
        }
        return;
    }

    // Without them the rounds are general-purpose arithmetic and width is the whole point
    // -- but only at the one width that vectorises. Any other width falls through to
    // `sha2`'s tuned one-lane compression, which is never worse than hand-written scalar
    // rounds. See `LANES`.
    #[cfg(target_arch = "x86_64")]
    if N == LANES {
        // SAFETY: each arm calls the wrapper whose feature `wide_support` just confirmed.
        match wide_support() {
            Wide::Avx512 => {
                unsafe { compress_wide_avx512(states, blocks) };
                return;
            }
            Wide::Avx2 => {
                unsafe { compress_wide_avx2(states, blocks) };
                return;
            }
            Wide::None => {}
        }
    }

    for (state, block) in states.iter_mut().zip(blocks) {
        compress512(state, std::slice::from_ref(block));
    }
}

/// `N` streams of rounds, written so that every operation is the same operation on `N`
/// adjacent `u64`s.
///
/// That shape is the entire trick. The state and the message schedule are transposed to
/// word-major on entry, so `a[lane]` for all lanes is contiguous and a round becomes a
/// dozen elementwise array operations -- which is what the vectoriser needs to see to
/// turn each of them into one instruction. Nothing here is an intrinsic: the code is
/// portable, correct on any target, and simply faster on one with vector registers.
///
/// The message schedule is a **rolling sixteen words** rather than the textbook eighty.
/// Eighty words at `N` lanes is 5 KB of live state at eight lanes, which no register file
/// holds; sixteen is a quarter of the recurrence's reach and all it ever looks back
/// through. The rounds are taken in groups of sixteen so that `(base + j) & 15` is just
/// `j` and every subscript is a compile-time constant -- a dynamically indexed local
/// array cannot stay in registers. Same shape, and the same reason, as the ring buffer in
/// kernels/sha512.h.
///
/// **`#[inline(always)]` is load-bearing.** This is the body both `#[target_feature]`
/// wrappers above call, and a function is codegen'd with the features of the function it
/// is inlined into. Left out of line it would be compiled once at the build's baseline --
/// scalar -- and the wrappers would call into it having enabled nothing.
///
/// It stays compiled on every target, not just the one that calls it: it is portable code
/// and `wide_lanes_agree_with_one` checks it wherever the tests run, which is the only
/// reason its correctness is knowable on a machine that would never dispatch to it.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[inline(always)]
fn compress_wide<const N: usize>(states: &mut [[u64; 8]; N], blocks: &[[u8; 128]; N]) {
    // Word-major: `v[word][lane]`.
    let mut w = [[0u64; N]; 16];
    for (i, word) in w.iter_mut().enumerate() {
        for (lane, block) in word.iter_mut().zip(blocks) {
            *lane = u64::from_be_bytes(block[i * 8..i * 8 + 8].try_into().unwrap());
        }
    }

    let mut s = [[0u64; N]; 8];
    for (i, word) in s.iter_mut().enumerate() {
        for (lane, state) in word.iter_mut().zip(states.iter()) {
            *lane = state[i];
        }
    }
    let original = s;

    // One round, over every lane. The state rotation is by assignment, exactly as in the
    // one-lane reference -- `s` is eight vectors and rotating their roles is free.
    macro_rules! round {
        ($k:expr, $w:expr) => {{
            let [a, b, c, d, e, f, g, h] = &mut s;
            for l in 0..N {
                let s1 = e[l].rotate_right(14) ^ e[l].rotate_right(18) ^ e[l].rotate_right(41);
                let ch = (e[l] & f[l]) ^ (!e[l] & g[l]);
                let t1 = h[l]
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add($k)
                    .wrapping_add($w[l]);
                let s0 = a[l].rotate_right(28) ^ a[l].rotate_right(34) ^ a[l].rotate_right(39);
                let maj = (a[l] & b[l]) ^ (a[l] & c[l]) ^ (b[l] & c[l]);
                let t2 = s0.wrapping_add(maj);
                h[l] = g[l];
                g[l] = f[l];
                f[l] = e[l];
                e[l] = d[l].wrapping_add(t1);
                d[l] = c[l];
                c[l] = b[l];
                b[l] = a[l];
                a[l] = t1.wrapping_add(t2);
            }
        }};
    }

    // Extend the schedule by one word and take the round that consumes it. `$j` is always
    // a literal, which is what keeps the four subscripts constant.
    macro_rules! schedule_and_round {
        ($j:expr, $base:expr) => {{
            let x = w[($j + 1) & 15];
            let y = w[($j + 14) & 15];
            let back = w[($j + 9) & 15];
            let word = &mut w[$j];
            for l in 0..N {
                let s0 = x[l].rotate_right(1) ^ x[l].rotate_right(8) ^ (x[l] >> 7);
                let s1 = y[l].rotate_right(19) ^ y[l].rotate_right(61) ^ (y[l] >> 6);
                word[l] = word[l]
                    .wrapping_add(s0)
                    .wrapping_add(back[l])
                    .wrapping_add(s1);
            }
            let word = w[$j];
            round!(K64[$base + $j], word);
        }};
    }

    macro_rules! sixteen_rounds {
        ($base:expr) => {{
            schedule_and_round!(0, $base);
            schedule_and_round!(1, $base);
            schedule_and_round!(2, $base);
            schedule_and_round!(3, $base);
            schedule_and_round!(4, $base);
            schedule_and_round!(5, $base);
            schedule_and_round!(6, $base);
            schedule_and_round!(7, $base);
            schedule_and_round!(8, $base);
            schedule_and_round!(9, $base);
            schedule_and_round!(10, $base);
            schedule_and_round!(11, $base);
            schedule_and_round!(12, $base);
            schedule_and_round!(13, $base);
            schedule_and_round!(14, $base);
            schedule_and_round!(15, $base);
        }};
    }

    // The first sixteen rounds read the message as it stands; every group after re-derives
    // the sixteen words in place.
    for j in 0..16 {
        let word = w[j];
        round!(K64[j], word);
    }
    sixteen_rounds!(16);
    sixteen_rounds!(32);
    sixteen_rounds!(48);
    sixteen_rounds!(64);

    for (state, i) in states.iter_mut().zip(0..) {
        for word in 0..8 {
            state[word] = original[word][i].wrapping_add(s[word][i]);
        }
    }
}


/// Whether this CPU has the ARMv8.2 SHA-512 instructions.
///
/// Detection is not free enough to repeat 2048 times per PBKDF2 stream, so the answer
/// is resolved once and read from a static afterwards.
#[cfg(target_arch = "aarch64")]
#[inline]
fn has_sha512_instructions() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHED: AtomicU8 = AtomicU8::new(u8::MAX);

    let cached = CACHED.load(Ordering::Relaxed);
    if cached != u8::MAX {
        return cached == 1;
    }
    let detected = std::arch::is_aarch64_feature_detected!("sha3");
    CACHED.store(detected as u8, Ordering::Relaxed);
    detected
}

/// The two-lane compression itself.
///
/// The round sequence, the message schedule and the state rotation are all exactly
/// the one-lane version; the only change is that every value became a two-element
/// array and every step runs over both. Writing it this way rather than as two
/// hand-unrolled copies keeps it checkable against the original line by line, and the
/// `lane` loops are unrolled at compile time.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sha3")]
unsafe fn compress_x2_sha3(states: &mut [[u64; 8]; 2], blocks: &[[u8; 128]; 2]) {
    use std::arch::aarch64::*;

    // SAFETY: the caller has established that this CPU has the `sha3` feature, which
    // is what every instruction below requires. All loads and stores are at fixed
    // offsets inside the fixed-size arrays the signature guarantees.
    unsafe {
        // Working state, as four vectors of two words: [ab, cd, ef, gh].
        let mut v = [[vdupq_n_u64(0); 4]; 2];
        // The message schedule, as eight vectors of two words.
        let mut w = [[vdupq_n_u64(0); 8]; 2];

        for lane in 0..2 {
            for i in 0..4 {
                v[lane][i] = vld1q_u64(states[lane][i * 2..].as_ptr());
            }
            for i in 0..8 {
                // The block is big-endian; the vector unit wants it byte-reversed.
                w[lane][i] =
                    vreinterpretq_u64_u8(vrev64q_u8(vld1q_u8(blocks[lane][i * 16..].as_ptr())));
            }
        }
        let original = v;

        for t in (0..80).step_by(16) {
            for group in 0..8 {
                // Which of ab/cd/ef/gh plays which role rotates every two rounds.
                let (p, q, r, s) = (
                    (8 - group) % 4,
                    (9 - group) % 4,
                    (10 - group) % 4,
                    (11 - group) % 4,
                );
                for lane in 0..2 {
                    let w = &mut w[lane];
                    if t > 0 {
                        w[group] = vsha512su1q_u64(
                            vsha512su0q_u64(w[group], w[(group + 1) % 8]),
                            w[(group + 7) % 8],
                            vextq_u64::<1>(w[(group + 4) % 8], w[(group + 5) % 8]),
                        );
                    }
                    let v = &mut v[lane];
                    let initial = vaddq_u64(w[group], vld1q_u64(K64.as_ptr().add(t + group * 2)));
                    let sum = vaddq_u64(vextq_u64::<1>(initial, initial), v[s]);
                    let intermed =
                        vsha512hq_u64(sum, vextq_u64::<1>(v[r], v[s]), vextq_u64::<1>(v[q], v[r]));
                    v[s] = vsha512h2q_u64(intermed, v[q], v[p]);
                    v[q] = vaddq_u64(v[q], intermed);
                }
            }
        }

        for lane in 0..2 {
            for i in 0..4 {
                let sum = vaddq_u64(v[lane][i], original[lane][i]);
                vst1q_u64(states[lane][i * 2..].as_mut_ptr(), sum);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    /// Every width must produce exactly what one lane produces, for any state and any
    /// block. This is the only thing this module has to get right: it is a scheduling
    /// change, so `sha2`'s own compression is the whole specification.
    ///
    /// **`compress_wide` is checked directly rather than through `compress`**, which is
    /// what makes this test worth anything on a machine that would never dispatch to it.
    /// The wide path is portable code, so it runs and must be correct everywhere -- and
    /// the target it exists for is not the target most of this is developed on. Going
    /// through `compress` would silently check the aarch64 intrinsics twice and the code
    /// under test not at all.
    ///
    /// Random states rather than only the SHA-512 IV, because the PBKDF2 caller feeds
    /// it HMAC pad midstates and never starts from the IV at all.
    #[test]
    fn wide_lanes_agree_with_one() {
        fn check<const N: usize>(rng: &mut impl RngExt) {
            for _ in 0..500 {
                let states: [[u64; 8]; N] = std::array::from_fn(|_| rng.random());
                let blocks: [[u8; 128]; N] =
                    std::array::from_fn(|_| std::array::from_fn(|_| rng.random()));

                let mut expected = states;
                for (state, block) in expected.iter_mut().zip(&blocks) {
                    compress512(state, std::slice::from_ref(block));
                }

                let mut got = states;
                compress_wide(&mut got, &blocks);
                assert_eq!(got, expected, "{N} lanes disagreed with one");
            }
        }

        let mut rng = rand::rng();
        check::<1>(&mut rng);
        check::<2>(&mut rng);
        check::<3>(&mut rng);
        check::<4>(&mut rng);
        check::<8>(&mut rng);
    }

    /// The dispatcher must agree with one lane too, at every width and whatever path it
    /// picks on this machine -- including the group sizes that are not a multiple of the
    /// two the aarch64 intrinsics work in.
    #[test]
    fn every_dispatched_width_agrees_with_one() {
        fn check<const N: usize>(rng: &mut impl RngExt) {
            for _ in 0..500 {
                let states: [[u64; 8]; N] = std::array::from_fn(|_| rng.random());
                let blocks: [[u8; 128]; N] =
                    std::array::from_fn(|_| std::array::from_fn(|_| rng.random()));

                let mut expected = states;
                for (state, block) in expected.iter_mut().zip(&blocks) {
                    compress512(state, std::slice::from_ref(block));
                }

                let mut got = states;
                compress(&mut got, &blocks);
                assert_eq!(got, expected, "{N} dispatched lanes disagreed with one");
            }
        }

        let mut rng = rand::rng();
        check::<1>(&mut rng);
        check::<2>(&mut rng);
        check::<3>(&mut rng);
        check::<4>(&mut rng);
        check::<8>(&mut rng);
    }

    /// The lanes must not leak into each other: compressing the same block in both
    /// lanes of a pair has to give the same answer as compressing it alone.
    #[test]
    fn the_lanes_are_independent() {
        let mut rng = rand::rng();
        let state: [u64; 8] = rng.random();
        let block: [u8; 128] = std::array::from_fn(|_| rng.random());
        let other: [u8; 128] = std::array::from_fn(|_| rng.random());

        let mut alone = [state, state];
        compress(&mut alone, &[block, block]);

        let mut mixed = [state, state];
        compress(&mut mixed, &[block, other]);

        assert_eq!(alone[0], mixed[0], "lane 0 changed with lane 1's input");
    }
}
