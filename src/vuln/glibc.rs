//! Wallets built on C's `random()` / `rand()`, seeded with something small.
//!
//! `srandom(time(NULL))` in a wallet generator is one of the oldest mistakes available,
//! and it leaves a search space the size of however many seconds the program could have
//! been run in. The generator is not a Mersenne twister and not an LCG: glibc's default
//! is a **TYPE_3 additive-feedback** generator over a 31-word state.
//!
//! # The algorithm
//!
//! The state is seeded by a Lehmer LCG, `r[i] = 16807 * r[i-1] mod 2147483647`, computed
//! with the Schrage trick so it fits in 32 bits. The generator then runs
//! `r[i] = r[i-3] + r[i-31]` (mod 2^32) and returns the top 31 bits of each result. The
//! first 310 outputs are discarded, which is what mixes the LCG's very regular early
//! state; skipping that warm-up is the classic way to get a nearly-correct
//! implementation that produces entirely wrong numbers.
//!
//! # Which bytes a generator would have taken
//!
//! `random()` returns 31 bits, and the overwhelmingly common idiom is to keep the low
//! byte -- `buf[i] = random() & 0xff`, or the `% 256` that compiles to the same thing.
//! That is the mapping scanned here. It is an assumption about the affected program
//! rather than a property of the generator, and the guide says so.
//!
//! On glibc, `rand()` is the same generator as `random()`, so a program using either is
//! covered. Note that this is *not* true of every libc: macOS's `rand()` is a different
//! LCG, though its `random()` is bit-for-bit glibc's -- which is what makes the C oracle
//! in the tests below possible on a Mac at all.

use crate::scan::derive::Route;
use crate::vuln::{Defaults, Expanded, Guide, KernelSpec, Point, Space, Vulnerability};

/// The device half of [`GlibcRand::expand`].
const KERNEL_SOURCE: &str = include_str!("../../kernels/vuln/glibc.h");

/// Words of state in the default TYPE_3 generator.
const DEG: usize = 31;
/// The feedback tap.
const SEP: usize = 3;
/// Outputs discarded before the generator is considered warmed up.
const WARMUP: usize = 310;

/// glibc's `random()`, seeded with `srandom(seed)`.
pub struct GlibcRandom {
    state: [u32; DEG],
    f: usize,
    r: usize,
}

impl GlibcRandom {
    pub fn new(seed: u32) -> Self {
        // glibc maps seed 0 to 1: an all-zero state would stay all-zero forever.
        let seed = if seed == 0 { 1 } else { seed };

        let mut state = [0u32; DEG];
        state[0] = seed;
        for i in 1..DEG {
            // r[i] = 16807 * r[i-1] % 2147483647, via Schrage's trick so the
            // intermediate product never leaves 32 bits. The signed correction is
            // glibc's, kept verbatim -- computing this in i64 and reducing gives the
            // same answer here but diverges from glibc for some seeds.
            let prev = state[i - 1] as i32 as i64;
            let hi = prev / 127_773;
            let lo = prev % 127_773;
            let mut word = 16_807 * lo - 2_836 * hi;
            if word < 0 {
                word += 2_147_483_647;
            }
            state[i] = word as u32;
        }

        let mut rng = Self { state, f: SEP, r: 0 };
        // The warm-up is not optional: without it the first outputs still carry the
        // LCG's structure and every derived wallet is wrong.
        for _ in 0..WARMUP {
            rng.next_u31();
        }
        rng
    }

    /// One 31-bit output.
    #[inline]
    pub fn next_u31(&mut self) -> u32 {
        self.state[self.f] = self.state[self.f].wrapping_add(self.state[self.r]);
        let result = self.state[self.f] >> 1;
        self.f = (self.f + 1) % DEG;
        self.r = (self.r + 1) % DEG;
        result
    }

    /// The low byte, which is what `random() & 0xff` and `random() % 256` both take.
    #[inline]
    pub fn next_byte(&mut self) -> u8 {
        self.next_u31() as u8
    }
}

/// A wallet generator seeded with `srandom()` / `srand()` over a 32-bit value.
pub struct GlibcRand;

impl Vulnerability for GlibcRand {
    fn id(&self) -> &'static str {
        "glibc-rand"
    }

    fn classification(&self) -> &'static str {
        "no CVE -- a generator class, not a product"
    }
    fn aliases(&self) -> &'static [&'static str] {
        &["glibc", "srandom"]
    }

    fn describe(&self) -> Vec<String> {
        vec![
            "C's random()/rand() on glibc, a TYPE_3 additive-feedback generator seeded \
             with srandom(). Typically seeded from time(NULL), so if you know roughly \
             when the wallet was made the range narrows to those seconds. The byte \
             mapping scanned is the low byte of each output, which is what `& 0xff` and \
             `% 256` both produce."
                .into(),
        ]
    }

    fn guide(&self) -> Guide {
        Guide {
            what: "A C program that called `srandom(time(NULL))` and built a wallet from \
                   `random()` tied the key to the second it ran. Anyone who knows roughly \
                   when the wallet was created only has to try the seconds in that window, \
                   and the whole range is only 4.3 billion in any case.",
            affected: "Any C or C++ wallet tool using rand() or random() instead of a \
                       cryptographic source. No CVE -- this is the textbook mistake \
                       rather than one shipped bug. Note glibc specifically: some other \
                       C libraries use a different rand().",
            command: "keyforge scan --vuln glibc-rand -f funded.bf",
            time: "About 9 days on a laptop, or 9 hours on a recent NVIDIA card, for the \
                   whole range. If the seed was a clock reading and you know the year, \
                   narrowing --start and --end to that window cuts it to minutes.",
            hit: "A mnemonic phrase or a 64-character private key per line in \
                  matches.txt. This scan assumes the program kept the low byte of each \
                  output, so a clean sweep does not rule the program out.",
        }
    }

    /// Starts at 1, not 0.
    ///
    /// glibc's `srandom` maps seed 0 to seed 1 -- an all-zero state would stay all-zero
    /// forever -- so scanning both would walk the same stream twice and report nothing
    /// new for the second pass. A program that called `srandom(0)` is still covered:
    /// its wallet is the one found at seed 1.
    fn space(&self) -> Space {
        Space::Integers { start: 1, end: 1 << 32 }
    }

    /// A C wallet generator is as likely to have used the bytes as a key as to have fed
    /// them to BIP39, so nothing is narrowed.
    fn defaults(&self) -> Defaults {
        Defaults {
            material_sizes: vec![16, 32],
            routes: vec![Route::Bip39, Route::Bip32Seed, Route::PrivKey],
            paths: Vec::new(),
        }
    }

    /// Seeds usually came from `time(NULL)`, so a known creation window is a real saving.
    fn range_is_narrowable(&self) -> bool {
        true
    }

    fn expand(&self, point: Point<'_>, out: &mut Vec<Expanded>) {
        let Point::Integer(n) = point else {
            debug_assert!(false, "glibc-rand does not read a corpus");
            return;
        };
        let mut rng = GlibcRandom::new(n as u32);
        let mut material = [0u8; 32];
        for byte in &mut material {
            *byte = rng.next_byte();
        }
        out.push(material);
    }

    /// One stream: the low byte of each output, which is the mapping this plugin models.
    fn kernel(&self) -> Option<KernelSpec> {
        Some(KernelSpec { source: KERNEL_SOURCE, streams: vec![0], defines: Vec::new() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sequence every glibc `srandom(1)` produces, and the one value from this
    /// generator that is quotable from memory rather than from a machine.
    #[test]
    fn matches_the_well_known_srandom_1_sequence() {
        let mut rng = GlibcRandom::new(1);
        let got: Vec<u32> = (0..6).map(|_| rng.next_u31()).collect();
        assert_eq!(
            got,
            vec![1804289383, 846930886, 1681692777, 1714636915, 1957747793, 424238335]
        );
    }

    /// Byte streams printed by a C program compiled against the system libc:
    ///
    /// ```c
    /// srandom(seed);
    /// for (int i = 0; i < 32; i++) printf("%02x", (unsigned char)(random() & 0xff));
    /// ```
    ///
    /// A real second implementation in another language, which is the only thing that
    /// proves the 310-output warm-up and the Schrage seeding are right -- both produce
    /// plausible-looking wrong numbers when omitted.
    ///
    /// **Seed 0 is deliberately not in this list.** It is the one input where libcs
    /// disagree: glibc maps it to seed 1, and macOS -- whose `random()` is otherwise
    /// bit-for-bit glibc's, which is what makes this oracle possible on a Mac -- does
    /// something else with it. This models glibc, so seed 0 is handled by
    /// `seed_zero_is_seed_one` and excluded from the space; see `GlibcRand::space`.
    #[test]
    fn matches_a_compiled_c_implementation() {
        let vectors = [
            (1u32, "67c6697351ff4aec29cdbaabf2fbe3467cc254f81be8e78d765a2e63339fc99a"),
            (2, "fa7f444fd5d2002d294b96c34dc57d297ed55fda3214d99bd79f7a0ef8972df2"),
            (500, "9e19e9d667e6575a690eddc0ff34603abceaefa3d9217a7398bf7bbd67b09f06"),
            (1234567890, "be99f0d55687b85ba2c0db94acff21d4d12cc1cb25ec1d24658b791f636acc22"),
            (4294967295, "3bcc08e1e4aee6fb5028c0e936efed6df2b1df8ef9946db9e65d554d9f5256da"),
        ];
        for (seed, want) in vectors {
            let mut out = Vec::new();
            GlibcRand.expand(Point::Integer(seed as u128), &mut out);
            assert_eq!(hex::encode(out[0]), want, "srandom({seed}) byte stream");
        }
    }

    /// glibc maps seed 0 to seed 1, because an all-zero state never leaves itself.
    ///
    /// This is also why the space starts at 1: scanning both would walk one stream
    /// twice. Note that this is glibc's behaviour specifically -- macOS's `srandom(0)`
    /// produces a third thing, which is why seed 0 is absent from the C oracle above.
    #[test]
    fn seed_zero_is_seed_one() {
        let mut zero = GlibcRandom::new(0);
        let mut one = GlibcRandom::new(1);
        for _ in 0..8 {
            assert_eq!(zero.next_u31(), one.next_u31());
        }
    }

    /// Skipping the warm-up is the classic near-miss: the generator still produces
    /// numbers, they are just the wrong ones. This pins that the warm-up happened.
    #[test]
    fn the_warmup_actually_runs() {
        // Without the 310 discarded outputs the first value would be the raw
        // `state[3] + state[0]` of the seeded LCG, which is this:
        let mut cold = GlibcRandom { state: [0; DEG], f: SEP, r: 0 };
        cold.state[0] = 1;
        for i in 1..DEG {
            let prev = cold.state[i - 1] as i32 as i64;
            let (hi, lo) = (prev / 127_773, prev % 127_773);
            let mut w = 16_807 * lo - 2_836 * hi;
            if w < 0 {
                w += 2_147_483_647;
            }
            cold.state[i] = w as u32;
        }
        let unwarmed = cold.next_u31();
        assert_ne!(
            unwarmed,
            GlibcRandom::new(1).next_u31(),
            "the warm-up is being skipped; every derived wallet would be wrong"
        );
    }

    /// Distinct seeds must give distinct wallets.
    #[test]
    fn distinct_seeds_give_distinct_material() {
        let mut seen = std::collections::HashSet::new();
        for seed in 1..64u128 {
            let mut out = Vec::new();
            GlibcRand.expand(Point::Integer(seed), &mut out);
            assert!(seen.insert(out[0]), "seed {seed} repeated an earlier wallet");
        }
    }
}
