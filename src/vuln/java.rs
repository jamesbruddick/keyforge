//! Wallets built on `java.util.Random`.
//!
//! `java.util.Random` is a 48-bit linear congruential generator, documented in full in
//! its own Javadoc, and it is not a cryptographic source. Its state is small enough to
//! enumerate outright in some cases and trivially small once you know roughly when the
//! wallet was made, because the overwhelmingly common seeding is
//! `new Random(System.currentTimeMillis())`.
//!
//! # The seed space is 2^48, and that is too big
//!
//! Unlike everything else in this registry, a full sweep here is not feasible: 2^48 is
//! about 281 trillion points. That is not a reason to leave the vulnerability out, it is
//! a reason to say so loudly. A millisecond timestamp narrows extremely well -- a single
//! known day is 86.4 million points, which finishes in under a minute -- so this is a
//! vulnerability you scan with `--start` and `--end` set, and the guide says exactly
//! that. `range_is_narrowable` is true for the same reason.
//!
//! # The oracle for this module is weaker than the others
//!
//! There is no JVM on the machine this was written on, so the test vectors below come
//! from an independent implementation of the published Javadoc algorithm rather than
//! from Java itself. Two implementations from one written specification catch typos and
//! transcription errors but not a shared misreading of the spec, which is a real gap.
//! It is narrowed by `new Random(0).nextInt() == -1155484576` -- a value published in
//! countless places and quotable independently of any implementation -- and it should be
//! closed properly by running `agrees_with_a_real_jvm` on a machine that has one.

use crate::scan::derive::Route;
use crate::vuln::{Defaults, Expanded, Guide, KernelSpec, Point, Space, Vulnerability};

/// The device half of [`JavaUtilRandom::expand`].
const KERNEL_SOURCE: &str = include_str!("../../kernels/vuln/java.h");

const MULTIPLIER: u64 = 0x5DEECE66D;
const ADDEND: u64 = 0xB;
const MASK: u64 = (1 << 48) - 1;

/// `java.util.Random`, seeded as the constructor does.
pub struct JavaRandom {
    seed: u64,
}

impl JavaRandom {
    /// `new Random(seed)`. The constructor scrambles its argument -- passing the raw
    /// value straight into the state is the single most common way to get this wrong.
    pub fn new(seed: u64) -> Self {
        Self { seed: (seed ^ MULTIPLIER) & MASK }
    }

    /// `protected int next(int bits)`.
    #[inline]
    pub fn next(&mut self, bits: u32) -> u32 {
        self.seed = self.seed.wrapping_mul(MULTIPLIER).wrapping_add(ADDEND) & MASK;
        (self.seed >> (48 - bits)) as u32
    }

    /// `nextInt()`, as the unsigned bit pattern. Java prints this signed.
    #[inline]
    pub fn next_int(&mut self) -> u32 {
        self.next(32)
    }

    /// `nextBytes(byte[])`.
    ///
    /// Each `nextInt` fills four bytes **low byte first**, and a short tail takes only
    /// as many as it needs from the last word. Getting the byte order backwards here
    /// produces a stream that looks just as random and is entirely wrong.
    pub fn next_bytes(&mut self, out: &mut [u8]) {
        let mut i = 0;
        while i < out.len() {
            let mut rnd = self.next_int();
            for _ in 0..4.min(out.len() - i) {
                out[i] = rnd as u8;
                rnd >>= 8;
                i += 1;
            }
        }
    }
}

/// A wallet generator seeded with `new Random(...)`.
pub struct JavaUtilRandom;

impl Vulnerability for JavaUtilRandom {
    fn id(&self) -> &'static str {
        "java-random"
    }

    fn classification(&self) -> &'static str {
        "no CVE -- a generator class, not a product"
    }
    fn aliases(&self) -> &'static [&'static str] {
        &["java"]
    }

    fn describe(&self) -> Vec<String> {
        vec![
            "java.util.Random, a 48-bit LCG, usually seeded from \
             System.currentTimeMillis(). The full 2^48 space is NOT feasible to sweep: \
             narrow --start and --end to the window the wallet was created in, which is \
             what makes this tractable. A single known day is 86.4 million points."
                .into(),
        ]
    }

    fn guide(&self) -> Guide {
        Guide {
            what: "Java's built-in `Random` class is a simple, predictable generator \
                   meant for simulations, not keys. A wallet built on it -- almost always \
                   seeded from the current time in milliseconds -- can be recreated by \
                   anyone who knows roughly when it was made.",
            affected: "Any Java or Android wallet tool using `java.util.Random` instead \
                       of `SecureRandom`. No CVE: it is a misuse of a correctly-working \
                       class rather than a bug in it.",
            command: "keyforge scan --vuln java-random --start 1420070400000 --end 1451606400000 -f funded.bf",
            time: "You must narrow the range: the full 2^48 seed space is far too large \
                   to sweep. If the seed was a millisecond timestamp, one known day is \
                   86.4 million points and takes under a minute; a known year is about \
                   an hour. The example command scans the whole of 2015.",
            hit: "A mnemonic phrase or a 64-character private key per line in \
                  matches.txt. Because you scanned a window rather than the whole space, \
                  a clean sweep only rules out that window.",
        }
    }

    /// The full seed space the constructor can hold. See the module note: sweeping all
    /// of this is not feasible and the guide says so rather than pretending otherwise.
    fn space(&self) -> Space {
        Space::Integers { start: 0, end: 1 << 48 }
    }

    fn defaults(&self) -> Defaults {
        Defaults {
            material_sizes: vec![16, 32],
            routes: vec![Route::Bip39, Route::Bip32Seed, Route::PrivKey],
            paths: Vec::new(),
        }
    }

    /// Seeds are overwhelmingly `System.currentTimeMillis()`, so narrowing to a known
    /// creation window is not merely sound here -- it is the only way to run this at all.
    fn range_is_narrowable(&self) -> bool {
        true
    }

    fn expand_at(&self, point: Point<'_>, offsets: &[usize], out: &mut Vec<Expanded>) {
        let Point::Integer(n) = point else {
            debug_assert!(false, "java-random does not read a corpus");
            return;
        };
        for &offset in offsets {
            let mut rng = JavaRandom::new(n as u64);
            // Through `next_bytes`, which is how a real earlier `nextBytes(byte[])` would
            // have advanced it: whole 32-bit draws, four bytes at a time. `offset_step`
            // is what keeps the length a multiple of that, so this consumes exactly the
            // draws the earlier call did and no partial one.
            if offset > 0 {
                let mut skipped = vec![0u8; offset];
                rng.next_bytes(&mut skipped);
            }
            let mut material = [0u8; 32];
            rng.next_bytes(&mut material);
            out.push(material);
        }
    }

    /// `java.util.Random` hands out 32-bit draws, so only whole words are real stream
    /// positions -- an offset between two of them names a wallet no program produced.
    fn offset_step(&self) -> Option<usize> {
        Some(4)
    }

    /// One stream. The only plugin whose kernel genuinely needs both halves of the
    /// point: the seed space is 2^48, and a millisecond window in 2015 is already past
    /// 2^32.
    fn kernel_at(&self, offsets: &[usize]) -> Option<KernelSpec> {
        // One stream, walked once per offset.
        Some(KernelSpec {
            source: KERNEL_SOURCE,
            streams: vec![0; offsets.len()],
            offsets: offsets.iter().map(|&o| o as u32).collect(),
            defines: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `new Random(0).nextInt()` is -1155484576.
    ///
    /// This is the one value in the module quotable independently of any implementation:
    /// it appears in the Java documentation's own examples and in a great many textbooks,
    /// and it pins the constructor's seed scramble, the LCG step and the 32-bit
    /// truncation all at once. If the scramble were skipped this would not match.
    #[test]
    fn matches_the_published_java_first_value() {
        let mut rng = JavaRandom::new(0);
        assert_eq!(rng.next_int() as i32, -1_155_484_576);

        // The next two, and a second seed, to pin the LCG step rather than just the
        // constructor.
        assert_eq!(rng.next_int() as i32, -723_955_400);
        assert_eq!(rng.next_int() as i32, 1_033_096_058);
        assert_eq!(JavaRandom::new(42).next_int() as i32, -1_170_105_035);
    }

    /// `nextBytes(new byte[32])` for several seeds.
    ///
    /// **Weaker oracle than the rest of this crate**: these come from an independent
    /// implementation of the published Javadoc algorithm rather than from a JVM, because
    /// the machine this was written on has none. See the module note. Run
    /// `agrees_with_a_real_jvm` where a JVM exists to close the gap.
    #[test]
    fn matches_the_specified_next_bytes_stream() {
        let vectors = [
            (0u64, "60b420bb3851d9d47acb933dbe70399bf6c92da33af01d4fb770e98c0325f41d"),
            (1, "73d51abbd89cb8196f0efb6892f94d68fccc2c35f0b84609e5f12c55dd85aba8"),
            (500, "fef101c2c4b12c1f19549160bd182947a33e568c8bba5227150fcedd19a50e8f"),
            (1234567890, "ea6addb72596428eca77acf9a43797b34b58fcf1a0363cf35ec04f2620d015d0"),
            (4294967295, "b36c882bc3220ae31720eb769eff06a2695e1dec4cbb80d19ab3d2f30cc5a23c"),
        ];
        for (seed, want) in vectors {
            let mut out = Vec::new();
            JavaUtilRandom.expand(Point::Integer(seed as u128), &mut out);
            assert_eq!(hex::encode(out[0]), want, "new Random({seed}).nextBytes(...)");
        }
    }

    /// The real oracle, for a machine that has a JVM. Skips with a message otherwise,
    /// the same way the filter-backed tests do.
    ///
    /// ```java
    /// // Oracle.java
    /// import java.util.Random;
    /// public class Oracle {
    ///     public static void main(String[] a) {
    ///         for (long s : new long[]{0, 1, 500, 1234567890L, 4294967295L}) {
    ///             byte[] b = new byte[32];
    ///             new Random(s).nextBytes(b);
    ///             StringBuilder h = new StringBuilder();
    ///             for (byte x : b) h.append(String.format("%02x", x));
    ///             System.out.println(s + " " + h);
    ///         }
    ///     }
    /// }
    /// ```
    #[test]
    fn agrees_with_a_real_jvm() {
        let Ok(java) = std::env::var("KEYFORGE_JAVA") else {
            eprintln!("skipping: set KEYFORGE_JAVA to a `java` binary to run this");
            return;
        };
        let dir = std::env::temp_dir().join("keyforge-java-oracle");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let src = dir.join("Oracle.java");
        std::fs::write(
            &src,
            r#"import java.util.Random;
public class Oracle {
    public static void main(String[] a) {
        for (long s : new long[]{0, 1, 500, 1234567890L, 4294967295L}) {
            byte[] b = new byte[32];
            new Random(s).nextBytes(b);
            StringBuilder h = new StringBuilder();
            for (byte x : b) h.append(String.format("%02x", x));
            System.out.println(s + " " + h);
        }
    }
}"#,
        )
        .expect("write oracle");

        // Modern JVMs run a single source file directly, no javac step.
        let output = std::process::Command::new(&java)
            .arg(&src)
            .output()
            .expect("run java");
        assert!(
            output.status.success(),
            "java failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let text = String::from_utf8(output.stdout).expect("utf-8");
        let mut checked = 0;
        for line in text.lines() {
            let (seed, want) = line.split_once(' ').expect("seed and hex");
            let seed: u64 = seed.parse().expect("seed");
            let mut out = Vec::new();
            JavaUtilRandom.expand(Point::Integer(seed as u128), &mut out);
            assert_eq!(hex::encode(out[0]), want, "JVM disagrees for seed {seed}");
            checked += 1;
        }
        assert_eq!(checked, 5, "the JVM did not report every seed");
    }

    /// The constructor scrambles its argument. Skipping that is the most common way to
    /// get this class wrong, and it would produce a plausible but entirely wrong stream.
    #[test]
    fn the_constructor_scrambles_the_seed() {
        assert_eq!(JavaRandom::new(0).seed, MULTIPLIER);
        assert_ne!(JavaRandom::new(12345).seed, 12345);
    }

    /// A short tail must take only the bytes it needs, not round up to a whole word.
    #[test]
    fn a_short_buffer_takes_a_partial_word() {
        let mut full = [0u8; 8];
        JavaRandom::new(7).next_bytes(&mut full);
        for n in 1..8usize {
            let mut short = vec![0u8; n];
            JavaRandom::new(7).next_bytes(&mut short);
            assert_eq!(short, full[..n], "a {n}-byte buffer diverged from the prefix");
        }
    }

    #[test]
    fn distinct_seeds_give_distinct_material() {
        let mut seen = std::collections::HashSet::new();
        for seed in 0..64u128 {
            let mut out = Vec::new();
            JavaUtilRandom.expand(Point::Integer(seed), &mut out);
            assert!(seen.insert(out[0]), "seed {seed} repeated an earlier wallet");
        }
    }
}
