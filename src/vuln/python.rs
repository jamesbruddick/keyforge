//! Wallets built on CPython's `random` module.
//!
//! `random.seed(n)` is MT19937, and a script that seeded it with a timestamp, a PID or
//! a small integer and then drew wallet entropy from it left a search space the size of
//! whatever it seeded with.
//!
//! # This is not `std::mt19937` seeded with the same number
//!
//! CPython does not call `init_genrand`. It converts the integer to an array of 32-bit
//! words and runs `init_by_array` over it, which produces a completely unrelated stream.
//! `random.seed(500)` and `std::mt19937 twister(500)` share a name and nothing else.
//! Scanning one while meaning the other is the worst kind of failure available here: it
//! finishes, reports a clean sweep, and has checked nothing. `Mt19937::from_key` is the
//! CPython path, and [`tests::matches_real_cpython_output`] pins it against values a
//! real interpreter printed.
//!
//! # Which bytes a script would have taken
//!
//! CPython exposes several ways to get bytes out, and they are not the same stream.
//! `getrandbits(32)` is one raw 32-bit output; `os.urandom` is not this generator at
//! all. The mapping scanned here is successive `getrandbits(32)` words in
//! **little-endian** byte order, which is what `random.getrandbits(256).to_bytes(32,
//! 'little')` and the common `bytes([random.getrandbits(8) ...])` idioms both reduce to.
//! That is an assumption about the affected script, not a fact about the generator, and
//! the guide says so.

use crate::scan::derive::Route;
use crate::vuln::mt19937::Mt19937;
use crate::vuln::{Defaults, Expanded, Guide, KernelSpec, Point, Space, Vulnerability};

/// The device half of [`PythonRandom::expand`].
const KERNEL_SOURCE: &str = include_str!("../../kernels/vuln/python.h");

/// A wallet generator seeded with `random.seed(n)` over a 32-bit `n`.
pub struct PythonRandom;

impl Vulnerability for PythonRandom {
    fn id(&self) -> &'static str {
        "python-random"
    }

    fn classification(&self) -> &'static str {
        "no CVE -- a generator class, not a product"
    }
    fn aliases(&self) -> &'static [&'static str] {
        &["python", "cpython"]
    }

    fn describe(&self) -> Vec<String> {
        vec![
            "CPython random.seed(n) over a 32-bit n: MT19937 seeded through init_by_array, \
             which is a different stream from std::mt19937 with the same number. The byte \
             mapping scanned is successive getrandbits(32) words, little-endian, which is \
             an assumption about the script rather than a property of the generator."
                .into(),
        ]
    }

    fn guide(&self) -> Guide {
        Guide {
            what: "Python's `random` module is a predictable generator meant for \
                   simulations, not keys. A script that called `random.seed(...)` with \
                   something small -- a timestamp, a process id, a counter -- and then \
                   built a wallet from it left only about 4.3 billion possibilities.",
            affected: "Any script generating wallets with the `random` module rather \
                       than `secrets` or `os.urandom`. No CVE: this is a class of \
                       mistake rather than one shipped bug, so a hit tells you about \
                       one script.",
            command: "keyforge scan --vuln python-random -f funded.bf",
            time: "Around 9 days on a laptop, or 9 hours on a recent NVIDIA card. The \
                   seed has no time structure unless you know the script used a clock, \
                   so by default the whole range is walked.",
            hit: "A mnemonic phrase or a 64-character private key per line in \
                  matches.txt. Note that this scan assumes the script read bytes a \
                  particular way; a clean sweep does not rule the script out.",
        }
    }

    fn space(&self) -> Space {
        Space::Integers { start: 0, end: 1 << 32 }
    }

    fn defaults(&self) -> Defaults {
        Defaults {
            material_sizes: vec![16, 32],
            routes: vec![Route::Bip39, Route::Bip32Seed, Route::PrivKey],
            paths: Vec::new(),
        }
    }

    fn expand(&self, point: Point<'_>, out: &mut Vec<Expanded>) {
        let Point::Integer(n) = point else {
            debug_assert!(false, "python-random does not read a corpus");
            return;
        };
        let mut rng = Mt19937::from_key(&[n as u32]);
        let mut material = [0u8; 32];
        for chunk in material.chunks_exact_mut(4) {
            chunk.copy_from_slice(&rng.next_u32().to_le_bytes());
        }
        out.push(material);
    }

    /// One stream: the byte mapping this scans is a documented assumption about the
    /// script, not a choice between two libraries the way `milksad`'s is.
    fn kernel(&self) -> Option<KernelSpec> {
        Some(KernelSpec { source: KERNEL_SOURCE, streams: vec![0], defines: Vec::new() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real CPython output, printed by an interpreter rather than by another run of this
    /// code:
    ///
    /// ```text
    /// python3 -c "import random
    /// for s in (0,1,500,4294967295):
    ///     random.seed(s)
    ///     print(s, [format(random.getrandbits(32),'08x') for _ in range(8)])"
    /// ```
    ///
    /// This is the test the whole module rests on. `init_by_array` differs from
    /// `init_genrand` in a way that produces plausible-looking wrong answers, so nothing
    /// short of the real interpreter's numbers proves the seeding is right.
    #[test]
    fn matches_real_cpython_output() {
        let vectors: [(u32, [u32; 8]); 4] = [
            (0, [0xd82c07cd, 0x629f6fbe, 0xc2094cac, 0xe3e70682,
                 0x6baa9455, 0x0a5d2f34, 0x42485e3a, 0xf728b4fa]),
            (1, [0x2265b1f5, 0x91b7584a, 0xd8f16adf, 0xcd613e30,
                 0xc386bbc4, 0x1027c4d1, 0x414c343c, 0x1e2feb89]),
            (500, [0xcc1e50f8, 0x762bb111, 0xf3104e05, 0x86d704d2,
                   0x940b5fef, 0x77f1862b, 0x417175e4, 0xa2b0033f]),
            (u32::MAX, [0xa2a6c909, 0x9e9c04f4, 0x3404e941, 0x3720bfa9,
                        0x9b7c60b4, 0x85bbddd4, 0x4a94f707, 0x9a6188fb]),
        ];

        for (seed, want) in vectors {
            let mut rng = Mt19937::from_key(&[seed]);
            let got: Vec<u32> = (0..8).map(|_| rng.next_u32()).collect();
            assert_eq!(got, want, "CPython stream mismatch for random.seed({seed})");
        }
    }

    /// CPython's seeding must not be confused with `std::mt19937`'s. If these ever agree
    /// the two constructors have been crossed, and a sweep would silently scan the wrong
    /// generator -- see the module note.
    #[test]
    fn is_a_different_stream_from_std_mt19937() {
        for seed in [0u32, 1, 500, u32::MAX] {
            let cpython = Mt19937::from_key(&[seed]).next_u32();
            let cpp = Mt19937::new(seed).next_u32();
            assert_ne!(
                cpython, cpp,
                "init_by_array and init_genrand agree for {seed}; one of them is wrong"
            );
        }
    }

    /// The material must be the little-endian byte order the module documents, since
    /// that is the assumption a user is relying on.
    #[test]
    fn material_is_the_documented_byte_order() {
        let mut out = Vec::new();
        PythonRandom.expand(Point::Integer(500), &mut out);
        assert_eq!(out.len(), 1);
        // First word for seed 500 is 0xcc1e50f8, little-endian.
        assert_eq!(&out[0][..4], &[0xf8, 0x50, 0x1e, 0xcc]);
    }

    /// Different seeds must give different wallets -- a constructor that ignored its
    /// argument would otherwise pass every other test here.
    #[test]
    fn distinct_seeds_give_distinct_material() {
        let mut seen = std::collections::HashSet::new();
        for seed in 0..64u128 {
            let mut out = Vec::new();
            PythonRandom.expand(Point::Integer(seed), &mut out);
            assert!(seen.insert(out[0]), "seed {seed} repeated an earlier wallet");
        }
    }
}
