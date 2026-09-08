//! Private keys that are not random at all.
//!
//! These are the simplest vulnerabilities in the registry and among the most productive:
//! no generator to reimplement and no assumption about what software did, just keys with
//! so little entropy that enumerating them is trivial. They exist because a truncated
//! buffer, an uninitialised variable, a test fixture left in production or a
//! deliberately-set "1" all produce a key someone can reach.
//!
//! Both vulnerabilities here expand to the key directly, so only [`Route::PrivKey`]
//! applies and no derivation path is walked -- see the note in
//! [`crate::vuln::brainwallet`], which narrows for the same reason.

use crate::scan::derive::Route;
use crate::vuln::{Defaults, Expanded, Guide, KernelSpec, Point, Space, Vulnerability};

/// The device half of [`LowInteger::expand`].
const LOW_INT_KERNEL: &str = include_str!("../../kernels/vuln/low_int.h");

/// The device half of [`RepeatedByte::expand`].
const REPEATED_BYTE_KERNEL: &str = include_str!("../../kernels/vuln/repeated_byte.h");

/// Private keys that are just small numbers: `1`, `2`, `3`, ...
///
/// Famous as the "puzzle" addresses, but also what a generator produces when its
/// randomness silently returns zeros and a counter is added, or when a buffer holds an
/// index rather than a key. Key `1` is the secp256k1 generator point itself, which is
/// why `ec::tests::the_generator_is_the_published_point` doubles as ground truth here.
pub struct LowInteger;

impl Vulnerability for LowInteger {
    fn id(&self) -> &'static str {
        "low-int"
    }

    fn classification(&self) -> &'static str {
        "no CVE -- a failure mode, not a product"
    }
    fn aliases(&self) -> &'static [&'static str] {
        &["puzzle", "small-key"]
    }

    fn describe(&self) -> Vec<String> {
        vec![
            "Private keys that are small integers: 1, 2, 3 and upwards. Produced by a \
             generator whose randomness returned zeros, by a buffer holding an index \
             rather than a key, and deliberately by the puzzle addresses. Only the \
             privkey route applies, so this is the cheapest sweep in the registry."
                .into(),
        ]
    }

    fn guide(&self) -> Guide {
        Guide {
            what: "Some wallets end up with a private key that is just a small number -- \
                   1, 2, 3 and so on -- because the randomness failed and returned zeros, \
                   or because a counter was stored where a key should have been. These \
                   keys are trivially guessable and are swept constantly.",
            affected: "No single product. This is what broken randomness looks like when \
                       it fails to zero, plus the deliberately-funded puzzle addresses. \
                       Worth running first on any investigation because it is so cheap.",
            command: "keyforge scan --vuln low-int --end 16777216 -f funded.bf",
            time: "Seconds to minutes. There is one key per point and no tree to derive, \
                   so this runs hundreds of times faster per point than a BIP39 sweep -- \
                   16 million keys is a few seconds.",
            hit: "One 64-character private key per line in matches.txt. These addresses \
                  are watched by many people, so a hit is very unlikely to still hold a \
                  spendable balance.",
        }
    }

    /// Starts at 1: zero is not a valid secp256k1 scalar, and including it would make
    /// the first point of every run a guaranteed skip.
    fn space(&self) -> Space {
        Space::Integers { start: 1, end: 1 << 32 }
    }

    fn defaults(&self) -> Defaults {
        Defaults {
            material_sizes: vec![32],
            routes: vec![Route::PrivKey],
            paths: Vec::new(),
        }
    }

    fn expand(&self, point: Point<'_>, out: &mut Vec<Expanded>) {
        let Point::Integer(n) = point else {
            debug_assert!(false, "low-int does not read a corpus");
            return;
        };
        // Big-endian in the low 16 bytes, which is how a 128-bit counter lands in a
        // 32-byte key buffer. `u128` rather than `u64` so the space can be widened
        // without changing the layout of what has already been scanned.
        let mut key = [0u8; 32];
        key[16..].copy_from_slice(&n.to_be_bytes());
        out.push(key);
    }

    /// One stream: a point is one key, so there is no byte mapping to walk.
    fn kernel(&self) -> Option<KernelSpec> {
        Some(KernelSpec { source: LOW_INT_KERNEL, streams: vec![0], defines: Vec::new() })
    }
}

/// Private keys that are one byte repeated: `0x0101..01`, `0xffff..ff`.
///
/// What `memset` on the wrong buffer leaves behind, and what an uninitialised page of
/// a fresh allocation often holds. Tiny enough that it costs nothing to include.
pub struct RepeatedByte;

impl Vulnerability for RepeatedByte {
    fn id(&self) -> &'static str {
        "repeated-byte"
    }

    fn classification(&self) -> &'static str {
        "no CVE -- a failure mode, not a product"
    }

    fn describe(&self) -> Vec<String> {
        vec![
            "Private keys made of one byte repeated 32 times, which is what a memset on \
             the wrong buffer or an uninitialised page leaves behind. Only 255 keys, so \
             it costs nothing to include in any investigation."
                .into(),
        ]
    }

    fn guide(&self) -> Guide {
        Guide {
            what: "A private key made of the same byte 32 times over -- all 0x01s, all \
                   0xffs. This is what a wallet ends up with when a buffer is filled with \
                   a constant instead of randomness, which happens more often than it \
                   should.",
            affected: "No single product; it is a symptom of a memory bug rather than a \
                       design mistake. There are only 255 such keys, so it is worth \
                       checking regardless.",
            command: "keyforge scan --vuln repeated-byte -f funded.bf",
            time: "Instant. There are 255 keys in the whole space.",
            hit: "One 64-character private key per line in matches.txt.",
        }
    }

    /// 1..=255. Byte zero would be the all-zero key, which is not a valid scalar.
    fn space(&self) -> Space {
        Space::Integers { start: 1, end: 256 }
    }

    fn defaults(&self) -> Defaults {
        Defaults {
            material_sizes: vec![32],
            routes: vec![Route::PrivKey],
            paths: Vec::new(),
        }
    }

    fn expand(&self, point: Point<'_>, out: &mut Vec<Expanded>) {
        let Point::Integer(n) = point else {
            debug_assert!(false, "repeated-byte does not read a corpus");
            return;
        };
        out.push([n as u8; 32]);
    }

    /// One stream. The whole space is 255 points, so this kernel buys nothing on the
    /// clock -- a device spends longer opening than the CPU spends finishing. It exists
    /// so that `--gpu` over a list of vulnerabilities does not stop at this one.
    fn kernel(&self) -> Option<KernelSpec> {
        Some(KernelSpec {
            source: REPEATED_BYTE_KERNEL,
            streams: vec![0],
            defines: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::ec;
    use crate::crypto::hash::hash160;
    use crate::wallet::address::{HashForm, encode};

    /// Private key 1 is the secp256k1 generator, whose two addresses are among the most
    /// widely published in Bitcoin. Ground truth from the chain, not from this code.
    #[test]
    fn key_one_is_the_published_generator_address() {
        let mut out = Vec::new();
        LowInteger.expand(Point::Integer(1), &mut out);
        assert_eq!(hex::encode(out[0]), format!("{:0>64}", "1"));

        let public = ec::public_key(&out[0]);
        let compressed = encode(HashForm::Compressed, &hash160(&public.serialize()));
        assert_eq!(
            compressed.split(' ').next(),
            Some("1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH")
        );
        let uncompressed = encode(
            HashForm::Uncompressed,
            &hash160(&public.serialize_uncompressed()),
        );
        assert_eq!(uncompressed, "1EHNa6Q4Jz2uvNExL497mE43ikXhwF6kZm");
    }

    /// The counter has to land where the guide says it does, since a user narrowing
    /// `--end` is relying on point `n` meaning key `n`.
    #[test]
    fn point_n_is_key_n() {
        for n in [1u128, 2, 255, 256, 4_294_967_295] {
            let mut out = Vec::new();
            LowInteger.expand(Point::Integer(n), &mut out);
            assert_eq!(
                u128::from_be_bytes(out[0][16..].try_into().unwrap()),
                n,
                "point {n} did not produce key {n}"
            );
            assert_eq!(&out[0][..16], &[0u8; 16], "key {n} has high bits set");
        }
    }

    /// Neither space may contain the all-zero key, which is not a valid scalar: the walk
    /// would silently drop it and report a point as scanned that never was.
    #[test]
    fn no_space_includes_the_invalid_zero_key() {
        for (v, first) in [
            (&LowInteger as &dyn Vulnerability, 1u128),
            (&RepeatedByte as &dyn Vulnerability, 1),
        ] {
            let Space::Integers { start, .. } = v.space() else {
                panic!("{} should walk an integer range", v.id())
            };
            assert_eq!(start, first, "{} starts at the zero key", v.id());

            let mut out = Vec::new();
            v.expand(Point::Integer(start), &mut out);
            assert_ne!(out[0], [0u8; 32]);
            assert!(
                secp256k1::SecretKey::from_byte_array(out[0]).is_ok(),
                "{}'s first point is not a valid key",
                v.id()
            );
        }
    }

    #[test]
    fn repeated_byte_fills_the_whole_key() {
        let mut out = Vec::new();
        RepeatedByte.expand(Point::Integer(0xab), &mut out);
        assert_eq!(out[0], [0xabu8; 32]);
        // 255 keys, and every one of them valid.
        assert_eq!(RepeatedByte.space().len(), Some(255));
    }
}
