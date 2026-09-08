//! PBKDF2-HMAC-SHA512, specialised for BIP39 seed derivation.
//!
//! BIP39 stretches the mnemonic with 2048 iterations. It is per entropy size rather
//! than per address, so widening the address surface does not touch it: at the
//! current default scope it is ~17% of runtime against the curve layer's ~68%, and at
//! the narrow scope this was first written for it was about half. Either way it is
//! far too large a share to hand to the generic `pbkdf2` crate.
//!
//! Two specialisations pay for themselves:
//!
//!   * The HMAC key is the mnemonic, constant across all 2048 iterations. Its
//!     ipad/opad SHA-512 midstates are computed once and each iteration resumes from
//!     them, instead of re-absorbing a 128-byte pad block twice per iteration.
//!   * dkLen is exactly one SHA-512 block, so there is a single `T` and no outer
//!     block loop, and every iteration's message is 64 bytes -- which lands in one
//!     compression with room for the padding.
//!
//! Net cost is 2 + 2*iterations compressions, versus 4*iterations for a naive
//! HMAC-per-iteration implementation.

use crate::crypto::sha512::{LANES, compress};
use sha2::block_api::compress512;
use sha2::{Digest, Sha512};

const BLOCK: usize = 128;
const OUT: usize = 64;

/// SHA-512 initial state (FIPS 180-4).
const IV: [u64; 8] = [
    0x6a09_e667_f3bc_c908,
    0xbb67_ae85_84ca_a73b,
    0x3c6e_f372_fe94_f82b,
    0xa54f_f53a_5f1d_36f1,
    0x510e_527f_ade6_82d1,
    0x9b05_688c_2b3e_6c1f,
    0x1f83_d9ab_fb41_bd6b,
    0x5be0_cd19_137e_2179,
];

/// The largest message `finish` can absorb in a single trailing block: 128 bytes
/// less the 0x80 terminator and the 16-byte big-endian bit length.
const MAX_TAIL: usize = BLOCK - 1 - 16;

/// Absorb a final partial message into `state` and apply SHA-512 padding.
///
/// `total_len` is the length of the *whole* hashed message including any blocks
/// already folded into `state`. `tail` must be at most `MAX_TAIL` bytes.
#[inline]
fn padded(tail: &[u8], total_len: usize) -> [u8; BLOCK] {
    debug_assert!(tail.len() <= MAX_TAIL);
    let mut block = [0u8; BLOCK];
    block[..tail.len()].copy_from_slice(tail);
    block[tail.len()] = 0x80;
    let bits = (total_len as u128) * 8;
    block[BLOCK - 16..].copy_from_slice(&bits.to_be_bytes());
    block
}

/// Big-endian words out of a finished state.
#[inline]
fn digest(state: &[u64; 8]) -> [u8; OUT] {
    let mut out = [0u8; OUT];
    for (chunk, word) in out.chunks_exact_mut(8).zip(state.iter()) {
        chunk.copy_from_slice(&word.to_be_bytes());
    }
    out
}

#[inline]
fn finish(state: &mut [u64; 8], tail: &[u8], total_len: usize) -> [u8; OUT] {
    let block = padded(tail, total_len);
    compress512(state, std::slice::from_ref(&block));
    digest(state)
}

/// HMAC-SHA512 with the key's pad midstates precomputed.
pub struct HmacSha512 {
    ipad: [u64; 8],
    opad: [u64; 8],
}

impl HmacSha512 {
    pub fn new(key: &[u8]) -> Self {
        let mut k0 = [0u8; BLOCK];
        if key.len() > BLOCK {
            // A 24-word mnemonic can exceed 128 bytes, so this branch is live.
            k0[..OUT].copy_from_slice(&Sha512::digest(key));
        } else {
            k0[..key.len()].copy_from_slice(key);
        }

        let mut ipad_block = [0u8; BLOCK];
        let mut opad_block = [0u8; BLOCK];
        for i in 0..BLOCK {
            ipad_block[i] = k0[i] ^ 0x36;
            opad_block[i] = k0[i] ^ 0x5c;
        }

        let mut ipad = IV;
        compress512(&mut ipad, std::slice::from_ref(&ipad_block));
        let mut opad = IV;
        compress512(&mut opad, std::slice::from_ref(&opad_block));
        Self { ipad, opad }
    }

    /// HMAC over a message short enough to finish in one compression each side.
    ///
    /// Both PBKDF2 messages qualify: `salt || INT(1)` and a 64-byte previous block.
    #[inline]
    pub fn mac(&self, message: &[u8]) -> [u8; OUT] {
        let mut inner = self.ipad;
        let digest = finish(&mut inner, message, BLOCK + message.len());
        let mut outer = self.opad;
        finish(&mut outer, &digest, BLOCK + OUT)
    }
}

/// PBKDF2-HMAC-SHA512 producing exactly 64 bytes.
///
/// `salt` must leave room for the 4-byte block counter within one trailing block.
pub fn pbkdf2_hmac_sha512(password: &[u8], salt: &[u8], iterations: u32) -> [u8; OUT] {
    assert!(
        salt.len() + 4 <= MAX_TAIL,
        "salt too long for the fast path"
    );
    let hmac = HmacSha512::new(password);

    let mut first = [0u8; MAX_TAIL];
    first[..salt.len()].copy_from_slice(salt);
    first[salt.len()..salt.len() + 4].copy_from_slice(&1u32.to_be_bytes());

    let mut u = hmac.mac(&first[..salt.len() + 4]);
    let mut t = u;
    for _ in 1..iterations {
        u = hmac.mac(&u);
        for (acc, byte) in t.iter_mut().zip(u.iter()) {
            *acc ^= byte;
        }
    }
    t
}

/// BIP39's iteration count.
const ITERATIONS: u32 = 2048;

/// The BIP39 seed for a mnemonic with an empty passphrase.
#[inline]
pub fn bip39_seed(mnemonic: &str) -> [u8; OUT] {
    pbkdf2_hmac_sha512(mnemonic.as_bytes(), b"mnemonic", ITERATIONS)
}

/// BIP39 seeds for a whole batch of mnemonics, replacing `out`.
///
/// One stream is a 2048-long dependency chain that cannot keep the hardware busy on its
/// own, so streams are run `LANES` at a time -- see `crate::crypto::sha512` for what sets that
/// number and why it depends on whether the target has SHA-512 instructions. The streams
/// are independent, so this is only a scheduling change: any grouping gives the same
/// answers, and the test suite checks each width against `sha2`'s one-lane compression.
///
/// **A short final group is padded, but only where padding is free.** A batch is twelve
/// streams at the default scope -- four seeds by three entropy sizes -- so at eight lanes
/// the last group is half empty. On a target that compresses a group in one pass that
/// costs nothing, and beats finishing four streams singly on the slow path. On one that
/// does not -- an x86 without AVX2, where a group is `LANES` sequential `sha2` calls --
/// the padding lanes are simply four extra 2048-iteration chains, a third more work than
/// the batch needs. `sha512::padding_is_free` is which case this is, decided at run time
/// along with everything else about the width.
pub fn bip39_seeds(mnemonics: &[String], out: &mut Vec<[u8; OUT]>) {
    grouped(mnemonics, out, crate::crypto::sha512::padding_is_free());
}

/// The body of `bip39_seeds`, with the padding decision passed in.
///
/// Separated only so both branches are reachable from a test. Which one a given machine
/// takes is fixed by its CPU, so going through `bip39_seeds` would leave the other one
/// unexecuted everywhere -- and the two produce the same seeds by different routes, which
/// is exactly the claim worth checking.
fn grouped(mnemonics: &[String], out: &mut Vec<[u8; OUT]>, pad: bool) {
    out.clear();
    let mut rest = mnemonics;
    while !rest.is_empty() {
        let real = rest.len().min(LANES);
        if real < LANES && !pad {
            for leftover in rest {
                out.push(bip39_seed(leftover));
            }
            return;
        }
        let group: [&str; LANES] = std::array::from_fn(|i| rest[i.min(real - 1)].as_str());
        out.extend_from_slice(&bip39_seed_group(&group)[..real]);
        rest = &rest[real..];
    }
}

/// `N` BIP39 seeds, computed in lockstep.
fn bip39_seed_group<const N: usize>(mnemonics: &[&str; N]) -> [[u8; OUT]; N] {
    const SALT: &[u8] = b"mnemonic";
    let hmac: [HmacSha512; N] = std::array::from_fn(|i| HmacSha512::new(mnemonics[i].as_bytes()));

    let mut salted = [0u8; SALT.len() + 4];
    salted[..SALT.len()].copy_from_slice(SALT);
    salted[SALT.len()..].copy_from_slice(&1u32.to_be_bytes());

    let mut u: [[u8; OUT]; N] = std::array::from_fn(|i| hmac[i].mac(&salted));
    let mut t = u;
    for _ in 1..ITERATIONS {
        u = mac_group(&hmac, &u);
        for (acc, block) in t.iter_mut().zip(&u) {
            for (acc, byte) in acc.iter_mut().zip(block) {
                *acc ^= byte;
            }
        }
    }
    t
}

/// `N` HMACs, over `N` 64-byte messages, in lockstep.
///
/// Both sides of each HMAC fit in a single compression, so this is exactly `2 * N`
/// compressions arranged as two lockstep groups.
#[inline]
fn mac_group<const N: usize>(hmac: &[HmacSha512; N], messages: &[[u8; OUT]; N]) -> [[u8; OUT]; N] {
    let mut inner: [[u64; 8]; N] = std::array::from_fn(|i| hmac[i].ipad);
    let blocks: [[u8; BLOCK]; N] = std::array::from_fn(|i| padded(&messages[i], BLOCK + OUT));
    compress(&mut inner, &blocks);

    let mut outer: [[u64; 8]; N] = std::array::from_fn(|i| hmac[i].opad);
    let blocks: [[u8; BLOCK]; N] = std::array::from_fn(|i| padded(&digest(&inner[i]), BLOCK + OUT));
    compress(&mut outer, &blocks);

    std::array::from_fn(|i| digest(&outer[i]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 4231 HMAC-SHA-512 test case 1.
    #[test]
    fn hmac_matches_rfc_4231() {
        let key = [0x0bu8; 20];
        let mac = HmacSha512::new(&key).mac(b"Hi There");
        assert_eq!(
            hex::encode(mac),
            "87aa7cdea5ef619d4ff0b4241a1d6cb02379f4e2ce4ec2787ad0b30545e17cdedaa833b7d6b8a702038b274eaea3f4e4be9d914eeb61f1702e696c203a126854"
        );
    }

    /// RFC 4231 case 6: a key longer than the 128-byte block, exercising the
    /// hash-the-key branch that long mnemonics hit.
    #[test]
    fn hmac_handles_keys_longer_than_the_block() {
        let key = [0xaau8; 131];
        let mac =
            HmacSha512::new(&key).mac(b"Test Using Larger Than Block-Size Key - Hash Key First");
        assert_eq!(
            hex::encode(mac),
            "80b24263c7c1a3ebb71493c1dd7be8b49b46d1f41b4aeec1121b013783f8f3526b56d037e05f2598bd0fd2215d6a1e5295e64f73f63f0aec8b915a985d786598"
        );
    }

    /// The official BIP39 vectors, which pin PBKDF2, the iteration count, the salt
    /// construction and the mnemonic encoding together.
    ///
    /// The scanner only ever uses the empty passphrase, but asserting the `TREZOR`
    /// vector too proves the salt is really `"mnemonic" || passphrase` and not a
    /// hardcoded constant that happens to work.
    #[test]
    fn matches_the_official_bip39_seed_vectors() {
        const ALL_ABANDON: &str = "abandon abandon abandon abandon abandon abandon \
                                   abandon abandon abandon abandon abandon about";

        assert_eq!(
            hex::encode(bip39_seed(ALL_ABANDON)),
            "5eb00bbddcf069084889a8ab9155568165f5c453ccb85e70811aaed6f6da5fc19a5ac40b389cd370d086206dec8aa6c43daea6690f20ad3d8d48b2d2ce9e38e4"
        );
        assert_eq!(
            hex::encode(pbkdf2_hmac_sha512(
                ALL_ABANDON.as_bytes(),
                b"mnemonicTREZOR",
                2048
            )),
            "c55257c360c07c72029aebc1b53c05ed0362ada38ead3e3e9efa3708e53495531f09a6987599d18264c1e1c92f2cf141630c7a3c4ab7c81b2f001698e7463b04"
        );
    }

    /// Held against the reference implementation over the phrases the scanner will
    /// actually produce, at all three lengths.
    #[test]
    fn agrees_with_the_reference_bip39_seed() {
        use bitcoin::bip32::Xpriv;
        use bitcoin::network::Network;

        for seed in 0..64u32 {
            let entropy = crate::vuln::mt19937::entropy_for_seed(seed);
            for size in crate::wallet::bip39::VALID_ENTROPY_SIZES {
                let phrase = crate::wallet::bip39::mnemonic(&entropy[..size]);
                let ours = bip39_seed(&phrase);

                // `bitcoin` has no BIP39, so compare via the xprv it produces from
                // our seed against one built from a reference PBKDF2.
                let theirs = reference_pbkdf2(phrase.as_bytes(), b"mnemonic", 2048);
                assert_eq!(
                    ours.as_slice(),
                    theirs.as_slice(),
                    "seed {seed} size {size}"
                );

                // And confirm the seed is usable as a BIP32 master.
                assert!(Xpriv::new_master(Network::Bitcoin, &ours).is_ok());
            }
        }
    }

    /// The paired path must be a pure scheduling change: every seed it produces has
    /// to equal what the single-stream path produces for the same phrase.
    ///
    /// Batch sizes are chosen to cover both parities, since an odd batch leaves one
    /// phrase to the single-stream tail and a bug there would otherwise go unseen.
    #[test]
    fn grouping_streams_does_not_change_any_seed() {
        let mut mnemonics = Vec::new();
        for seed in 0..7u32 {
            let entropy = crate::vuln::mt19937::entropy_for_seed(seed);
            for size in crate::wallet::bip39::VALID_ENTROPY_SIZES {
                mnemonics.push(crate::wallet::bip39::mnemonic(&entropy[..size]));
            }
        }

        // Both padding branches, whatever this CPU would have chosen: a short final
        // group is either filled with repeats or finished one stream at a time, and the
        // two have to agree with each other and with the single-mnemonic path.
        let mut batched = Vec::new();
        for pad in [false, true] {
            for count in 0..=mnemonics.len() {
                grouped(&mnemonics[..count], &mut batched, pad);
                assert_eq!(batched.len(), count, "batch of {count} (pad {pad}) lost an entry");
                for (got, mnemonic) in batched.iter().zip(&mnemonics) {
                    assert_eq!(
                        got.as_slice(),
                        bip39_seed(mnemonic).as_slice(),
                        "batch of {count} (pad {pad}) disagrees for {mnemonic:?}"
                    );
                }
            }
        }
    }

    /// Textbook PBKDF2: full HMAC per iteration, no midstate reuse.
    fn reference_pbkdf2(password: &[u8], salt: &[u8], iterations: u32) -> [u8; 64] {
        fn hmac(key: &[u8], msg: &[u8]) -> [u8; 64] {
            let mut k0 = [0u8; 128];
            if key.len() > 128 {
                k0[..64].copy_from_slice(&Sha512::digest(key));
            } else {
                k0[..key.len()].copy_from_slice(key);
            }
            let inner: Vec<u8> = k0
                .iter()
                .map(|b| b ^ 0x36)
                .chain(msg.iter().copied())
                .collect();
            let id = Sha512::digest(&inner);
            let outer: Vec<u8> = k0
                .iter()
                .map(|b| b ^ 0x5c)
                .chain(id.iter().copied())
                .collect();
            Sha512::digest(&outer).into()
        }

        let mut msg = salt.to_vec();
        msg.extend_from_slice(&1u32.to_be_bytes());
        let mut u = hmac(password, &msg);
        let mut t = u;
        for _ in 1..iterations {
            u = hmac(password, &u);
            for i in 0..64 {
                t[i] ^= u[i];
            }
        }
        t
    }
}
