//! The two short hashes a Bitcoin address commits to.
//!
//! Both hashers come from `RustCrypto` and pick their own SHA-NI / ARMv8 backend at
//! runtime, so there is no assembly feature to enable here and nothing to special-case
//! per target. Neither is on a hot enough path to justify the hand-written multi-lane
//! treatment `crate::crypto::sha512` gets: at ~1,400 probes per seed these are a few
//! percent of the budget, where SHA-512 inside PBKDF2 is closer to forty.

use ripemd::Ripemd160;
use sha2::{Digest, Sha256};

/// RIPEMD160(SHA256(data)).
#[inline]
pub fn hash160(data: &[u8]) -> [u8; 20] {
    let sha = Sha256::digest(data);
    Ripemd160::digest(sha).into()
}

/// SHA256(SHA256(data)) -- the checksum half of base58check, and Bitcoin's
/// double-SHA everywhere else.
#[inline]
pub fn hash256(data: &[u8]) -> [u8; 32] {
    Sha256::digest(Sha256::digest(data)).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published hash160 of the genesis coinbase public key, which is the payload of
    /// the address every Bitcoin node has agreed on since 2009. An independent value:
    /// it comes from the chain, not from another run of this code.
    #[test]
    fn hashes_the_genesis_public_key_to_its_known_address_payload() {
        let pubkey = hex_bytes(
            "04678afdb0fe5548271967f1a67130b7105cd6a828e03909a67962e0ea1f61deb\
             649f6bc3f4cef38c4f35504e51ec112de5c384df7ba0b8d578a4c702b6bf11d5f",
        );
        assert_eq!(
            hash160(&pubkey),
            hex_bytes("62e907b15cbf27d5425399ebf6f0fb50ebb88f18")[..]
        );
    }

    /// `hash256("")` against the value every double-SHA implementation reports.
    #[test]
    fn double_sha_matches_the_published_empty_digest() {
        assert_eq!(
            hash256(b"")[..],
            hex_bytes("5df6e0e2761359d30a8275058e299fcc0381534545f55cf43e41983f5d4c9456")[..]
        );
    }

    fn hex_bytes(s: &str) -> Vec<u8> {
        let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
            .collect()
    }
}
