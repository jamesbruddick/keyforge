//! Cryptographic primitives, none of which know what a vulnerability is.
//!
//! Everything here is held to an independent oracle rather than to another run of
//! itself: the field against `num-bigint`, the multiplier against libsecp256k1 over
//! 500k scalars, SHA-512 and PBKDF2 against RFC 4231 and the published BIP39 vectors.
//! That is what makes it safe to keep the hand-written fast paths below.

pub mod ec;
pub mod field;
pub mod hash;
pub mod pbkdf2;
pub mod sha512;
