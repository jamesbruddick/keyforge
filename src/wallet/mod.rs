//! Wallet standards: how key material becomes a phrase, a tree, and an address.
//!
//! Split from `crate::crypto` because these are *conventions* rather than primitives.
//! A vulnerability decides what bytes exist; this module decides what a wallet would
//! have done with them.

pub mod address;
pub mod bip39;
pub mod path;
