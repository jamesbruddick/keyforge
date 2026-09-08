//! keyforge -- a scanner for Bitcoin vulnerabilities that produce guessable keys.
//!
//! The pipeline is one shape, and every vulnerability enters at the front of it:
//!
//! ```text
//!   point -> secret material -> BIP32 tree -> hash160 forms -> bloom filter -> candidate
//! ```
//!
//! A *point* is one element of a vulnerability's search space -- a 32-bit timestamp for
//! Milk Sad, a 48-bit LCG state for `java.util.Random`, a line of a passphrase corpus
//! for a brainwallet. `crate::vuln` turns a point into secret material; everything after
//! that is shared, which is why adding a vulnerability is a small, local piece of work.
//!
//! The layers, in dependency order:
//!
//! * [`crypto`] -- field arithmetic, the fixed-base multiplier, SHA-512, PBKDF2, hash160.
//!   Knows nothing about wallets.
//! * [`wallet`] -- BIP39, addresses. Conventions rather than primitives.
//! * [`vuln`] -- the vulnerabilities themselves, and the registry the CLI names them by.
//! * [`scan`] -- the walk, the work queue, the checkpoint, the output file.
//! * [`target`] -- the bloom filter a derived hash160 is tested against.
//! * [`ui`] -- the terminal.
//!
//! Note what is *not* in that list: nothing below `vuln` depends on anything above it.
//! A change to a vulnerability cannot reach the arithmetic.

pub mod crypto;
pub mod gpu;
pub mod scan;
pub mod target;
pub mod ui;
pub mod vuln;
pub mod wallet;
