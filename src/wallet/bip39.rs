//! BIP39 English mnemonics, both directions.
//!
//! Encoding is the hot path -- the scanner produces a phrase per point per material
//! size, billions of times -- and is written to reuse one caller-owned `String`.
//!
//! Decoding is entirely cold. It exists for `verify`, which takes a phrase from the
//! command line, and for the check that every line written to `matches.txt` really is
//! an importable secret. Neither cares about speed, so it is written for clarity.

use sha2::{Digest, Sha256};
use std::sync::LazyLock;

/// The official BIP39 English wordlist, sha256 `2f5eed53a4727b4bf8880d8f3f199efc90e58503646d9ff8eff3a2ed3b24dbda`
/// over the newline-joined file.
static WORDLIST_SRC: &str = include_str!("../../assets/wordlist.txt");

pub static WORDLIST: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    let words: Vec<&str> = WORDLIST_SRC.split_whitespace().collect();
    assert_eq!(words.len(), 2048, "BIP39 wordlist must contain 2048 words");
    words
});

/// Every entropy size BIP39 defines, in bytes: 12, 15, 18, 21 and 24 words.
///
/// The scanner this grew out of listed only 16, 24 and 32, because those were the three
/// `bx seed` could emit. Keeping that here would make the encoder narrower than the
/// decoder and, worse, would panic on a vulnerability that produced 20 or 28 bytes --
/// which is a property of the affected software, not something this module gets to
/// decide. A vulnerability still narrows to the sizes it could actually have produced
/// through its own defaults.
pub const VALID_ENTROPY_SIZES: [usize; 5] = [16, 20, 24, 28, 32];

/// Longest mnemonic: 24 words of at most 8 chars plus separators.
pub const MAX_PHRASE_LEN: usize = 24 * 9;

/// Write the mnemonic for `entropy` into `out`, which is cleared first.
///
/// `out` is caller-owned so the hot loop can reuse one allocation across millions of
/// seeds. `entropy.len()` must be 16, 24 or 32.
pub fn write_mnemonic(entropy: &[u8], out: &mut String) {
    debug_assert!(VALID_ENTROPY_SIZES.contains(&entropy.len()));

    let words = &*WORDLIST;
    let checksum_bits = entropy.len() * 8 / 32;
    let checksum = Sha256::digest(entropy)[0] >> (8 - checksum_bits);
    let total_bits = entropy.len() * 8 + checksum_bits;

    out.clear();
    // Walk the entropy||checksum bitstream 11 bits at a time through a rolling
    // accumulator, rather than materialising a 264-bit big integer. `acc` never
    // holds more than 18 live bits, so a u32 is ample; stale high bits are masked
    // off by the 0x7ff extraction.
    let mut acc: u32 = 0;
    let mut acc_bits: usize = 0;
    let mut emitted = 0;

    // The checksum rides along as a final, narrower "byte".
    let checksum_pos = entropy.len();
    for (i, &byte) in entropy.iter().chain(std::iter::once(&checksum)).enumerate() {
        let width = if i == checksum_pos { checksum_bits } else { 8 };
        acc = (acc << width) | byte as u32;
        acc_bits += width;

        while acc_bits >= 11 {
            let index = ((acc >> (acc_bits - 11)) & 0x7ff) as usize;
            acc_bits -= 11;
            if emitted > 0 {
                out.push(' ');
            }
            out.push_str(words[index]);
            emitted += 1;
        }
    }

    debug_assert_eq!(acc_bits, 0);
    debug_assert_eq!(emitted, total_bits / 11);
}

/// A phrase back to its entropy, or `None` if it is not a valid BIP39 mnemonic.
///
/// Rejects, in this order: a word count that is not 12, 15, 18, 21 or 24; a word not in
/// the list; and a checksum that does not match. The checksum is the point -- a phrase
/// with the right words in the wrong order almost always fails it, which is what makes
/// this a real check on a line of output rather than a spell check.
///
/// Word lookup is linear over 2048 entries. That is fine: the callers are `verify` and
/// the output contract test, not the walk.
pub fn decode(phrase: &str) -> Option<Vec<u8>> {
    let words = &*WORDLIST;
    let parts: Vec<&str> = phrase.split_whitespace().collect();
    if !matches!(parts.len(), 12 | 15 | 18 | 21 | 24) {
        return None;
    }

    // 11 bits per word, of which the trailing `words / 3` are the checksum. Expanded to
    // one bit per entry rather than shifted through an accumulator: this is a cold path,
    // and the accumulator version has an off-by-one waiting in it at every boundary.
    let mut bits: Vec<u8> = Vec::with_capacity(parts.len() * 11);
    for part in &parts {
        let index = words.iter().position(|w| w == part)? as u16;
        for shift in (0..11).rev() {
            bits.push(((index >> shift) & 1) as u8);
        }
    }

    let checksum_bits = parts.len() / 3;
    let entropy_bits = bits.len() - checksum_bits;
    let mut entropy = vec![0u8; entropy_bits / 8];
    for (i, bit) in bits[..entropy_bits].iter().enumerate() {
        entropy[i / 8] |= bit << (7 - i % 8);
    }

    // The checksum is the leading bits of sha256(entropy). This is what makes decoding a
    // real check rather than a spell check: the right words in the wrong order almost
    // always fail here.
    let want = Sha256::digest(&entropy)[0];
    for (i, bit) in bits[entropy_bits..].iter().enumerate() {
        if *bit != (want >> (7 - i)) & 1 {
            return None;
        }
    }
    Some(entropy)
}

/// Whether `phrase` is a well-formed BIP39 mnemonic with a valid checksum.
pub fn mnemonic_is_valid(phrase: &str) -> bool {
    decode(phrase).is_some()
}

/// Convenience wrapper for tests, `verify`, and other cold paths.
pub fn mnemonic(entropy: &[u8]) -> String {
    let mut out = String::with_capacity(MAX_PHRASE_LEN);
    write_mnemonic(entropy, &mut out);
    out
}

#[cfg(test)]
mod decode_tests {
    use super::*;

    /// Decoding must invert encoding at every valid size, for entropy that is not all
    /// one byte -- an off-by-one in the bit packing survives an all-zero round trip.
    #[test]
    fn round_trips_every_entropy_size() {
        for size in [16usize, 20, 24, 28, 32] {
            let entropy: Vec<u8> = (0..size).map(|i| (i as u8).wrapping_mul(37).wrapping_add(11)).collect();
            let phrase = mnemonic(&entropy);
            assert_eq!(
                decode(&phrase).as_deref(),
                Some(&entropy[..]),
                "{size}-byte entropy did not survive a round trip"
            );
        }
    }

    /// The published BIP39 vectors, decoded.
    #[test]
    fn decodes_the_published_vectors() {
        assert_eq!(
            decode(
                "abandon abandon abandon abandon abandon abandon \
                 abandon abandon abandon abandon abandon about"
            )
            .as_deref(),
            Some(&[0u8; 16][..])
        );
        assert_eq!(
            decode(
                "legal winner thank year wave sausage worth useful legal winner thank yellow"
            )
            .as_deref(),
            Some(&[0x7f; 16][..])
        );
    }

    /// A wrong checksum must be rejected, or this is a spell check rather than a
    /// validity check -- and `matches.txt` would be allowed to contain phrases that open
    /// nothing.
    #[test]
    fn rejects_a_broken_checksum() {
        // The all-zero vector with its last word changed to another valid word.
        let bad = "abandon abandon abandon abandon abandon abandon \
                   abandon abandon abandon abandon abandon abandon";
        assert_eq!(decode(bad), None);

        // Swapping two words almost always breaks the checksum.
        assert_eq!(
            decode("about abandon abandon abandon abandon abandon \
                    abandon abandon abandon abandon abandon abandon"),
            None
        );
    }

    #[test]
    fn rejects_malformed_phrases() {
        assert_eq!(decode(""), None);
        assert_eq!(decode("abandon"), None);
        // A word not in the list.
        assert_eq!(
            decode("zzzz abandon abandon abandon abandon abandon \
                    abandon abandon abandon abandon abandon about"),
            None
        );
        // 13 words: not a valid length.
        assert_eq!(
            decode("abandon abandon abandon abandon abandon abandon abandon \
                    abandon abandon abandon abandon abandon about"),
            None
        );
    }

    /// Extra whitespace is not a reason to reject a phrase a user pasted.
    #[test]
    fn tolerates_surrounding_whitespace() {
        let phrase = mnemonic(&[0u8; 16]);
        let padded = format!("  {}  ", phrase.replace(' ', "   "));
        assert_eq!(decode(&padded).as_deref(), Some(&[0u8; 16][..]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vuln::mt19937::entropy_for_seed;

    #[test]
    fn wordlist_is_the_official_english_list() {
        assert_eq!(WORDLIST.len(), 2048);
        assert_eq!(WORDLIST[0], "abandon");
        assert_eq!(WORDLIST[2047], "zoo");
        let joined: String = WORDLIST.iter().map(|w| format!("{w}\n")).collect();
        assert_eq!(
            hex::encode(Sha256::digest(joined.as_bytes())),
            "2f5eed53a4727b4bf8880d8f3f199efc90e58503646d9ff8eff3a2ed3b24dbda"
        );
    }

    /// The namesake. Seed 0 at 128 bits is the mnemonic the vulnerability is named
    /// after; the 192- and 256-bit forms extend the same entropy prefix.
    #[test]
    fn reproduces_the_milk_sad_mnemonics_for_seed_zero() {
        let entropy = entropy_for_seed(0);
        assert_eq!(
            mnemonic(&entropy[..16]),
            "milk sad wage cup reward umbrella raven visa give list decorate broccoli"
        );
        assert_eq!(
            mnemonic(&entropy[..24]),
            "milk sad wage cup reward umbrella raven visa give list decorate bulb \
             gold raise twenty fly manual sport"
        );
        assert_eq!(
            mnemonic(&entropy[..32]),
            "milk sad wage cup reward umbrella raven visa give list decorate bulb \
             gold raise twenty fly manual stand float super gentle climb fold park"
        );
    }

    #[test]
    fn produces_the_right_word_counts() {
        let entropy = entropy_for_seed(12345);
        for (bytes, words) in [(16, 12), (24, 18), (32, 24)] {
            assert_eq!(mnemonic(&entropy[..bytes]).split(' ').count(), words);
        }
    }

    /// Held against the reference BIP39 implementation over random entropy, so the
    /// bit-packing is checked well beyond the Milk Sad vectors.
    #[test]
    fn agrees_with_the_reference_bip39_implementation() {
        for seed in 0..2000u32 {
            let entropy = entropy_for_seed(seed);
            for size in VALID_ENTROPY_SIZES {
                let ours = mnemonic(&entropy[..size]);
                let theirs = reference_mnemonic(&entropy[..size]);
                assert_eq!(ours, theirs, "seed {seed} size {size}");
            }
        }
    }

    /// A deliberately naive, obviously-correct big-integer encoder to check the
    /// streaming one against.
    fn reference_mnemonic(entropy: &[u8]) -> String {
        let checksum_bits = entropy.len() * 8 / 32;
        let checksum = Sha256::digest(entropy)[0] >> (8 - checksum_bits);
        let mut bits: Vec<bool> = Vec::new();
        for &b in entropy {
            for i in (0..8).rev() {
                bits.push((b >> i) & 1 == 1);
            }
        }
        for i in (0..checksum_bits).rev() {
            bits.push((checksum >> i) & 1 == 1);
        }
        bits.chunks(11)
            .map(|c| {
                let idx = c.iter().fold(0usize, |a, &b| (a << 1) | b as usize);
                WORDLIST[idx]
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// The 100M known-good phrases the verified Python reference generated for
    /// seeds 10,000,000..110,000,000 at 16-byte entropy.
    ///
    /// Named by an env var, with the sibling checkout as the fallback -- an
    /// absolute path baked in here is a path on one machine, and the test would
    /// then skip everywhere else while still reporting success.
    const CORPUS_VAR: &str = "KEYFORGE_CORPUS";
    const CORPUS_DEFAULT: &str = "../allkeys-keycheck/milk-sad-10000000-110000000.txt";
    const CORPUS_FIRST_SEED: u32 = 10_000_000;

    fn corpus_path() -> std::path::PathBuf {
        match std::env::var_os(CORPUS_VAR) {
            Some(path) => std::path::PathBuf::from(path),
            // Relative to the manifest, not to whatever directory `cargo test`
            // happened to be invoked from.
            None => std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(CORPUS_DEFAULT),
        }
    }

    fn check_corpus(limit: usize) -> usize {
        use std::io::{BufRead, BufReader};

        let path = corpus_path();
        let Ok(file) = std::fs::File::open(&path) else {
            eprintln!(
                "skipping: no corpus at {}; set {CORPUS_VAR} to run this",
                path.display()
            );
            return 0;
        };
        // 1 MiB of buffer keeps a multi-GB sequential read from being syscall-bound.
        let reader = BufReader::with_capacity(1 << 20, file);

        let mut buf = String::with_capacity(MAX_PHRASE_LEN);
        let mut checked = 0usize;
        for (offset, line) in reader.lines().take(limit).enumerate() {
            let expected = line.unwrap();
            let seed = CORPUS_FIRST_SEED + offset as u32;
            write_mnemonic(&entropy_for_seed(seed)[..16], &mut buf);
            assert_eq!(buf, expected, "seed {seed} (corpus line {})", offset + 1);
            checked += 1;
        }
        checked
    }

    /// The end-to-end check on the entropy pipeline: MT seeding, the libstdc++ byte
    /// distribution, and BIP39 encoding, against a million phrases produced by an
    /// independent implementation.
    #[test]
    fn agrees_with_the_reference_corpus() {
        let checked = check_corpus(1_000_000);
        if checked > 0 {
            assert_eq!(checked, 1_000_000);
        }
    }

    /// The same check over all 100 million phrases. Takes a few minutes, so it is
    /// opt-in: `cargo test --release -- --ignored`. Worth running once, because at
    /// 1.6e9 entropy bytes it exercises roughly ten thousand of the rare cases where
    /// libstdc++'s division disagrees with a naive shift.
    #[test]
    #[ignore = "reads a 7.7 GB corpus; run explicitly"]
    fn agrees_with_the_entire_reference_corpus() {
        let checked = check_corpus(usize::MAX);
        if checked > 0 {
            assert_eq!(checked, 100_000_000);
        }
    }

    #[test]
    fn write_mnemonic_reuses_the_buffer_cleanly() {
        let mut buf = String::new();
        write_mnemonic(&entropy_for_seed(0)[..32], &mut buf);
        let long = buf.clone();
        write_mnemonic(&entropy_for_seed(0)[..16], &mut buf);
        assert_eq!(buf.split(' ').count(), 12);
        assert!(long.starts_with("milk sad"));
    }
}
