//! Wallets whose private key is the hash of a passphrase somebody chose.
//!
//! A brainwallet trades an unguessable 256-bit key for a phrase a person can remember,
//! and the second thing is worth far less than the first. `sha256("satoshi")` is a valid
//! private key, and so is the hash of every song lyric, quotation and password anyone
//! has ever written down.
//!
//! # The search space is a file, not a range
//!
//! Every other vulnerability here walks an integer range, and its cost is known before
//! it starts. This one walks a corpus -- a wordlist, a password dump, a book -- so the
//! space is whatever the user supplies, and the sweep is only ever as good as that file.
//! A clean pass proves the passphrase was not in the corpus and nothing more, which is a
//! much weaker statement than the other guides can make, and the guide says so.
//!
//! # Only one route applies
//!
//! The hash *is* the key. There is no entropy to feed to BIP39 and no seed to derive a
//! tree from, so the default is [`Route::PrivKey`] alone. That also means the derivation
//! path set is irrelevant here, and leaving it wide is pure waste.

use crate::crypto::hash::hash256;
use crate::scan::derive::Route;
use crate::vuln::{Defaults, Expanded, Guide, Point, Space, Vulnerability};
use sha2::{Digest, Sha256};

/// The classic brainwallet: `privkey = sha256(passphrase)`.
pub struct Brainwallet;

impl Vulnerability for Brainwallet {
    fn id(&self) -> &'static str {
        "brainwallet"
    }
    fn aliases(&self) -> &'static [&'static str] {
        &["brain"]
    }

    fn describe(&self) -> Vec<String> {
        vec![
            "brainwallet (no CVE -- a practice, not a bug)".into(),
            "privkey = sha256(passphrase), over a corpus you supply. The search space is \
             that file, so a clean pass proves the passphrase was not in your wordlist \
             and nothing more. Only the privkey route applies: the hash is the key, so \
             there is no phrase to stretch and no tree to walk."
                .into(),
        ]
    }

    fn guide(&self) -> Guide {
        Guide {
            what: "A brainwallet turns a passphrase you can remember into a private key \
                   by hashing it. Anything a person can remember, a computer can guess: \
                   these have been emptied at scale since 2013, usually within seconds \
                   of being funded.",
            affected: "Any wallet created from a remembered passphrase -- the old \
                       brainwallet.org, `bitaddress.org`'s brain wallet tab, and any \
                       `sha256(phrase)` script. Not a software bug, so there is no CVE \
                       and no version to check.",
            command: "keyforge scan --vuln brainwallet --corpus phrases.txt -f funded.bf",
            time: "As long as your corpus takes -- this walks a file, not a number \
                   range. Roughly a million phrases a second per core, so a 14-million \
                   word list is seconds and a large password dump is minutes.",
            hit: "One 64-character private key per line in matches.txt. The passphrase \
                  behind it is recorded in matches.jsonl if you passed --details. A \
                  clean sweep only means the phrase was not in your corpus.",
        }
    }

    fn space(&self) -> Space {
        Space::Corpus
    }

    /// The hash is the key: one route, one size, and no derivation path to walk.
    fn defaults(&self) -> Defaults {
        Defaults {
            material_sizes: vec![32],
            routes: vec![Route::PrivKey],
            paths: Vec::new(),
        }
    }

    fn expand(&self, point: Point<'_>, out: &mut Vec<Expanded>) {
        let Point::Input(passphrase) = point else {
            debug_assert!(false, "brainwallet walks a corpus, not an integer range");
            return;
        };
        out.push(Sha256::digest(passphrase).into());
        // The other mapping in the wild: some generators double-hashed, matching
        // Bitcoin's usual hash256 rather than a bare SHA-256. It is one extra hash per
        // phrase against a corpus that is cheap to walk anyway, and missing it would
        // mean a clean sweep over a corpus that did contain the phrase.
        out.push(hash256(passphrase));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SHA-256 values printed by CPython's `hashlib`, an implementation with no shared
    /// code with this one.
    #[test]
    fn matches_reference_sha256_of_known_passphrases() {
        let vectors = [
            ("satoshi", "da2876b3eb31edb4436fa4650673fc6f01f90de2f1793c4ec332b2387b09726f"),
            ("correct horse battery staple",
             "c4bbcb1fbec99d65bf59d85c8cb62ee2db963f0fe106f483d9afa73bd4e39a8a"),
            ("password", "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8"),
            ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        ];
        for (phrase, want) in vectors {
            let mut out = Vec::new();
            Brainwallet.expand(Point::Input(phrase.as_bytes()), &mut out);
            assert_eq!(hex::encode(out[0]), want, "sha256({phrase:?})");
        }
    }

    /// Both mappings are walked, and they are genuinely different keys -- if the second
    /// ever equalled the first it would be a wasted probe per phrase.
    #[test]
    fn walks_both_the_single_and_double_hash() {
        let mut out = Vec::new();
        Brainwallet.expand(Point::Input(b"satoshi"), &mut out);
        assert_eq!(out.len(), 2);
        assert_ne!(out[0], out[1]);
        // The second is hash256, i.e. sha256 applied to the first.
        assert_eq!(out[1], hash256(b"satoshi"));
    }

    /// The scope must not multiply this by derivation paths or BIP39: the hash is the
    /// key, and any other route would derive wallets that cannot exist.
    #[test]
    fn narrows_to_the_privkey_route_alone() {
        let d = Brainwallet.defaults();
        assert_eq!(d.routes, vec![Route::PrivKey]);
        assert_eq!(d.material_sizes, vec![32]);

        let mut scope = crate::scan::derive::Scope::default();
        d.apply(&mut scope);
        // One key, one probe per hash form -- not the 1,443 a full HD scope would ask
        // for. Getting this wrong would make a corpus sweep hundreds of times slower for
        // no possible extra finding.
        assert_eq!(scope.probes_per_point(), scope.forms.len() as u64);
    }

    /// A corpus vulnerability has no countable space; the engine has to read the file to
    /// know, and code that assumed otherwise would divide by a length it never had.
    #[test]
    fn reports_an_uncountable_space() {
        assert_eq!(Brainwallet.space(), Space::Corpus);
        assert_eq!(Brainwallet.space().len(), None);
    }
}
