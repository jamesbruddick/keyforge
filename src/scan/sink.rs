//! Where a filter match goes.
//!
//! Two files, with one job each.
//!
//! `matches.txt` holds **one importable secret per line and nothing else** -- either a
//! BIP39 mnemonic or a 64-character private key. No columns, no timestamps, no
//! commentary, so it can be pasted straight into a wallet or piped into another tool
//! without anything having to parse it first.
//!
//! `matches.jsonl` holds everything else -- which vulnerability, which point, which
//! derivation path, which address form -- one JSON object per hit, and only when
//! `--details` asks for it. That information is genuinely useful and keeping it out of
//! the first file is what preserves that file's one job.
//!
//! # Which secret gets written
//!
//! The rule is: **whatever a user would import to control the funds.** For the BIP39
//! route that is the phrase, which recovers the whole wallet. For every other route it
//! is the **private key of the leaf that matched**, and the distinction matters more
//! than it looks.
//!
//! The scanner this grew out of wrote the phrase for its `raw-master` route too, on the
//! grounds that the phrase encodes the same bytes. It does -- but a wallet given that
//! phrase runs it through PBKDF2 first, and lands on a completely different master key.
//! The phrase names the right bytes and opens the wrong wallet. The old code had this
//! right for raw private keys, where it wrote the key, and wrong for raw seeds. Writing
//! the leaf key covers both, and is exactly what the funds sit under.
//!
//! Re-deriving that key costs an unshared field inversion, which is why it happens here
//! rather than being carried through the walk: a match is roughly a one-in-a-billion
//! event and the hot loop should not pay for it.

use crate::scan::derive::{Location, Route, Scope, leaf_private_key};
use crate::ui::{self, Ui};
use crate::wallet::{address, bip39};
use anyhow::{Context, Result};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// Everything known about one filter match.
pub struct Hit<'a> {
    /// The vulnerability's id, for the details file.
    pub vuln: &'static str,
    /// The point of the search space that produced it.
    pub point: u128,
    /// Which of the point's materials, for a vulnerability that expands to more than
    /// one byte stream. A point alone does not name a wallet when it yields several.
    pub stream: usize,
    /// The material, truncated to the size this location used.
    pub material: &'a [u8],
    pub location: &'a Location,
    pub hash: &'a [u8; 20],
    /// The BIP39 phrase for this material, which the walk already wrote.
    pub phrase: &'a str,
}

/// The secret to write for a hit: the whole of its line in `matches.txt`.
///
/// See the module note for why this is not simply the phrase.
pub fn secret_of(hit: &Hit<'_>, scope: &Scope) -> String {
    if hit.location.route == Route::Bip39 {
        return hit.phrase.to_string();
    }
    let spec = scope.paths.get(hit.location.spec as usize);
    match leaf_private_key(hit.material, hit.location.route, spec, hit.location.leaf as u64) {
        Some(key) => ui::hex(&key),
        // Unreachable in practice: the walk only emitted this leaf because it derived
        // it. Falling back to the material keeps a find from being lost outright if it
        // ever happens, and it is still a 64-character key.
        None => ui::hex(hit.material),
    }
}

pub struct MatchSink<'a> {
    ui: &'a Ui,
    file: File,
    details: Option<File>,
    seen: HashSet<String>,
    /// Probe positions that matched. One secret can match at several paths and forms,
    /// so this is always at least `candidates` and usually more.
    pub locations: u64,
    /// Distinct secrets, which is exactly the number of lines in the file.
    pub candidates: u64,
}

impl<'a> MatchSink<'a> {
    pub fn open(ui: &'a Ui, path: &Path, details: Option<&Path>) -> Result<Self> {
        // Pre-load existing lines so a resumed run does not repeat itself.
        let mut seen = HashSet::new();
        if path.exists() {
            let reader = BufReader::new(
                File::open(path).with_context(|| format!("reading {}", path.display()))?,
            );
            for line in reader.lines() {
                seen.insert(line?);
            }
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("opening {}", path.display()))?;

        let details = match details {
            Some(p) => Some(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(p)
                    .with_context(|| format!("opening {}", p.display()))?,
            ),
            None => None,
        };

        Ok(Self { ui, file, details, seen, locations: 0, candidates: 0 })
    }

    /// Record a filter match.
    ///
    /// The secret is the whole output line, as promised; the context is announced on the
    /// terminal and written to the details file, where neither pollutes `matches.txt`.
    pub fn record(&mut self, hit: &Hit<'_>, scope: &Scope) {
        self.locations += 1;
        let secret = secret_of(hit, scope);
        // One point can match at several paths, and a resumed run re-derives secrets
        // already in the file, so the written line is what decides whether this is a new
        // one -- and the counter follows the file, not the probe.
        let first = self.seen.insert(secret.clone());

        // Where this came from, on one line: the vulnerability and point name the
        // wallet, path and form name the key, and the address and hash160 are what a
        // block explorer and the filter respectively were asked about.
        //
        // The point leads every announcement rather than only the first, so each one
        // stands on its own -- another thread's find can land between two paths of the
        // same point, and a line that reads as a continuation of something no longer
        // above it is worse than a repeated word.
        let headline = format!(
            "{} {} · {}B · {} · {} · {} · {}  {}",
            hit.vuln,
            hit.point,
            hit.location.material_len,
            hit.location.route.as_str(),
            hit.location.path(scope).unwrap_or_else(|| "(raw key)".into()),
            hit.location.form.as_str(),
            self.ui.data(&address::encode(hit.location.form, hit.hash)),
            self.ui.dim(&ui::hex(hit.hash))
        );

        if let Some(details) = &mut self.details {
            let _ = writeln!(details, "{}", details_line(hit, scope, &secret));
            let _ = details.flush();
        }

        if !first {
            // A further path of a secret already announced and already written. The
            // label carries both halves of that: it is not a `candidate`, so the blocks
            // under that label stay equal in number to the lines in the file, and it
            // names the secret as one already shown, which is why none is printed
            // under it.
            let repeat = match hit.location.route {
                Route::Bip39 => "same phrase",
                _ => "same key",
            };
            self.ui.announce(repeat, &headline, &[]);
            return;
        }

        self.candidates += 1;
        // The secret gets the second line to itself: it is the whole output of the
        // sweep, it is what the triage tools want pasted into them, and at 24 words it
        // would push everything else off the edge if it shared a line.
        self.ui.announce("candidate", &headline, &[self.ui.data(&secret)]);
        // Matches are rare enough that syncing on each one is free, and it means a crash
        // or a power cut never costs a candidate.
        let _ = writeln!(self.file, "{secret}");
        let _ = self.file.flush();
        let _ = self.file.sync_data();
    }
}

/// One JSON object for the details file.
///
/// Hand-rolled rather than pulling in a serialiser for six fields, none of which need
/// escaping beyond the phrase -- which is BIP39 words and spaces, and cannot contain a
/// quote or a backslash. The path can contain an apostrophe, which JSON does not care
/// about.
fn details_line(hit: &Hit<'_>, scope: &Scope, secret: &str) -> String {
    let path = hit.location.path(scope).unwrap_or_default();
    format!(
        r#"{{"vuln":"{}","point":{},"stream":{},"material":"{}","material_len":{},"route":"{}","path":"{}","form":"{}","address":"{}","hash160":"{}","secret":"{}"}}"#,
        hit.vuln,
        hit.point,
        hit.stream,
        ui::hex(hit.material),
        hit.location.material_len,
        hit.location.route.as_str(),
        path,
        hit.location.form.as_str(),
        address::encode(hit.location.form, hit.hash),
        ui::hex(hit.hash),
        secret,
    )
}

/// Whether a line of `matches.txt` is one of the two things it is allowed to be.
///
/// Used by the tests that hold the file to its contract, and cheap enough to be worth
/// exposing rather than duplicating.
pub fn is_importable_secret(line: &str) -> bool {
    let is_key = line.len() == 64 && line.bytes().all(|b| b.is_ascii_hexdigit());
    is_key || bip39::mnemonic_is_valid(line)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wallet::address::HashForm;
    use crate::wallet::path::PathSpec;

    fn scope() -> Scope {
        Scope {
            material_sizes: vec![32],
            routes: vec![Route::Bip39, Route::Bip32Seed, Route::PrivKey],
            paths: vec![PathSpec::parse("m/44'/0'/0'/0/0").unwrap()],
            forms: vec![HashForm::Compressed],
        }
    }

    fn hit<'a>(material: &'a [u8], route: Route, _phrase: &str) -> (Location, &'a [u8]) {
        (
            Location {
                batch_index: 0,
                material_len: 32,
                route,
                spec: 0,
                leaf: 0,
                form: HashForm::Compressed,
            },
            material,
        )
    }

    /// Every route must write something a wallet can actually import, and the two
    /// permitted shapes are the only two that appear.
    #[test]
    fn every_route_writes_an_importable_secret() {
        let scope = scope();
        let material = [7u8; 32];
        let phrase = bip39::mnemonic(&material);

        for route in [Route::Bip39, Route::Bip32Seed, Route::PrivKey] {
            let (location, material) = hit(&material, route, &phrase);
            let h = Hit {
                vuln: "test",
                point: 1,
                stream: 0,
                material,
                location: &location,
                hash: &[0u8; 20],
                phrase: &phrase,
            };
            let secret = secret_of(&h, &scope);
            assert!(
                is_importable_secret(&secret),
                "{route:?} wrote something unimportable: {secret:?}"
            );
        }
    }

    /// The BIP39 route writes the phrase; the others write a 64-character key.
    ///
    /// The `bip32-seed` half is the bug this module's note is about: writing the phrase
    /// there would name the right bytes and open the wrong wallet, because a wallet runs
    /// a phrase through PBKDF2 before deriving anything.
    #[test]
    fn a_seed_route_writes_the_leaf_key_not_the_phrase() {
        let scope = scope();
        let material = [7u8; 32];
        let phrase = bip39::mnemonic(&material);

        let (bip39_loc, m) = hit(&material, Route::Bip39, &phrase);
        let bip39_secret = secret_of(
            &Hit { vuln: "t", point: 1, stream: 0, material: m, location: &bip39_loc,
                   hash: &[0; 20], phrase: &phrase },
            &scope,
        );
        assert_eq!(bip39_secret, phrase);

        let (seed_loc, m) = hit(&material, Route::Bip32Seed, &phrase);
        let seed_secret = secret_of(
            &Hit { vuln: "t", point: 1, stream: 0, material: m, location: &seed_loc,
                   hash: &[0; 20], phrase: &phrase },
            &scope,
        );
        assert_eq!(seed_secret.len(), 64);
        assert_ne!(seed_secret, phrase);

        // And it is genuinely the key at that leaf, not the material itself.
        assert_ne!(seed_secret, ui::hex(&material));
        let expected = leaf_private_key(&material, Route::Bip32Seed, scope.paths.first(), 0);
        assert_eq!(seed_secret, ui::hex(&expected.unwrap()));
    }

    /// A private-key route writes the key itself, which is the material.
    #[test]
    fn a_privkey_route_writes_the_material() {
        let scope = scope();
        let material = [7u8; 32];
        let phrase = bip39::mnemonic(&material);
        let (location, m) = hit(&material, Route::PrivKey, &phrase);
        let secret = secret_of(
            &Hit { vuln: "t", point: 1, stream: 0, material: m, location: &location,
                   hash: &[0; 20], phrase: &phrase },
            &scope,
        );
        assert_eq!(secret, ui::hex(&material));
    }

    /// The contract check itself has to reject the things it is there to catch.
    #[test]
    fn the_importable_check_rejects_anything_else() {
        assert!(is_importable_secret(&"a".repeat(64)));
        assert!(is_importable_secret(&bip39::mnemonic(&[0u8; 32])));
        assert!(!is_importable_secret(""));
        assert!(!is_importable_secret(&"a".repeat(63)));
        assert!(!is_importable_secret(&"z".repeat(64)));
        assert!(!is_importable_secret("seed 500 m/44'/0'/0'/0/0"));
        // A phrase with a broken checksum is not importable either.
        assert!(!is_importable_secret(
            "abandon abandon abandon abandon abandon abandon \
             abandon abandon abandon abandon abandon abandon"
        ));
    }
}
