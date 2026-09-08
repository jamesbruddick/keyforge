//! What a sweep is looking for: the filter, and the second opinion that came with it.
//!
//! A bloom filter cannot confirm a hit, only rule one out, and at the rate this sweep
//! probes one that is not an abstract worry. The default scope costs 1,443 probes a
//! seed-walk and walks the space twice, once per C++ library, so a full 2^32 sweep asks
//! the filter 12 trillion questions; at the 1e-7 the primary is now sized for that is over
//! a million junk phrases in `matches.txt`, with any real find lost among them.
//!
//! So `keyscan bf-gen` writes a second filter beside the first, `NAME.verify.bf`, over the
//! same entries but probing bits unrelated to them -- see [`crate::target::bloom::resalt`]. It is
//! consulted once per candidate rather than a trillion times, so it costs the sweep
//! nothing, and the two rates multiply: a hash has to clear both to be reported. That is
//! not certainty -- certainty would mean carrying every entry's hash160, four times the
//! size -- but at 1e-10 behind a 1e-7 it turns a million junk lines into approximately
//! none.
//!
//! Where the two filters spend their bits moved with the layouts. The primary loosened
//! from 1e-9 to 1e-7 and the companion tightened from 1e-8 to 1e-10 -- the same product,
//! but the bits moved off the file that has to stay resident and onto the one probed a few
//! hundred times a second. What reaches this side is that a million candidates now cross
//! the primary over a full sweep where thousands used to, which is still nothing beside the
//! days the sweep takes, and that the pair no longer shares a probe schedule.
//!
//! Nothing requires it. A filter built before `bf-gen` wrote companions, or one copied
//! without its own, sweeps exactly as it did before; the startup block is what says which
//! of the two happened, because the companion is picked up from beside the filter rather
//! than asked for and otherwise nothing would tell the operator it was in play.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

pub mod bloom;

use crate::target::bloom::{BloomFilter, FILTER_EXT, Kind, resalt};
use crate::wallet::address::HashForm;

/// Fill below which a filter holds nothing anyone built on purpose.
///
/// A verification filter this empty rejects every candidate the primary passes, which
/// is not a slow sweep or a noisy one but a sweep that cannot report anything at all.
const EMPTY_FILL: f64 = 0.01;

/// A filter and, where one was found beside it, the verification filter over the same
/// entries.
#[derive(Debug)]
pub struct Target {
    primary: BloomFilter,
    verify: Option<BloomFilter>,
}

/// What the target has to say about one hash160.
///
/// Three states rather than a bool because the middle one is worth naming: triage of a
/// line from an older sweep's `matches.txt` wants to distinguish "the filter never saw
/// this" from "the filter passed it and the second one ruled it out", and only the
/// second says the word *false positive* out loud.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The filter ruled it out. Definite: a bloom filter has no false negatives.
    Absent,
    /// The filter passed it and the verification filter ruled it out -- a false positive,
    /// named as such. Also definite, for the same reason.
    RuledOut,
    /// Everything the target has to say agrees. Still "possibly present": both filters
    /// are probabilistic, they are just wrong together far less often than either alone.
    Present,
}

impl Target {
    /// Open the verification filter beside `path`, if there is one.
    ///
    /// The two halves of a target are opened separately rather than behind one call, so
    /// that the caller can name and cost each file as it arrives: for a multi-gigabyte
    /// pair that read is most of the wait before a sweep starts.
    ///
    /// Its absence is ordinary. Its being *unusable* is not: a companion that exists but
    /// will not open, or one with no bits set, stops the run here. The filter was named
    /// on the command line and the operator owns what they pointed at, but this file was
    /// picked up on the strength of its name alone, and a bad one costs the whole sweep
    /// silently -- every candidate rejected, no error, and days later an empty
    /// `matches.txt` that looks exactly like a clean sweep of the keyspace.
    pub fn open_verify(path: &Path) -> Result<Option<BloomFilter>> {
        let companion = verify_path(path);
        if !companion.exists() {
            return Ok(None);
        }
        let filter = BloomFilter::open(&companion).with_context(|| {
            format!(
                "opening the verification filter beside {}. Delete {} to sweep without \
                 one, or rebuild the pair with `keyscan bf-gen`",
                path.display(),
                companion.display()
            )
        })?;
        if filter.sampled_fill_ratio(200_000) < EMPTY_FILL {
            bail!(
                "{} has almost no bits set, so it would reject every candidate the filter \
                 passes and the sweep could not report anything. Delete it to sweep without \
                 a verification filter, or rebuild the pair with `keyscan bf-gen`.",
                companion.display()
            );
        }
        Ok(Some(filter))
    }

    /// A filter and the companion already opened for it.
    pub fn new(primary: BloomFilter, verify: Option<BloomFilter>) -> Target {
        Target { primary, verify }
    }

    /// The filter itself, which is what a GPU screens against. Only the primary is bound
    /// to a device: candidates are rare enough that the second opinion is a host-side
    /// question, and device memory is the budget the launch size comes out of.
    pub fn primary(&self) -> &BloomFilter {
        &self.primary
    }

    /// The verification filter, where one was found.
    pub fn verify(&self) -> Option<&BloomFilter> {
        self.verify.as_ref()
    }

    /// The rate at which a hash that is in neither filter is reported anyway.
    ///
    /// Read out of the bits actually set rather than the rate the filters were sized for,
    /// for the reason [`crate::target::bloom::false_positive_rate`] gives -- and each filter through
    /// its own layout, the pair being a blocked primary and a scattered companion, which do
    /// not admit strangers at the same rate for the same fill. The two are independent tests,
    /// so theirs is the product.
    pub fn false_positive_rate(&self, samples: usize) -> f64 {
        let rate = |f: &BloomFilter| f.false_positive_rate(samples);
        rate(&self.primary) * self.verify.as_ref().map_or(1.0, rate)
    }

    /// The forms in `forms` this filter was never given an entry for, and so cannot match.
    ///
    /// This is the question the `.bf` extension was standing in for. A filter holds the
    /// address forms `bf-gen` was told to take, and until version 2 of the header nothing in
    /// the file said which -- so a sweep deriving P2SH-P2WPKH against a filter built without
    /// `3...` addresses spent a third of its work on probes that could never hit, finished,
    /// and reported a clean sweep of the keyspace. The mask makes that answerable, and this
    /// is the answer.
    ///
    /// Reported rather than refused. The operator named this filter, a narrower one is a
    /// legitimate thing to sweep against, and the pairing with `--hash-forms` is theirs to
    /// make; what they cannot do is see it, which is what this is for.
    pub fn unmatchable_forms(&self, forms: &[HashForm]) -> Vec<HashForm> {
        let kinds = self.primary.kinds();
        forms
            .iter()
            .copied()
            .filter(|form| covering_kinds(*form) & kinds == 0)
            .collect()
    }

    /// Everything the target has to say about one hash160. See [`Verdict`].
    pub fn screen(&self, hash160: &[u8; 20]) -> Verdict {
        if !self.primary.contains(hash160) {
            return Verdict::Absent;
        }
        match &self.verify {
            Some(v) if !v.contains(&resalt(hash160)) => Verdict::RuledOut,
            _ => Verdict::Present,
        }
    }

    /// Is this hash160 worth reporting?
    pub fn contains(&self, hash160: &[u8; 20]) -> bool {
        self.screen(hash160) == Verdict::Present
    }

    /// Test a whole chunk, appending the indices of the survivors to `hits`.
    ///
    /// The chunk goes through the primary in one batch, for the reason
    /// [`BloomFilter::contains_batch`] exists, and only what comes out the other side --
    /// a few per billion -- is re-salted and asked a second time. So the second filter
    /// costs the sweep nothing on the path that matters, however large it is.
    ///
    /// Appends, like the call it wraps, so only the survivors this call added are
    /// filtered and anything already in `hits` is left alone.
    pub fn contains_batch(&self, hashes: &[[u8; 20]], hits: &mut Vec<u32>) {
        let first = hits.len();
        self.primary.contains_batch(hashes, hits);
        let Some(verify) = &self.verify else {
            return;
        };
        let mut kept = first;
        for at in first..hits.len() {
            let i = hits[at];
            if verify.contains(&resalt(&hashes[i as usize])) {
                hits[kept] = i;
                kept += 1;
            }
        }
        hits.truncate(kept);
    }
}

/// The filter entries a derived hash form could be found among.
///
/// A filter entry is the 20 bytes an address commits to, so the question is which address
/// forms commit to the hash this sweep derives -- not which address the scanner would print
/// for it. A compressed-key hash160 is the payload of both a `1...` and a 20-byte `bc1q...`,
/// which is why one entry of either kind can match it. [`Kind::Hex`] covers every form: a
/// hash160 handed to `bf-gen` directly says nothing about what it was a hash of.
fn covering_kinds(form: HashForm) -> u32 {
    Kind::Hex.bit()
        | match form {
            HashForm::Compressed => Kind::P2pkh.bit() | Kind::P2wpkh.bit(),
            // An uncompressed key has no segwit form at all -- BIP143 outlaws it -- so only a
            // `1...` entry can hold this one.
            HashForm::Uncompressed => Kind::P2pkh.bit(),
            // The hash of a redeem script, which reaches a filter as a `3...` and nothing else.
            HashForm::P2shP2wpkh => Kind::P2sh.bit(),
        }
}

/// Where a filter's verification companion lives: beside it, under the same stem.
///
/// Still a `.bf`, because that is the only extension [`BloomFilter::open`] will read and
/// there is no reason for this one to be a different kind of file than the filter it
/// belongs to. The name is `keyscan bf-gen`'s and this side only follows it.
pub fn verify_path(filter: &Path) -> PathBuf {
    let stem = filter.file_stem().unwrap_or_default().to_string_lossy();
    filter.with_file_name(format!("{stem}.verify.{FILTER_EXT}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::bloom::testing::{
        env_path, reference_add, reference_add_blocked, scratch, write_filter, write_filter_as,
    };
    use crate::target::bloom::{BitLayout, Kind, LEGACY_KINDS};

    /// The two halves `load_filter` opens one at a time, for tests that do not care
    /// what each cost.
    fn open(path: &Path) -> Result<Target> {
        Ok(Target::new(
            BloomFilter::open(path)?,
            Target::open_verify(path)?,
        ))
    }

    fn hash(i: u32) -> [u8; 20] {
        let mut h = [0u8; 20];
        for (j, b) in h.iter_mut().enumerate() {
            *b = (i.wrapping_mul(2_654_435_761).rotate_left(j as u32 * 5)) as u8;
        }
        h
    }

    /// A target over `members`, with a verification filter of the given word count over
    /// the same entries. `verify_words` of zero writes no companion at all.
    fn target_over(name: &str, members: &[[u8; 20]], words: usize, verify_words: usize) -> Target {
        let path = scratch(&format!("{name}.bf"));
        let mut bits = vec![0u64; words];
        for m in members {
            reference_add(&mut bits, m);
        }
        write_filter(&path, &bits);

        let companion = verify_path(&path);
        let _ = std::fs::remove_file(&companion);
        if verify_words > 0 {
            let mut bits = vec![0u64; verify_words];
            for m in members {
                reference_add(&mut bits, &resalt(m));
            }
            write_filter(&companion, &bits);
        }
        open(&path).unwrap()
    }

    #[test]
    fn the_companion_sits_beside_the_filter_under_the_same_stem() {
        assert_eq!(
            verify_path(Path::new("/data/addresses.bf")),
            PathBuf::from("/data/addresses.verify.bf")
        );
        assert_eq!(
            verify_path(Path::new("addresses.bf")),
            PathBuf::from("addresses.verify.bf")
        );
    }

    /// The property the verification filter exists for, and the one it must not break.
    ///
    /// Every entry has to survive both filters. A second opinion that rejected real hits
    /// would be far worse than the noise it removes, and the failure would be silent: a
    /// sweep that finds nothing looks exactly like a sweep with nothing to find. The
    /// primary here is saturated, so it passes everything and every rejection below is
    /// the companion's -- which is the arrangement that tests the companion rather than
    /// the filter in front of it.
    #[test]
    fn the_companion_keeps_every_entry_and_rules_out_what_the_filter_waved_through() {
        let members: Vec<[u8; 20]> = (0..200).map(hash).collect();
        let path = scratch("saturated.bf");
        write_filter(&path, &vec![u64::MAX; 256]);
        let mut bits = vec![0u64; 4096];
        for m in &members {
            reference_add(&mut bits, &resalt(m));
        }
        write_filter(&verify_path(&path), &bits);
        let target = open(&path).unwrap();

        for m in &members {
            assert_eq!(target.screen(m), Verdict::Present, "an entry was rejected");
        }
        let ruled_out = (10_000..12_000)
            .map(hash)
            .filter(|h| target.screen(h) == Verdict::RuledOut)
            .count();
        assert_eq!(
            ruled_out, 2000,
            "the filter passes everything, so the companion must rule out every stranger"
        );
    }

    /// Without a companion the target is the filter and nothing else, which is how every
    /// sweep against a filter built before there were companions still works.
    #[test]
    fn a_filter_with_no_companion_answers_exactly_as_the_filter_does() {
        let members: Vec<[u8; 20]> = (0..64).map(hash).collect();
        let target = target_over("no-companion", &members, 4096, 0);
        assert!(target.verify().is_none());
        for m in &members {
            assert_eq!(target.screen(m), Verdict::Present);
        }
        assert_eq!(target.screen(&hash(99_999)), Verdict::Absent);
    }

    /// `contains_batch` is the hot path and a pure batching of `contains`, so the two
    /// must never disagree -- including about the entries the companion removes, which
    /// only the batch path filters after the fact.
    ///
    /// The primary is deliberately small enough to false-positive freely, so the tail
    /// this exercises is not empty. It appends, so a pre-loaded `hits` has to come back
    /// untouched.
    #[test]
    fn batched_lookups_agree_with_single_lookups() {
        let members: Vec<[u8; 20]> = (0..400).map(hash).collect();
        let target = target_over("batched-pair", &members, 512, 4096);

        let hashes: Vec<[u8; 20]> = (0..crate::target::bloom::PROBE_CHUNK as u32)
            .map(|i| {
                if i % 3 == 0 {
                    hash(i)
                } else {
                    hash(i + 10_000)
                }
            })
            .collect();
        assert!(
            hashes.iter().any(|h| target.screen(h) == Verdict::RuledOut),
            "the fixture must produce false positives for the companion to remove"
        );

        let mut hits = vec![u32::MAX];
        for len in 0..=hashes.len() {
            hits.truncate(1);
            target.contains_batch(&hashes[..len], &mut hits);
            let expected: Vec<u32> = std::iter::once(u32::MAX)
                .chain((0..len as u32).filter(|&i| target.contains(&hashes[i as usize])))
                .collect();
            assert_eq!(hits, expected, "chunk of {len} disagrees with contains()");
        }
    }

    /// The pair `keyscan bf-gen` actually writes: a blocked primary and a scattered
    /// companion. The layouts differ per file, so a target that read one schedule for both
    /// would reject every entry it was built from -- silently, and for the length of a
    /// multi-day sweep.
    #[test]
    fn the_pair_bf_gen_writes_has_a_layout_per_file() {
        let members: Vec<[u8; 20]> = (0..200).map(hash).collect();
        let path = scratch("mixed-layouts.bf");
        let mut bits = vec![0u64; 4096];
        for m in &members {
            reference_add_blocked(&mut bits, m);
        }
        write_filter_as(&path, &bits, BitLayout::Blocked, LEGACY_KINDS);

        let mut bits = vec![0u64; 4096];
        for m in &members {
            reference_add(&mut bits, &resalt(m));
        }
        write_filter_as(&verify_path(&path), &bits, BitLayout::Scattered, LEGACY_KINDS);

        let target = open(&path).unwrap();
        assert_eq!(target.primary().layout(), BitLayout::Blocked);
        assert_eq!(target.verify().unwrap().layout(), BitLayout::Scattered);
        for m in &members {
            assert_eq!(target.screen(m), Verdict::Present, "an entry was rejected");
        }
        assert_eq!(target.screen(&hash(99_999)), Verdict::Absent);
    }

    /// A form the filter holds no entries for is named, and one it does hold is not.
    ///
    /// This is the check the `.bf` extension was standing in for before a header recorded
    /// what went into a filter. The failure it catches has no other symptom: the sweep runs
    /// at full speed, probes for something the filter was never given, and ends looking
    /// exactly like a clean sweep of the keyspace.
    #[test]
    fn a_form_the_filter_holds_nothing_for_is_named() {
        let members: Vec<[u8; 20]> = (0..64).map(hash).collect();
        let mut bits = vec![0u64; 4096];
        for m in &members {
            reference_add(&mut bits, m);
        }
        let every = [
            HashForm::Compressed,
            HashForm::Uncompressed,
            HashForm::P2shP2wpkh,
        ];

        // A filter of `1...` and `bc1q...` addresses only: the P2SH-P2WPKH third of a
        // default sweep can never match it.
        let path = scratch("no-p2sh.bf");
        write_filter_as(
            &path,
            &bits,
            BitLayout::Scattered,
            Kind::P2pkh.bit() | Kind::P2wpkh.bit(),
        );
        let target = open(&path).unwrap();
        assert_eq!(target.unmatchable_forms(&every), vec![HashForm::P2shP2wpkh]);
        assert!(
            target
                .unmatchable_forms(&[HashForm::Compressed, HashForm::Uncompressed])
                .is_empty(),
            "the forms it does hold entries for must not be named"
        );

        // A taproot-only filter, which no form of this sweep reaches at all.
        let path = scratch("taproot-only.bf");
        write_filter_as(&path, &bits, BitLayout::Scattered, Kind::P2tr.bit());
        assert_eq!(open(&path).unwrap().unmatchable_forms(&every), every);

        // And what `bf-gen` writes by default, or wrote before it recorded anything, holds
        // something for every form -- so an ordinary sweep is never warned about.
        for kinds in [LEGACY_KINDS, Kind::Hex.bit()] {
            let path = scratch("complete.bf");
            write_filter_as(&path, &bits, BitLayout::Scattered, kinds);
            assert!(
                open(&path).unwrap().unmatchable_forms(&every).is_empty(),
                "a filter holding {kinds:#b} covers every form this sweep derives"
            );
        }
    }

    /// A companion that cannot be read stops the run, and a companion with nothing in it
    /// stops it too. Both would otherwise reject every candidate for the length of a
    /// multi-day sweep and report a clean, empty result -- and unlike the filter itself,
    /// this file was never named by the operator, so it cannot be taken as what they
    /// meant.
    #[test]
    fn an_unusable_companion_is_refused_rather_than_swept_with() {
        let members: Vec<[u8; 20]> = (0..64).map(hash).collect();
        let path = scratch("bad-companion.bf");
        let mut bits = vec![0u64; 4096];
        for m in &members {
            reference_add(&mut bits, m);
        }
        write_filter(&path, &bits);

        std::fs::write(verify_path(&path), vec![0u8; 64]).unwrap();
        let err = open(&path).unwrap_err().to_string();
        assert!(
            err.contains("verification filter"),
            "the refusal has to name what it could not open: {err}"
        );

        write_filter(&verify_path(&path), &vec![0u64; 4096]);
        let err = open(&path).unwrap_err().to_string();
        assert!(
            err.contains("almost no bits") && err.contains("keyscan bf-gen"),
            "an empty companion has to say what it would do and how to fix it: {err}"
        );

        // And with the file gone the same filter sweeps as it always did.
        std::fs::remove_file(verify_path(&path)).unwrap();
        assert!(open(&path).unwrap().verify().is_none());
    }

    /// The real cross-check: `keyscan bf-gen` wrote the companion, so every hash160 the
    /// filter was built from has to clear *both* filters. This is what pins
    /// [`crate::target::bloom::resalt`] to the transform that actually produced the file --
    /// nothing in the format records it, and a mismatch reads as a filter holding
    /// nothing.
    #[test]
    fn finds_every_hash_the_real_pair_was_built_from() {
        use std::io::{BufRead, BufReader};

        let (Some(real_filter), Some(source)) = (
            env_path("KEYFORGE_FILTER"),
            env_path("KEYFORGE_FILTER_SOURCE"),
        ) else {
            eprintln!("skipping: set KEYFORGE_FILTER and KEYFORGE_FILTER_SOURCE to run this");
            return;
        };
        if !real_filter.exists() || !source.exists() {
            eprintln!("skipping: real filter or source hash list not present");
            return;
        }
        if !verify_path(&real_filter).exists() {
            eprintln!(
                "skipping: no verification filter beside {}",
                real_filter.display()
            );
            return;
        }

        let target = open(&real_filter).unwrap();
        assert!(target.verify().is_some(), "the companion must have loaded");

        let reader = BufReader::new(std::fs::File::open(&source).unwrap());
        let mut checked = 0;
        for line in reader.lines().take(200_000) {
            let line = line.unwrap();
            let line = line.trim();
            if line.len() != 40 {
                continue;
            }
            let mut bytes = [0u8; 20];
            hex::decode_to_slice(line, &mut bytes).unwrap();
            assert_eq!(
                target.screen(&bytes),
                Verdict::Present,
                "an entry of the address list was rejected: {line}"
            );
            checked += 1;
        }
        assert!(
            checked > 100_000,
            "expected to check many hashes, got {checked}"
        );
    }
}
