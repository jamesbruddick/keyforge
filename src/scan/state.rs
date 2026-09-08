//! The scan checkpoint.
//!
//! A full sweep at the default scope runs for days, so it has to survive Ctrl-C,
//! crashes and reboots. The file records how many contiguous blocks from the start
//! of the range are finished, plus a fingerprint of the run's settings -- resuming
//! with a different scope or range would leave a silent hole in the coverage, which
//! for this kind of search is worse than starting over.
//!
//! Plain text on purpose: it is meant to be readable with `cat` mid-run.

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::Path;

const HEADER: &str = "keyforge state v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State {
    pub fingerprint: String,
    pub start: u64,
    pub end: u64,
    pub block: u64,
    /// Contiguous completed blocks from `start`. Everything below this seed has been
    /// scanned; blocks finished out of order are not counted until the gap fills.
    pub blocks_done: u64,
}

impl State {
    /// A stable digest of whatever identifies a run.
    pub fn fingerprint(parts: &[&str]) -> String {
        let mut hasher = Sha256::new();
        for part in parts {
            hasher.update(part.as_bytes());
            hasher.update(b"\x1f");
        }
        let digest = hasher.finalize();
        digest[..16].iter().map(|b| format!("{b:02x}")).collect()
    }

    /// The first seed a resumed run should scan.
    pub fn next_seed(&self) -> u64 {
        (self.start + self.blocks_done * self.block).min(self.end)
    }

    pub fn load(path: &Path) -> Result<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading checkpoint {}", path.display()))?;

        let mut lines = text.lines();
        if lines.next() != Some(HEADER) {
            bail!(
                "{} is not a keyforge checkpoint (expected {HEADER:?} on the first line)",
                path.display()
            );
        }

        let mut fingerprint = None;
        let (mut start, mut end, mut block, mut blocks_done) = (None, None, None, None);
        for line in lines {
            let Some((key, value)) = line.split_once(' ') else {
                continue;
            };
            match key {
                "config" => fingerprint = Some(value.to_string()),
                "start" => start = Some(value.parse()?),
                "end" => end = Some(value.parse()?),
                "block" => block = Some(value.parse()?),
                "blocks_done" => blocks_done = Some(value.parse()?),
                _ => {}
            }
        }

        let missing = || anyhow::anyhow!("{} is missing required fields", path.display());
        Ok(Some(Self {
            fingerprint: fingerprint.ok_or_else(missing)?,
            start: start.ok_or_else(missing)?,
            end: end.ok_or_else(missing)?,
            block: block.ok_or_else(missing)?,
            blocks_done: blocks_done.ok_or_else(missing)?,
        }))
    }

    /// Remove the checkpoint, which is what finishing the range means: there is
    /// nothing left to resume, and a file that says so is just litter next to the
    /// matches. Absent is already the desired state, so it is not an error.
    pub fn clear(path: &Path) -> Result<()> {
        match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => {
                Err(anyhow::Error::new(e)
                    .context(format!("removing checkpoint {}", path.display())))
            }
        }
    }

    /// Write atomically: a checkpoint truncated by a crash mid-write would be worse
    /// than no checkpoint at all, since it reads as a corrupt file and blocks resume.
    pub fn save(&self, path: &Path) -> Result<()> {
        let body = format!(
            "{HEADER}\nconfig {}\nstart {}\nend {}\nblock {}\nblocks_done {}\nnext_seed {}\n",
            self.fingerprint,
            self.start,
            self.end,
            self.block,
            self.blocks_done,
            self.next_seed(),
        );

        // Appended rather than `with_extension`, which would rewrite whatever
        // extension the caller's output path already had rather than adding one.
        let mut temp = path.as_os_str().to_os_string();
        temp.push(".tmp");
        let temp = std::path::PathBuf::from(temp);
        {
            let mut file = fs::File::create(&temp)
                .with_context(|| format!("writing checkpoint {}", temp.display()))?;
            file.write_all(body.as_bytes())?;
            file.sync_data()?;
        }
        fs::rename(&temp, path)
            .with_context(|| format!("installing checkpoint {}", path.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("keyforge-tests");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn sample() -> State {
        State {
            fingerprint: State::fingerprint(&["filter.bf", "0", "1000"]),
            start: 100,
            end: 10_000,
            block: 4096,
            blocks_done: 2,
        }
    }

    #[test]
    fn round_trips_through_the_file() {
        let path = scratch("round-trip.state");
        let state = sample();
        state.save(&path).unwrap();
        assert_eq!(State::load(&path).unwrap().unwrap(), state);
    }

    #[test]
    fn reports_no_state_when_the_file_is_absent() {
        let path = scratch("definitely-absent.state");
        let _ = std::fs::remove_file(&path);
        assert!(State::load(&path).unwrap().is_none());
    }

    /// Clearing is what completion does, and a completed range may well be
    /// completed twice -- a second run of the same command finds no file and must
    /// not fail on that.
    #[test]
    fn clearing_removes_the_file_and_tolerates_its_absence() {
        let path = scratch("cleared.state");
        sample().save(&path).unwrap();
        State::clear(&path).unwrap();
        assert!(!path.exists());
        State::clear(&path).unwrap();
    }

    #[test]
    fn rejects_files_it_did_not_write() {
        let path = scratch("foreign.state");
        std::fs::write(&path, "some other tool's file\n").unwrap();
        assert!(State::load(&path).is_err());
    }

    #[test]
    fn next_seed_follows_completed_blocks_and_is_clamped_to_the_end() {
        let state = sample();
        assert_eq!(state.next_seed(), 100 + 2 * 4096);

        let overrun = State {
            blocks_done: 1_000_000,
            ..sample()
        };
        assert_eq!(overrun.next_seed(), overrun.end);
    }

    #[test]
    fn the_fingerprint_separates_fields_unambiguously() {
        // Without a separator, ("ab", "c") and ("a", "bc") would collide -- which
        // would let a differently-scoped run resume another's checkpoint.
        assert_ne!(
            State::fingerprint(&["ab", "c"]),
            State::fingerprint(&["a", "bc"])
        );
        assert_eq!(
            State::fingerprint(&["a", "b"]),
            State::fingerprint(&["a", "b"])
        );
    }

    #[test]
    fn saving_leaves_no_temporary_file_behind() {
        // A name with its own extension, since the scanner's state path is the
        // output path with `.state` appended -- `matches.txt.state`.
        let path = scratch("matches.txt.state");
        sample().save(&path).unwrap();
        assert!(!scratch("matches.txt.state.tmp").exists());
        assert!(!scratch("matches.txt.tmp").exists());
        assert_eq!(State::load(&path).unwrap().unwrap(), sample());
    }
}
