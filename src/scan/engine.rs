//! Dividing a search space across threads, and surviving being stopped.
//!
//! The unit of work is a **block** of contiguous points, claimed by an atomic
//! `fetch_add`. Everything interesting here is about one property: a scan that is
//! interrupted and resumed must not skip a point. Repeating a little work is fine;
//! silently leaving a hole in a sweep is not, because the whole output of a clean pass
//! is the claim that there was nothing there.
//!
//! Three details carry that:
//!
//! * **An abandoned block is not a completed block.** A worker that notices the stop
//!   flag mid-block leaves without recording it, so the watermark stays below it and a
//!   resumed run walks it again from the start.
//! * **The watermark is contiguous.** Completed blocks go into a `BTreeSet` and are
//!   absorbed into a running mark as they become contiguous, so the set only ever holds
//!   the ragged edge of in-flight work -- at most one entry per thread. Checkpointing a
//!   high block number while a lower one was still running is exactly how a hole gets
//!   written to disk.
//! * **Block size comes from the range, never from the thread count.** Otherwise
//!   changing `-t` between runs would renumber the blocks and invalidate a checkpoint
//!   that is still perfectly good.
//!
//! The stop flag is checked per *batch* rather than per block. A block is thousands of
//! points, which is tens of seconds of work at a wide scope, and waiting that long to
//! acknowledge Ctrl-C reads as a hang.

use crate::scan::derive::{self, Deriver, Location, Scope};
use crate::scan::sink::{Hit, MatchSink};
use crate::scan::state::State;
use crate::target::Target;
use crate::ui::{self, Ui};
use crate::vuln::{Point, Vulnerability};
use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Everything a scan needs that is not derived from the vulnerability itself.
pub struct ScanConfig {
    pub filter: PathBuf,
    pub out: PathBuf,
    pub details: Option<PathBuf>,
    pub start: u128,
    pub end: u128,
    pub threads: usize,
    pub block: Option<u64>,
    pub restart: bool,
}

/// What a finished or interrupted scan reports back.
pub struct ScanReport {
    pub points_done: u64,
    pub total_points: u128,
    pub candidates: u64,
    pub locations: u64,
    pub elapsed: Duration,
    pub finished: bool,
}

/// The scan's visitor: probe the filter for a whole chunk, record what survives.
struct Candidates<'a> {
    target: &'a Target,
    scope: &'a Scope,
    sink: &'a Mutex<MatchSink<'a>>,
    found: &'a AtomicBool,
    vuln: &'static str,
    /// The first point of the batch being walked, so `batch_index` resolves back to the
    /// point that produced it. Set before each batch.
    base: u128,
    /// Which material of the point's expansion this batch came from. A point does not
    /// name a wallet on its own when a vulnerability expands it to several streams.
    stream: usize,
    /// The batch's materials, so a hit can be turned into a private key without the walk
    /// having carried one through every chunk.
    ///
    /// Held by value rather than borrowed: a borrow would tie the batch buffer's
    /// lifetime to the sink's, and 128 bytes copied once per four points is nothing
    /// against the ~1,400 probes those points cost.
    materials: [[u8; 32]; derive::POINTS_PER_BATCH],
}

impl derive::Visitor for Candidates<'_> {
    fn select(&mut self, hashes: &[[u8; 20]], hits: &mut Vec<u32>) {
        self.target.contains_batch(hashes, hits);
    }

    fn visit(&mut self, location: &Location, hash: &[u8; 20], phrase: &str) {
        let material = &self.materials[location.batch_index as usize];
        let hit = Hit {
            vuln: self.vuln,
            point: self.base + location.batch_index as u128,
            stream: self.stream,
            material: &material[..location.material_len as usize],
            location,
            hash,
            phrase,
        };
        self.sink.lock().unwrap().record(&hit, self.scope);
        self.found.store(true, Ordering::Relaxed);
    }
}

/// Block size for a range, when the user has not chosen one.
///
/// Derived from the range alone so that changing the thread count does not invalidate a
/// checkpoint. Clamped so a tiny range still gets more than one block (otherwise a
/// short run is single-threaded) and a huge one does not get blocks so large that
/// Ctrl-C loses minutes of work.
pub fn default_block(total_points: u128) -> u64 {
    ((total_points / 256).max(1)).min(4096) as u64
}

/// Walk `[start, end)` of a vulnerability's space, recording filter matches.
#[allow(clippy::too_many_arguments)]
pub fn run(
    ui: &Ui,
    vuln: &'static dyn Vulnerability,
    scope: &Scope,
    target: &Target,
    config: &ScanConfig,
    fingerprint: &str,
    state_path: &Path,
) -> Result<ScanReport> {
    let total_points = config.end.saturating_sub(config.start);
    let block = config.block.unwrap_or_else(|| default_block(total_points));
    let total_blocks = total_points.div_ceil(block as u128) as u64;

    // A checkpoint is only resumed when it describes this exact scan. A fingerprint
    // mismatch means the flags changed, and continuing would produce a file that looks
    // like a complete sweep of neither configuration.
    let resume_blocks = match (config.restart, State::load(state_path)?) {
        (false, Some(state)) if state.fingerprint == fingerprint => {
            let done = state.blocks_done;
            if done > 0 {
                ui.notice(
                    "resuming",
                    &format!(
                        "{} of {} blocks already walked",
                        ui::commas(done),
                        ui::commas(total_blocks)
                    ),
                );
            }
            done
        }
        (false, Some(_)) => {
            ui.warn(
                "the checkpoint beside this filter describes a different scan; \
                 starting over. Pass --restart to silence this.",
            );
            0
        }
        _ => 0,
    };

    if resume_blocks >= total_blocks {
        ui.notice("nothing to do", "this range has already been walked");
        return Ok(ScanReport {
            points_done: 0,
            total_points,
            candidates: 0,
            locations: 0,
            elapsed: Duration::ZERO,
            finished: true,
        });
    }

    let sink = Mutex::new(MatchSink::open(ui, &config.out, config.details.as_deref())?);
    let next_block = AtomicU64::new(resume_blocks);
    let completed: Mutex<BTreeSet<u64>> = Mutex::new(BTreeSet::new());
    let watermark = AtomicU64::new(resume_blocks);
    let points_done = AtomicU64::new(0);
    let stop = Arc::new(AtomicBool::new(false));
    // Set by a worker that found something, cleared by the monitor when it acts on it. A
    // find repaints the progress bar underneath itself, and without this the bar would
    // be redrawn with the count from the last tick -- reading "0 candidates" directly
    // below the candidate that had just been announced.
    let found = AtomicBool::new(false);

    install_interrupt(ui, &stop)?;

    let began = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..config.threads {
            let (target, scope, sink, stop, found) = (target, scope, &sink, &stop, &found);
            let (next_block, completed, watermark, points_done) =
                (&next_block, &completed, &watermark, &points_done);
            s.spawn(move || {
                let mut deriver = Deriver::new();
                let mut expanded: Vec<[u8; 32]> = Vec::new();
                let mut batch: Vec<[u8; 32]> = Vec::with_capacity(derive::POINTS_PER_BATCH);

                loop {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let unit = next_block.fetch_add(1, Ordering::Relaxed);
                    if unit >= total_blocks {
                        break;
                    }

                    let lo = config.start + unit as u128 * block as u128;
                    let hi = (lo + block as u128).min(config.end);
                    let mut interrupted = false;

                    // How many materials this vulnerability expands a point to. Asked
                    // once: it is a property of the vulnerability, not of the point.
                    expanded.clear();
                    vuln.expand(Point::Integer(lo), &mut expanded);
                    let streams = expanded.len().max(1);

                    // One pass per stream, both inside this block, so a block is still
                    // complete or not -- never half of one stream, which is what the
                    // watermark and the checkpoint mean.
                    for stream in 0..streams {
                        if interrupted {
                            break;
                        }
                        let mut candidates = Candidates {
                            target,
                            scope,
                            sink,
                            found,
                            vuln: vuln.id(),
                            base: lo,
                            stream,
                            materials: [[0u8; 32]; derive::POINTS_PER_BATCH],
                        };

                        let mut point = lo;
                        while point < hi {
                            // Checked per batch rather than per block: a batch is a few
                            // points, i.e. milliseconds, where a block can be a minute.
                            if stop.load(Ordering::Relaxed) {
                                interrupted = true;
                                break;
                            }

                            batch.clear();
                            let base = point;
                            while point < hi && batch.len() < derive::POINTS_PER_BATCH {
                                expanded.clear();
                                vuln.expand(Point::Integer(point), &mut expanded);
                                match expanded.get(stream) {
                                    Some(material) => batch.push(*material),
                                    // A vulnerability that expands some points to fewer
                                    // materials than others. None do today, and skipping
                                    // is the only safe response if one ever does.
                                    None => {}
                                }
                                point += 1;
                            }
                            if batch.is_empty() {
                                continue;
                            }

                            candidates.base = base;
                            candidates.materials[..batch.len()].copy_from_slice(&batch);
                            deriver.walk_batch(&batch, scope, &mut candidates);

                            // Credited per batch, not per block: a block is minutes of
                            // work at a wide scope, and crediting it only on completion
                            // leaves the progress line reading zero through the first one.
                            if stream == 0 {
                                points_done.fetch_add(batch.len() as u64, Ordering::Relaxed);
                            }
                        }
                    }

                    // An abandoned block is not a finished block. Leaving it out of the
                    // completed set holds the watermark below it, so a resumed run
                    // rescans it rather than skipping the part never walked.
                    if interrupted {
                        break;
                    }

                    // Advance the contiguous watermark. Completed blocks are removed as
                    // they are absorbed, so the set only holds the ragged edge.
                    let mut done = completed.lock().unwrap();
                    done.insert(unit);
                    let mut mark = watermark.load(Ordering::Relaxed);
                    while done.remove(&mark) {
                        mark += 1;
                    }
                    watermark.store(mark, Ordering::Relaxed);
                }
            });
        }

        // Progress and checkpointing share one thread: both are periodic reads of the
        // same counters.
        let (sink, stop, found) = (&sink, &stop, &found);
        let (watermark, points_done) = (&watermark, &points_done);
        s.spawn(move || {
            let interval = if ui.interactive() {
                Duration::from_secs(2)
            } else {
                // Redirected output gets a line a minute rather than one every two
                // seconds, so a log of a multi-day sweep stays readable.
                Duration::from_secs(60)
            };
            let mut last_checkpoint = Instant::now();
            let mut announced_stop = false;

            loop {
                // Sleep in slices so an interrupt is acted on promptly.
                for _ in 0..(interval.as_millis() / 100) {
                    std::thread::sleep(Duration::from_millis(100));
                    if stop.load(Ordering::Relaxed)
                        || watermark.load(Ordering::Relaxed) >= total_blocks
                        || found.swap(false, Ordering::Relaxed)
                    {
                        break;
                    }
                }

                // Said here rather than in the signal handler, so it goes through the
                // same writer as everything else and the progress bar survives it.
                if stop.load(Ordering::Relaxed) && !announced_stop {
                    announced_stop = true;
                    ui.notice("interrupted", "finishing the batch in flight and checkpointing");
                }

                let finished = watermark.load(Ordering::Relaxed) >= total_blocks;
                if !finished {
                    let done = points_done.load(Ordering::Relaxed);
                    let candidates = sink.lock().unwrap().candidates;
                    let rate = done as f64 / began.elapsed().as_secs_f64().max(1e-9);
                    let remaining = (total_points as f64 - done as f64).max(0.0);
                    ui.progress(
                        "scanning",
                        done,
                        total_points.min(u64::MAX as u128) as u64,
                        "points",
                        &[
                            format!("{:.0}/s", rate),
                            format!("eta {}", ui::duration(remaining / rate.max(1e-9))),
                            format!("{} candidates", ui::commas(candidates)),
                        ],
                    );
                }

                // Not on the last pass: the caller settles the file once the workers
                // have joined, and a finished range clears it rather than writing it.
                if last_checkpoint.elapsed() >= Duration::from_secs(10) && !finished {
                    let mark = watermark.load(Ordering::Relaxed);
                    if let Err(e) = save(state_path, fingerprint, config, block, mark) {
                        ui.warn(&format!("could not write the checkpoint: {e:#}"));
                    }
                    last_checkpoint = Instant::now();
                }
                if finished || stop.load(Ordering::Relaxed) {
                    break;
                }
            }
        });
    });

    // A finished range has nothing to resume, so the checkpoint goes rather than being
    // left behind saying so. An interrupted one keeps it: that file is the whole of what
    // a re-run needs to pick up where this stopped.
    let mark = watermark.load(Ordering::Relaxed);
    let finished = mark >= total_blocks;
    if finished {
        State::clear(state_path)?;
    } else {
        save(state_path, fingerprint, config, block, mark)?;
    }

    ui.clear();
    let sink = sink.lock().unwrap();
    Ok(ScanReport {
        points_done: points_done.load(Ordering::Relaxed),
        total_points,
        candidates: sink.candidates,
        locations: sink.locations,
        elapsed: began.elapsed(),
        finished,
    })
}

fn save(
    path: &Path,
    fingerprint: &str,
    config: &ScanConfig,
    block: u64,
    blocks_done: u64,
) -> Result<()> {
    State {
        fingerprint: fingerprint.to_string(),
        // The state file is a u64 format inherited from a tool whose space was 2^32.
        // Saturating is safe here because a space wider than u64 cannot be walked in any
        // case -- see `java-random`, whose guide says as much.
        start: config.start.min(u64::MAX as u128) as u64,
        end: config.end.min(u64::MAX as u128) as u64,
        block,
        blocks_done,
    }
    .save(path)
}

/// Install the Ctrl-C handler.
///
/// The handler only sets a flag. Announcing the interrupt is left to the monitor thread,
/// which picks it up within 100ms: it owns the progress bar and can put it back
/// afterwards, and printing from a signal handler means taking the stderr lock from a
/// context that may already hold it. A second Ctrl-C kills the process outright rather
/// than being swallowed.
fn install_interrupt(ui: &Ui, stop: &Arc<AtomicBool>) -> Result<()> {
    let stop = Arc::clone(stop);
    let armed = AtomicBool::new(false);
    // `process::exit` runs no destructors, so `Drop for Ui` never gets to put the cursor
    // back on that path. Whether it was hidden is captured here rather than read through
    // the `Ui`, which the handler outlives.
    let hid_cursor = ui.interactive();
    ctrlc::set_handler(move || {
        if armed.swap(true, Ordering::SeqCst) {
            if hid_cursor {
                ui::show_cursor();
            }
            std::process::exit(130);
        }
        stop.store(true, Ordering::SeqCst);
    })
    .context("installing the interrupt handler")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Block size must come from the range and nothing else, so that changing `-t`
    /// between runs never invalidates a checkpoint that is still good.
    #[test]
    fn block_size_depends_only_on_the_range() {
        assert_eq!(default_block(1 << 32), 4096);
        assert_eq!(default_block(256), 1);
        // Never zero, or the block count would divide by it.
        assert_eq!(default_block(1), 1);
        assert_eq!(default_block(0), 1);
        // Small ranges still get several blocks, so a short run is not single-threaded.
        assert!(default_block(100_000) < 100_000);
    }

    /// Every point of a range must fall in exactly one block, with none dropped off the
    /// end. `div_ceil` is what makes the last, short block still exist.
    #[test]
    fn blocks_cover_the_whole_range_exactly_once() {
        for (start, end) in [(0u128, 1000u128), (5, 5), (0, 1), (17, 4096), (1, 1 << 20)] {
            let total = end - start;
            let block = default_block(total);
            let blocks = total.div_ceil(block as u128) as u64;

            let mut covered = 0u128;
            let mut last_hi = start;
            for unit in 0..blocks {
                let lo = start + unit as u128 * block as u128;
                let hi = (lo + block as u128).min(end);
                assert_eq!(lo, last_hi, "block {unit} does not start where the last ended");
                assert!(hi > lo, "block {unit} is empty");
                covered += hi - lo;
                last_hi = hi;
            }
            assert_eq!(last_hi, end, "the blocks stop short of the end");
            assert_eq!(covered, total, "the blocks do not cover the range exactly");
        }
    }

    /// A whole scan, end to end, against a filter holding a real stolen wallet.
    ///
    /// Seed 500 of the Milk Sad stream produced `13Kqxkrms...` at `m/44'/0'/0'/0/0` --
    /// one of the four canary wallets the researchers funded and watched get emptied, so
    /// the (seed, path, address) triple comes from `bx` itself and from the chain rather
    /// than from anything in this repository.
    ///
    /// This exercises the parts nothing else does: the block queue, the visitor wiring,
    /// the filter probe, the watermark, and the promise that `matches.txt` holds exactly
    /// one importable secret per line.
    #[test]
    fn scans_a_range_and_writes_the_canary_wallet() {
        use crate::scan::derive::Route;
        use crate::scan::sink::is_importable_secret;
        use crate::target::bloom::testing::{reference_add, scratch, write_filter};
        use crate::wallet::address::HashForm;
        use crate::wallet::path::PathSpec;

        // A filter holding just the canary's hash160.
        let canary = {
            // Derive it rather than hard-coding the hash160: the address is the
            // published fact, and `address::encode` is already held to the reference.
            let scope = Scope {
                material_sizes: vec![32],
                routes: vec![Route::Bip39],
                paths: vec![PathSpec::parse("m/44'/0'/0'/0/0").unwrap()],
                forms: vec![HashForm::Compressed],
            };
            let mut found = None;
            Deriver::new().walk_batch(
                &[crate::vuln::mt19937::entropy_for_seed(500)],
                &scope,
                &mut crate::scan::derive::All(|_l: &Location, h: &[u8; 20], _: &str| {
                    found = Some(*h);
                }),
            );
            found.expect("the canary derives")
        };
        assert_eq!(
            crate::wallet::address::encode(HashForm::Compressed, &canary)
                .split(' ')
                .next(),
            Some("13KqxkrmsPKy8gyYwochCQTuPHC7Lp8bFU"),
            "this is not the published canary address"
        );
        let mut bits = vec![0u64; 1 << 16];
        reference_add(&mut bits, &canary);
        let filter_path = scratch("engine-canary.bf");
        write_filter(&filter_path, &bits);

        let out = scratch("engine-canary-matches.txt");
        let _ = std::fs::remove_file(&out);
        let state = scratch("engine-canary.state");
        let _ = std::fs::remove_file(&state);

        let ui = Ui::new(crate::ui::Stream::Stderr);
        let target = Target::new(
            crate::target::bloom::BloomFilter::open(&filter_path).unwrap(),
            None,
        );
        let scope = Scope {
            material_sizes: vec![32],
            routes: vec![Route::Bip39],
            paths: vec![PathSpec::parse("m/44'/0'/0'/0/0").unwrap()],
            forms: vec![HashForm::Compressed],
        };
        let config = ScanConfig {
            filter: filter_path.clone(),
            out: out.clone(),
            details: None,
            start: 490,
            end: 510,
            threads: 2,
            block: Some(4),
            restart: true,
        };

        let report = run(
            &ui,
            &crate::vuln::mt19937::MilkSad,
            &scope,
            &target,
            &config,
            "test-fingerprint",
            &state,
        )
        .expect("the scan runs");

        assert!(report.finished, "the scan did not finish its range");
        assert_eq!(report.points_done, 20, "the scan walked the wrong number of points");
        assert_eq!(report.candidates, 1, "expected exactly the canary wallet");

        // The output contract: one importable secret per line, and nothing else.
        let text = std::fs::read_to_string(&out).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 1);
        for line in &lines {
            assert!(is_importable_secret(line), "not an importable secret: {line:?}");
        }
        // And it is the right wallet: the 24-word phrase for seed 500.
        let want = crate::wallet::bip39::mnemonic(
            &crate::vuln::mt19937::entropy_for_seed(500)[..32],
        );
        assert_eq!(lines[0], want);

        // A finished range leaves no checkpoint behind to resume.
        assert!(!state.exists(), "a finished scan left a checkpoint");
    }

    /// The contiguous watermark must never pass a block that is still in flight, which
    /// is the property that keeps a checkpoint from describing a hole.
    #[test]
    fn the_watermark_never_passes_an_unfinished_block() {
        let mut done: BTreeSet<u64> = BTreeSet::new();
        let mut mark = 0u64;

        // Blocks 1, 2 and 4 finish before block 0 does -- the usual case with several
        // threads. The mark must stay at 0.
        for unit in [1u64, 2, 4] {
            done.insert(unit);
            while done.remove(&mark) {
                mark += 1;
            }
            assert_eq!(mark, 0, "the mark advanced past block 0, which is still running");
        }

        // Block 0 lands: the mark absorbs 0, 1, 2 and stops at 3, which nobody has
        // finished. Block 4 stays in the set as the ragged edge.
        done.insert(0);
        while done.remove(&mark) {
            mark += 1;
        }
        assert_eq!(mark, 3);
        assert_eq!(done.iter().copied().collect::<Vec<_>>(), vec![4]);
    }
}
