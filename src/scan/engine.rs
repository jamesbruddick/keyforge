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
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Once, OnceLock};
use std::time::{Duration, Instant};

/// Everything a scan needs that is not derived from the vulnerability itself.
pub struct ScanConfig {
    pub filter: PathBuf,
    /// The passphrase corpus, for a vulnerability whose space is a file rather than a
    /// number range.
    pub corpus: Option<PathBuf>,
    pub out: PathBuf,
    pub details: Option<PathBuf>,
    pub start: u128,
    pub end: u128,
    pub threads: usize,
    pub block: Option<u64>,
    pub restart: bool,
    /// Whether to use a device, and whether the CPU walks points of its own alongside it.
    pub gpu: Option<crate::gpu::Mode>,
    /// Points per device launch. `None` sizes it from the device's memory.
    pub gpu_batch: Option<usize>,
}

/// What a finished or interrupted scan reports back.
pub struct ScanReport {
    pub points_done: u64,
    /// Byte streams walked per point. A point costs this many walks of the pipeline, and
    /// naming it is what stops a rate being compared against one counted per walk.
    pub streams: usize,
    /// Where a launch's time went, per kernel, when `KEYFORGE_GPU_PROFILE` is set.
    pub gpu_profile: Option<String>,
    /// Device records the CPU could not reproduce.
    ///
    /// Should be zero, always. The device is a filter and the CPU is the oracle: a record
    /// says "look at this point", the host re-derives it, and only what the *CPU* derives
    /// reaches `matches.txt`. So a wrong kernel cannot write a wrong secret -- but it can
    /// silently miss wallets, and that has no symptom at all. This counter is the symptom:
    /// a launch producing records the CPU cannot reproduce means the two implementations
    /// have diverged, and zero across a multi-day sweep is a continuous free check on the
    /// whole kernel set.
    pub unconfirmed: u64,
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
            streams: 1,
            gpu_profile: None,
            unconfirmed: 0,
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
    let stop = install_interrupt(ui)?;
    // Set by a worker that found something, cleared by the monitor when it acts on it. A
    // find repaints the progress bar underneath itself, and without this the bar would
    // be redrawn with the count from the last tick -- reading "0 candidates" directly
    // below the candidate that had just been announced.
    let found = AtomicBool::new(false);
    let unconfirmed = AtomicU64::new(0);

    // A device claims blocks from the same queue the CPU workers do, so `--gpu both` needs
    // no separate range and no second checkpoint: whichever finishes a block first takes
    // the next one.
    let mut device = match config.gpu {
        Some(_) => Some(open_device(ui, vuln, scope, config, target, total_points)?),
        None => None,
    };
    let cpu_workers = match config.gpu {
        Some(crate::gpu::Mode::Only) => 0,
        _ => config.threads,
    };

    let began = Instant::now();
    std::thread::scope(|s| {
        if let Some(gpu) = device.as_mut() {
            let (target, scope, sink, stop, found) = (target, scope, &sink, &stop, &found);
            let (next_block, completed, watermark, points_done, unconfirmed) =
                (&next_block, &completed, &watermark, &points_done, &unconfirmed);
            s.spawn(move || {
                let result = run_device(
                    gpu, vuln, target, scope, sink, found, stop, next_block, completed,
                    watermark, points_done, unconfirmed, config, block, total_blocks,
                );
                if let Err(error) = result {
                    // A dead device must not look like a finished sweep. Stopping every
                    // other worker is what keeps the checkpoint honest: the watermark
                    // holds below the blocks this worker had claimed, so a resumed run
                    // rescans them.
                    stop.store(true, Ordering::SeqCst);
                    eprintln!("\ndevice worker failed: {error:#}");
                }
            });
        }

        for _ in 0..cpu_workers {
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
        streams: streams_of(vuln, config.start),
        gpu_profile: device.as_mut().and_then(|g| g.report()),
        unconfirmed: unconfirmed.load(Ordering::Relaxed),
        total_points,
        candidates: sink.candidates,
        locations: sink.locations,
        elapsed: began.elapsed(),
        finished,
    })
}

/// How many byte streams a vulnerability expands one point to.
///
/// Each is a separate walk of the whole pipeline, so a point costs this many. It is asked
/// once: it is a property of the vulnerability, not of the point.
pub fn streams_of(vuln: &dyn Vulnerability, at: u128) -> usize {
    let mut out = Vec::new();
    vuln.expand(Point::Integer(at), &mut out);
    out.len().max(1)
}

/// Open the device and put the filter on it.
///
/// Deliberately eager, and before any point is walked: a runtime-compiled kernel set can
/// only fail at runtime, and the moment to find out is during startup next to the filter
/// load rather than two hours into a detached sweep.
fn open_device(
    ui: &Ui,
    vuln: &'static dyn Vulnerability,
    scope: &Scope,
    config: &ScanConfig,
    target: &Target,
    total_points: u128,
) -> Result<crate::gpu::Gpu> {
    use crate::gpu::{Batch, Gpu};

    if vuln.kernel().is_none() {
        anyhow::bail!(
            "`{}` has no GPU kernel yet, so it runs on the CPU only. Drop --gpu to scan it.",
            vuln.id()
        );
    }
    let batch = match config.gpu_batch {
        Some(n) => Batch::Fixed(n),
        None => Batch::Auto {
            filter_bytes: target.primary().size_bytes() as u64,
            seeds_in_range: total_points.min(u64::MAX as u128) as u64,
        },
    };
    let mut gpu = Gpu::open(scope, vuln, batch)?;
    // The device block, as one row and its detail rather than five equal rows. What the
    // card is, and whether the CPU is walking points beside it, is the part read at a
    // glance; the compiler and the launch geometry are what a slow sweep is diagnosed
    // from, and they belong under it rather than beside it.
    ui.row(
        "device",
        &format!(
            "{}, {}",
            gpu.name(),
            match config.gpu {
                // `--gpu` alone defaults to `both`, which is worth confirming here rather
                // than leaving to the documentation.
                Some(crate::gpu::Mode::Only) => "with the CPU confirming what it finds",
                _ => "with the CPU walking points alongside",
            }
        ),
    );
    if let Some(compiler) = gpu.compiler() {
        ui.cont(&compiler);
    }
    // The launch size decides whether a device is fed or starved, and it is chosen rather
    // than given, so it has to be visible. A sweep running at a fraction of the expected
    // rate is nearly always this number being small.
    ui.cont(&format!(
        "{} points per launch{}, {} scratch",
        ui::commas(gpu.layout().capacity as u64),
        match config.gpu_batch {
            Some(_) => "",
            None => " (auto)",
        },
        ui::bytes(gpu.scratch_bytes() as u64),
    ));
    gpu.bind_filter(target.primary())?;
    Ok(gpu)
}

/// One run of blocks' worth of device records, waiting for the CPU to confirm them.
struct Claim {
    /// The blocks this covers, contiguous. All of them are marked complete together, once
    /// their records have been confirmed.
    units: std::ops::Range<u64>,
    base: u128,
    /// `(stream, point offset from `base`, hash)` for each record.
    hits: Vec<(usize, u32, [u8; 20])>,
}

/// Drive the device, and hand what it finds to the CPU to confirm.
///
/// The split is the whole design. At a real filter's false-positive rate a launch of
/// tens of thousands of points passes a few dozen, and re-deriving each of those on the
/// CPU costs milliseconds -- enough that doing it on this thread would leave the device
/// idle for a large part of every second. So records go over a **bounded** channel to
/// confirm workers, and the bound is what stops a slow CPU turning into unbounded memory.
///
/// A block is marked complete by whoever confirms it, never by this function. Marking it
/// here would let the watermark pass a block whose records had not been written yet, and a
/// checkpoint taken in that window would describe a hole.
#[allow(clippy::too_many_arguments)]
fn run_device<'a>(
    gpu: &mut crate::gpu::Gpu,
    vuln: &'static dyn Vulnerability,
    target: &'a Target,
    scope: &'a Scope,
    sink: &'a Mutex<MatchSink<'a>>,
    found: &'a AtomicBool,
    stop: &AtomicBool,
    next_block: &AtomicU64,
    completed: &Mutex<BTreeSet<u64>>,
    watermark: &AtomicU64,
    points_done: &AtomicU64,
    unconfirmed: &'a AtomicU64,
    config: &ScanConfig,
    block: u64,
    total_blocks: u64,
) -> Result<()> {
    let streams = streams_of(vuln, config.start);
    let capacity = gpu.layout().capacity;

    // How many blocks it takes to fill a launch.
    //
    // A block is sized for a CPU worker -- small enough that Ctrl-C does not lose much and
    // that threads stay balanced -- and a device wants thousands of points at once. Taking
    // one block per launch means launching a block's worth however large the device is,
    // which on a short range is a few hundred points and leaves the GPU almost idle: it
    // measured 297 points/s against the CPU's 946 before this existed. So the device
    // claims a contiguous *run* of blocks and launches across it.
    let blocks_per_claim = (capacity as u64).div_ceil(block).max(1);

    // Two claims in flight per confirmer.
    //
    // Depth is backpressure: too shallow and the launch loop blocks on `send` while a
    // confirmer is still walking the last claim, which idles the device for exactly as
    // long as a CPU re-derivation takes. Too deep and a queued claim is work already
    // walked but not yet complete, which is what a resumed run repeats. Two per worker
    // keeps them fed across a launch without letting the watermark lag far behind. A flat
    // depth of two -- which this had -- starves a device with four confirmers behind it.
    let workers = (std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) / 4)
        .clamp(1, 4);
    let (tx, rx) = std::sync::mpsc::sync_channel::<Claim>(workers * 2);
    let rx = Mutex::new(rx);

    std::thread::scope(|s| -> Result<()> {
        for _ in 0..workers {
            let (rx, sink, target, scope) = (&rx, sink, target, scope);
            s.spawn(move || {
                let mut deriver = Deriver::new();
                let mut expanded = Vec::new();
                loop {
                    let claim = {
                        let guard = rx.lock().unwrap();
                        match guard.recv() {
                            Ok(claim) => claim,
                            Err(_) => break,
                        }
                    };

                    // Per (stream, point): the same point on two streams is two different
                    // wallets and two separate walks.
                    let mut wanted: Vec<(usize, u32)> =
                        claim.hits.iter().map(|(s, p, _)| (*s, *p)).collect();
                    wanted.sort_unstable();
                    wanted.dedup();

                    for (stream, offset) in wanted {
                        let point = claim.base + offset as u128;
                        expanded.clear();
                        vuln.expand(Point::Integer(point), &mut expanded);
                        let Some(material) = expanded.get(stream).copied() else {
                            continue;
                        };

                        // Everything the CPU derives for this point, so each device record
                        // can be held against an exact hash rather than against "this
                        // point produced something".
                        let mut produced: Vec<[u8; 20]> = Vec::new();
                        let mut candidates = Candidates {
                            target,
                            scope,
                            sink,
                            found,
                            vuln: vuln.id(),
                            base: point,
                            stream,
                            materials: [material; derive::POINTS_PER_BATCH],
                        };
                        deriver.walk_batch(
                            &[material],
                            scope,
                            &mut Confirming { inner: &mut candidates, produced: &mut produced },
                        );
                        produced.sort_unstable();

                        for (_, _, hash) in claim
                            .hits
                            .iter()
                            .filter(|(s, p, _)| *s == stream && *p == offset)
                        {
                            if produced.binary_search(hash).is_err() {
                                unconfirmed.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }

                    // Only now are the blocks finished.
                    let mut done = completed.lock().unwrap();
                    for unit in claim.units {
                        done.insert(unit);
                    }
                    let mut mark = watermark.load(Ordering::Relaxed);
                    while done.remove(&mark) {
                        mark += 1;
                    }
                    watermark.store(mark, Ordering::Relaxed);
                }
            });
        }

        let result = (|| -> Result<()> {
            loop {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                let first = next_block.fetch_add(blocks_per_claim, Ordering::Relaxed);
                if first >= total_blocks {
                    break;
                }
                let last = (first + blocks_per_claim).min(total_blocks);
                let lo = config.start + first as u128 * block as u128;
                let hi = (config.start + last as u128 * block as u128).min(config.end);

                let mut hits = Vec::new();
                let mut interrupted = false;
                for stream in 0..streams {
                    let mut at = lo;
                    while at < hi {
                        if stop.load(Ordering::Relaxed) {
                            interrupted = true;
                            break;
                        }
                        let n = ((hi - at) as usize).min(capacity);
                        for hit in gpu.run(at, n, stream)? {
                            // The record's point is an offset within *its* launch; make it
                            // an offset within the block, which is what the base says.
                            let offset = (at - lo) as u32 + hit.point;
                            hits.push((stream, offset, hit.hash));
                        }
                        if stream == 0 {
                            points_done.fetch_add(n as u64, Ordering::Relaxed);
                        }
                        at += n as u128;
                    }
                    if interrupted {
                        break;
                    }
                }
                // An abandoned block is not a finished block, so it is not handed on to be
                // marked complete: the watermark holds below it and a resume rescans it.
                if interrupted {
                    break;
                }
                if tx.send(Claim { units: first..last, base: lo, hits }).is_err() {
                    break;
                }
            }
            Ok(())
        })();

        // Dropping the sender is what lets the confirm workers finish and the scope join.
        drop(tx);
        result
    })
}

/// Wraps the scan's visitor to also record everything the CPU derived, so a device record
/// can be checked against an exact hash.
struct Confirming<'a, 'b> {
    inner: &'a mut Candidates<'b>,
    produced: &'a mut Vec<[u8; 20]>,
}

impl derive::Visitor for Confirming<'_, '_> {
    fn select(&mut self, hashes: &[[u8; 20]], hits: &mut Vec<u32>) {
        self.produced.extend_from_slice(hashes);
        self.inner.select(hashes, hits);
    }

    fn visit(&mut self, location: &Location, hash: &[u8; 20], phrase: &str) {
        self.inner.visit(location, hash, phrase);
    }
}

/// Hands out blocks of corpus lines, in order, to whichever worker asks next.
///
/// A number range can be divided up front because any block can be computed from its
/// index. A file cannot: reaching line ten million means reading the nine million before
/// it. So the division happens here, behind one lock, and a worker claims the next block
/// rather than computing which block is its own.
///
/// The lock is held only for the read, which is a few thousand lines off a buffered
/// reader -- microseconds against the seconds those lines then take to derive. Blocks are
/// still numbered, so the watermark and the checkpoint mean exactly what they mean for a
/// number range.
struct Corpus {
    lines: std::io::Lines<BufReader<File>>,
    next_block: u64,
    exhausted: bool,
}

impl Corpus {
    /// Open a corpus, skipping the blocks a checkpoint says are already done.
    fn open(path: &Path, block: u64, skip_blocks: u64) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("opening the corpus {}", path.display()))?;
        let mut lines = BufReader::new(file).lines();

        // Resuming means reading past what was already walked. There is no cheaper way
        // into the middle of a text file, and it is still far cheaper than deriving those
        // lines again.
        for _ in 0..skip_blocks.saturating_mul(block) {
            if lines.next().is_none() {
                break;
            }
        }
        Ok(Self { lines, next_block: skip_blocks, exhausted: false })
    }

    /// The next block of passphrases, or `None` once the file runs out.
    ///
    /// Blank lines are skipped rather than hashed: a trailing newline is not a passphrase,
    /// and `sha256("")` is a valid private key that would be reported against every corpus
    /// that happened to end with one.
    fn claim(&mut self, block: u64, out: &mut Vec<String>) -> Option<u64> {
        out.clear();
        if self.exhausted {
            return None;
        }
        while (out.len() as u64) < block {
            match self.lines.next() {
                Some(Ok(line)) => {
                    if !line.trim().is_empty() {
                        out.push(line);
                    }
                }
                Some(Err(_)) | None => {
                    self.exhausted = true;
                    break;
                }
            }
        }
        if out.is_empty() {
            return None;
        }
        let unit = self.next_block;
        self.next_block += 1;
        Some(unit)
    }
}

/// A corpus's identity, for the checkpoint.
///
/// The full contents, hashed. A resume that read a *different* file from the same path
/// would skip lines it never walked and walk lines it already had, and neither shows up
/// as an error -- so the file itself is what the fingerprint commits to, not its name.
/// One pass over a wordlist is seconds; one pass over a sweep is hours.
pub fn corpus_fingerprint(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let file = File::open(path)
        .with_context(|| format!("reading the corpus {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    // Streamed in chunks rather than read whole: a password dump can be larger than RAM,
    // and the point of hashing it is to catch it changing, not to hold it.
    loop {
        let chunk = reader.fill_buf().context("reading the corpus")?;
        if chunk.is_empty() {
            break;
        }
        hasher.update(chunk);
        let n = chunk.len();
        reader.consume(n);
    }
    Ok(ui::hex(&hasher.finalize()[..8]))
}

/// Walk a passphrase corpus, recording filter matches.
///
/// Shares the sink, the watermark and the checkpoint with [`run`]; what differs is only
/// that blocks come from a file in order rather than from arithmetic.
pub fn run_corpus(
    ui: &Ui,
    vuln: &'static dyn Vulnerability,
    scope: &Scope,
    target: &Target,
    config: &ScanConfig,
    fingerprint: &str,
    state_path: &Path,
) -> Result<ScanReport> {
    let corpus_path = config
        .corpus
        .as_ref()
        .context("this vulnerability needs --corpus")?;
    let block = config.block.unwrap_or(4096);

    let resume_blocks = match (config.restart, State::load(state_path)?) {
        (false, Some(state)) if state.fingerprint == fingerprint => {
            if state.blocks_done > 0 {
                ui.notice(
                    "resuming",
                    &format!(
                        "{} passphrases already walked",
                        ui::commas(state.blocks_done * block)
                    ),
                );
            }
            state.blocks_done
        }
        (false, Some(_)) => {
            ui.warn(
                "the checkpoint beside this filter describes a different scan -- a                  different corpus, or different flags. Starting over.",
            );
            0
        }
        _ => 0,
    };

    let corpus = Mutex::new(Corpus::open(corpus_path, block, resume_blocks)?);
    let sink = Mutex::new(MatchSink::open(ui, &config.out, config.details.as_deref())?);
    let completed: Mutex<BTreeSet<u64>> = Mutex::new(BTreeSet::new());
    let watermark = AtomicU64::new(resume_blocks);
    let points_done = AtomicU64::new(0);
    let done_reading = AtomicBool::new(false);
    let stop = install_interrupt(ui)?;
    let found = AtomicBool::new(false);

    let began = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..config.threads {
            let (target, scope, sink, stop, found) = (target, scope, &sink, &stop, &found);
            let (corpus, completed, watermark, points_done, done_reading) =
                (&corpus, &completed, &watermark, &points_done, &done_reading);
            s.spawn(move || {
                let mut deriver = Deriver::new();
                let mut expanded: Vec<[u8; 32]> = Vec::new();
                let mut lines: Vec<String> = Vec::new();
                let mut batch: Vec<[u8; 32]> = Vec::with_capacity(derive::POINTS_PER_BATCH);
                let mut candidates = Candidates {
                    target,
                    scope,
                    sink,
                    found,
                    vuln: vuln.id(),
                    base: 0,
                    stream: 0,
                    materials: [[0u8; 32]; derive::POINTS_PER_BATCH],
                };

                loop {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let Some(unit) = corpus.lock().unwrap().claim(block, &mut lines) else {
                        done_reading.store(true, Ordering::Relaxed);
                        break;
                    };

                    let mut interrupted = false;
                    // A passphrase can expand to more than one key -- a brainwallet is
                    // hashed once and twice -- and every material of a line is walked
                    // inside this block, so a block stays complete or not.
                    for (i, line) in lines.iter().enumerate() {
                        if stop.load(Ordering::Relaxed) {
                            interrupted = true;
                            break;
                        }
                        expanded.clear();
                        vuln.expand(Point::Input(line.as_bytes()), &mut expanded);
                        for material in &expanded {
                            batch.clear();
                            batch.push(*material);
                            candidates.base = unit as u128 * block as u128 + i as u128;
                            candidates.materials[0] = *material;
                            deriver.walk_batch(&batch, scope, &mut candidates);
                        }
                    }
                    points_done.fetch_add(lines.len() as u64, Ordering::Relaxed);

                    if interrupted {
                        break;
                    }
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

        let (sink, stop, found) = (&sink, &stop, &found);
        let (watermark, points_done, done_reading) = (&watermark, &points_done, &done_reading);
        s.spawn(move || {
            let interval = if ui.interactive() {
                Duration::from_secs(2)
            } else {
                Duration::from_secs(60)
            };
            let mut last_checkpoint = Instant::now();
            let mut announced_stop = false;
            loop {
                for _ in 0..(interval.as_millis() / 100) {
                    std::thread::sleep(Duration::from_millis(100));
                    if stop.load(Ordering::Relaxed)
                        || done_reading.load(Ordering::Relaxed)
                        || found.swap(false, Ordering::Relaxed)
                    {
                        break;
                    }
                }
                if stop.load(Ordering::Relaxed) && !announced_stop {
                    announced_stop = true;
                    ui.notice("interrupted", "finishing the block in flight and checkpointing");
                }

                let finished = done_reading.load(Ordering::Relaxed);
                if !finished {
                    let done = points_done.load(Ordering::Relaxed);
                    let rate = done as f64 / began.elapsed().as_secs_f64().max(1e-9);
                    let candidates = sink.lock().unwrap().candidates;
                    // A corpus has no known length until it has been read, so there is no
                    // percentage and no ETA to give. Saying how far it has got is the most
                    // that is true.
                    ui.progress(
                        "scanning",
                        0,
                        0,
                        "passphrases",
                        &[
                            format!("{} walked", ui::commas(done)),
                            format!("{rate:.0}/s"),
                            format!("{} candidates", ui::commas(candidates)),
                        ],
                    );
                }
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

    let finished = done_reading.load(Ordering::Relaxed) && !stop.load(Ordering::Relaxed);
    if finished {
        State::clear(state_path)?;
    } else {
        save(state_path, fingerprint, config, block, watermark.load(Ordering::Relaxed))?;
    }

    ui.clear();
    let sink = sink.lock().unwrap();
    let points_done = points_done.load(Ordering::Relaxed);
    Ok(ScanReport {
        points_done,
        streams: 1,
        gpu_profile: None,
        unconfirmed: 0,
        total_points: points_done as u128,
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

/// The process-wide stop flag, reset for each scan.
///
/// A signal handler is a property of the *process*, and `ctrlc` refuses a second
/// registration outright. Running two scans in one process is a perfectly reasonable
/// thing to do -- the tests do it, and a future `scan --vuln a --vuln b` would -- so the
/// handler is installed once and the flag it sets is cleared at the start of each run,
/// rather than a new handler being installed per scan and the second one failing.
fn install_interrupt(ui: &Ui) -> Result<Arc<AtomicBool>> {
    static STOP: OnceLock<Arc<AtomicBool>> = OnceLock::new();
    static INSTALLED: Once = Once::new();

    let stop = STOP.get_or_init(|| Arc::new(AtomicBool::new(false)));

    // The handler only sets the flag. Announcing the interrupt is left to the monitor
    // thread, which picks it up within 100ms: it owns the progress bar and can put it
    // back afterwards, and printing from a signal handler means taking the stderr lock
    // from a context that may already hold it. A second Ctrl-C kills the process outright
    // rather than being swallowed.
    let mut outcome = Ok(());
    INSTALLED.call_once(|| {
        let stop = Arc::clone(stop);
        let armed = AtomicBool::new(false);
        // `process::exit` runs no destructors, so `Drop for Ui` never gets to put the
        // cursor back on that path. Whether it was hidden is captured here rather than
        // read through the `Ui`, which the handler outlives.
        let hid_cursor = ui.interactive();
        outcome = ctrlc::set_handler(move || {
            if armed.swap(true, Ordering::SeqCst) {
                if hid_cursor {
                    ui::show_cursor();
                }
                std::process::exit(130);
            }
            stop.store(true, Ordering::SeqCst);
        })
        .context("installing the interrupt handler");
    });
    outcome?;

    // Cleared per run, so a scan that follows an interrupted one is not stopped before it
    // starts.
    stop.store(false, Ordering::SeqCst);
    Ok(Arc::clone(stop))
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
            corpus: None,
            out: out.clone(),
            details: None,
            start: 490,
            end: 510,
            threads: 2,
            block: Some(4),
            restart: true,
            gpu: None,
            gpu_batch: None,
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

#[cfg(test)]
mod corpus_tests {
    use super::*;
    use crate::scan::derive::Route;
    use crate::scan::sink::is_importable_secret;
    use crate::target::bloom::testing::{reference_add, scratch, write_filter};
    use crate::wallet::address::HashForm;

    /// A whole corpus scan, end to end.
    ///
    /// The filter holds the address for `sha256("satoshi")` -- the canonical brainwallet
    /// example -- and the corpus contains that phrase among others, including a blank
    /// line, which must not be hashed: `sha256("")` is a valid private key and would
    /// otherwise be reported against every corpus that ends with a newline.
    #[test]
    fn scans_a_corpus_end_to_end() {
        use crate::crypto::{ec, hash::hash160};
        use crate::vuln::brainwallet::Brainwallet;

        let key: [u8; 32] = {
            let mut out = Vec::new();
            Brainwallet.expand(Point::Input(b"satoshi"), &mut out);
            out[0]
        };
        let target_hash = hash160(&ec::public_key(&key).serialize());

        let mut bits = vec![0u64; 1 << 16];
        reference_add(&mut bits, &target_hash);
        let filter_path = scratch("corpus.bf");
        write_filter(&filter_path, &bits);

        let corpus = scratch("corpus-phrases.txt");
        std::fs::write(&corpus, "alpha\nbravo\nsatoshi\n\ncharlie\ndelta\n").unwrap();

        let out = scratch("corpus-matches.txt");
        let state = scratch("corpus.state");
        for p in [&out, &state] {
            let _ = std::fs::remove_file(p);
        }

        let ui = Ui::new(crate::ui::Stream::Stderr);
        let target = Target::new(
            crate::target::bloom::BloomFilter::open(&filter_path).unwrap(),
            None,
        );
        let scope = Scope {
            material_sizes: vec![32],
            routes: vec![Route::PrivKey],
            paths: vec![],
            forms: vec![HashForm::Compressed],
        };
        let config = ScanConfig {
            filter: filter_path.clone(),
            corpus: Some(corpus.clone()),
            out: out.clone(),
            details: None,
            start: 0,
            end: 0,
            threads: 2,
            block: Some(2),
            restart: true,
            gpu: None,
            gpu_batch: None,
        };

        let fp = corpus_fingerprint(&corpus).unwrap();
        let report =
            run_corpus(&ui, &Brainwallet, &scope, &target, &config, &fp, &state).unwrap();

        assert!(report.finished, "the corpus scan did not finish");
        // Five passphrases, not six: the blank line is skipped.
        assert_eq!(report.points_done, 5, "the blank line was hashed");
        assert_eq!(report.candidates, 1);

        let text = std::fs::read_to_string(&out).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 1);
        assert!(is_importable_secret(lines[0]));
        assert_eq!(lines[0], crate::ui::hex(&key));

        // A finished corpus leaves no checkpoint.
        assert!(!state.exists(), "a finished corpus scan left a checkpoint");

        // The fingerprint is the corpus's contents, so an edited file under the same name
        // is a different scan and must not be resumed into.
        std::fs::write(&corpus, "alpha\nbravo\nsatoshi\n\ncharlie\nEDITED\n").unwrap();
        assert_ne!(corpus_fingerprint(&corpus).unwrap(), fp);
    }

    /// Resuming must start at exactly the right line -- none re-derived, and more
    /// importantly none skipped.
    #[test]
    fn a_resumed_corpus_starts_at_the_right_line() {
        let corpus = scratch("corpus-skip.txt");
        let body: String = (0..100).map(|i| format!("phrase{i}\n")).collect();
        std::fs::write(&corpus, &body).unwrap();

        // Block of 10, resuming after 3 blocks: the next line must be phrase30.
        let mut c = Corpus::open(&corpus, 10, 3).unwrap();
        let mut lines = Vec::new();
        let unit = c.claim(10, &mut lines).unwrap();
        assert_eq!(unit, 3, "the resumed block is misnumbered");
        assert_eq!(lines.len(), 10);
        assert_eq!(lines[0], "phrase30");
        assert_eq!(lines[9], "phrase39");

        // And reading to the end stops rather than looping.
        let mut seen = lines.len();
        while c.claim(10, &mut lines).is_some() {
            seen += lines.len();
        }
        assert_eq!(seen, 70, "resuming did not cover the rest of the file exactly once");
    }
}
