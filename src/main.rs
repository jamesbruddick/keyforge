//! The CLI. Scan logic lives in the library; this file parses arguments, decides where
//! output goes, and prints the banner.

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use keyforge::scan::derive::{Route, Scope};
use keyforge::scan::engine::{self, ScanConfig};
use keyforge::scan::state::State;
use keyforge::target::{Target, bloom::BloomFilter};
use keyforge::ui::{self, Stream, Ui};
use keyforge::vuln::{self, Space, Vulnerability};
use keyforge::wallet::address::HashForm;
use keyforge::wallet::path::PathSpec;
use std::path::PathBuf;

/// `--gpu`'s long help, with the last paragraph filled in per build.
///
/// A macro because `concat!` takes literals and nothing else, and the alternative is
/// either a formatting crate for four sentences or the same three paragraphs written out
/// four times, once per feature combination.
macro_rules! gpu_help {
    ($built:literal) => {
        concat!(
            "Use the GPU. `both` also walks points on the CPU; `only` leaves the CPU to \
             confirm what the device finds.\n\n\
             The device is a filter and the CPU is the oracle: a device record only means \
             \"look at this point\", and the host re-derives that point before anything \
             reaches the matches file. A wrong kernel therefore cannot write a wrong \
             secret -- but it can silently miss wallets, which is what the `unconfirmed` \
             counter at the end of a scan is there to catch.\n\n",
            $built
        )
    };
}

#[cfg(all(feature = "metal", feature = "cuda"))]
const GPU_HELP: &str = gpu_help!("This binary has both the Metal and CUDA backends.");
#[cfg(all(feature = "metal", not(feature = "cuda")))]
const GPU_HELP: &str = gpu_help!("This binary has the Metal backend, for Apple GPUs.");
#[cfg(all(feature = "cuda", not(feature = "metal")))]
const GPU_HELP: &str =
    gpu_help!("This binary has the CUDA backend, for NVIDIA cards (driver 580 or newer).");
#[cfg(not(any(feature = "metal", feature = "cuda")))]
const GPU_HELP: &str = gpu_help!(
    "This binary has NO GPU backend compiled in, so this flag will fail. Rebuild with \
     `--features metal` (Apple) or `--features cuda` (NVIDIA, driver 580 or newer)."
);

#[derive(Parser)]
#[command(
    name = "keyforge",
    version,
    about,
    long_about = "Scan Bitcoin vulnerabilities that produce guessable mnemonic phrases \
                  or private keys.\n\n\
                  You give it a vulnerability to sweep and a bloom filter of funded \
                  addresses. It derives the wallets that vulnerability could have \
                  produced, tests each against the filter, and writes out the secret \
                  behind anything that matches.\n\n\
                  What it writes are CANDIDATES, not confirmed funds: a bloom filter \
                  answers \"definitely not\" exactly and \"probably yes\" approximately. \
                  `keyforge verify` is the first triage step.",
    after_help = "Getting started:\n  \
        keyforge vulns                              what can be scanned, and how\n  \
        keyforge vulns milksad                      the full guide for one\n\n\
      Scanning:\n  \
        keyforge scan --vuln milksad -f funded.bf   sweep a range\n  \
        keyforge scan --vuln milksad -f funded.bf --gpu\n  \
        keyforge scan --vuln brainwallet --corpus phrases.txt -f funded.bf\n\n\
      Triage:\n  \
        keyforge verify \"<phrase>\"                  addresses behind one secret\n  \
        keyforge verify < matches.txt               a whole file of them\n\n\
      A scan can be stopped with Ctrl-C and resumed by re-running the same command.\n\
      Use `keyforge <command> --help` for the full options of one."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Sweep a vulnerability's search space against a bloom filter of funded addresses.
    Scan(ScanArgs),
    /// List the vulnerabilities that can be scanned, or explain one in detail.
    Vulns {
        /// A vulnerability id or alias. Omit to list them all.
        name: Option<String>,
    },
    /// Derive the addresses behind a candidate secret, so it can be looked up on chain.
    Verify(VerifyArgs),
}

/// What to derive for each point.
///
/// Flattened into every subcommand that walks a tree, so a `verify` run can be told
/// exactly what a `scan` searched. `Option<Vec<_>>` on the vulnerability-dependent ones
/// is how "unset" is told apart from "explicitly set to what the default happens to be".
#[derive(Args, Clone)]
#[command(next_help_heading = "Derivation scope")]
struct ScopeArgs {
    /// Which vulnerability to sweep. See `keyforge vulns`.
    #[arg(long, value_name = "ID")]
    vuln: String,

    /// Prefix lengths of the secret material to try, in bytes (16, 20, 24, 28, 32).
    #[arg(long, value_name = "N", value_delimiter = ',')]
    material: Option<Vec<usize>>,

    /// How the material became keys: bip39, bip32-seed, privkey.
    #[arg(long, value_name = "ROUTE", value_delimiter = ',')]
    routes: Option<Vec<String>>,

    /// Derivation paths, repeatable. E.g. "m/44'/0'/0'/{0,1}/{0..9}".
    #[arg(long = "path", value_name = "PATH")]
    paths: Vec<String>,

    /// Which hash160 forms to derive: compressed, uncompressed, p2sh-p2wpkh.
    #[arg(long, value_name = "FORM", value_delimiter = ',')]
    hash_forms: Option<Vec<String>>,
}

impl ScopeArgs {
    fn vulnerability(&self) -> Result<&'static dyn Vulnerability> {
        vuln::find(&self.vuln).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown vulnerability `{}`\n\navailable: {}",
                self.vuln,
                vuln::names().join(", ")
            )
        })
    }

    /// Build the scope: start from the vulnerability's defaults, then apply overrides.
    ///
    /// All validation lives here, so a scan that is going to be refused is refused
    /// before the filter is read rather than three seconds later.
    fn to_scope(&self, v: &dyn Vulnerability) -> Result<Scope> {
        let mut scope = Scope::default();
        v.defaults().apply(&mut scope);
        self.override_scope(&mut scope)?;
        Ok(scope)
    }

    /// Apply the explicit overrides to a scope that already carries its defaults, and
    /// check the result.
    ///
    /// Split out from [`to_scope`](Self::to_scope) because `verify` needs exactly this
    /// half: its scope starts from a vulnerability's defaults *or* the scanner's, and
    /// duplicating the parsing there is how `--path` comes to mean two different things
    /// in two subcommands.
    fn override_scope(&self, scope: &mut Scope) -> Result<()> {
        if let Some(sizes) = &self.material {
            for size in sizes {
                if !keyforge::wallet::bip39::VALID_ENTROPY_SIZES.contains(size) {
                    bail!(
                        "material size {size} is not a BIP39 entropy size; \
                         expected one of 16, 20, 24, 28, 32"
                    );
                }
            }
            scope.material_sizes = sizes.clone();
        }
        if let Some(routes) = &self.routes {
            scope.routes = routes
                .iter()
                .map(|r| {
                    Route::parse(r).ok_or_else(|| {
                        anyhow::anyhow!("unknown route `{r}`; expected bip39, bip32-seed or privkey")
                    })
                })
                .collect::<Result<_>>()?;
        }
        if !self.paths.is_empty() {
            scope.paths = self
                .paths
                .iter()
                .map(|p| PathSpec::parse(p).map_err(|e| anyhow::anyhow!("{e}")))
                .collect::<Result<_>>()?;
        }
        if let Some(forms) = &self.hash_forms {
            scope.forms = forms
                .iter()
                .map(|f| {
                    HashForm::parse(f).ok_or_else(|| {
                        anyhow::anyhow!(
                            "unknown hash form `{f}`; expected compressed, uncompressed \
                             or p2sh-p2wpkh"
                        )
                    })
                })
                .collect::<Result<_>>()?;
        }

        scope.validate().map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(())
    }

    /// A stable string describing everything that decides what this scan walks.
    ///
    /// Resuming with any of it changed is refused, because the resulting file would
    /// describe a complete sweep of neither configuration.
    fn fingerprint_input(&self, scope: &Scope) -> Vec<String> {
        let mut parts = vec![self.vuln.clone()];
        parts.push(
            scope.material_sizes.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(","),
        );
        parts.push(scope.routes.iter().map(|r| r.as_str()).collect::<Vec<_>>().join(","));
        parts.push(scope.paths.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(" "));
        parts.push(scope.forms.iter().map(|f| f.as_str()).collect::<Vec<_>>().join(","));
        parts
    }
}

#[derive(Args)]
struct ScanArgs {
    #[command(flatten)]
    scope: ScopeArgs,

    /// Bloom filter of funded addresses, in keyscan `.bf` format.
    #[arg(short, long, value_name = "PATH")]
    filter: PathBuf,

    /// Passphrase list, one per line, for a vulnerability that walks a corpus.
    #[arg(long, value_name = "PATH")]
    corpus: Option<PathBuf>,

    /// Where candidate secrets are written, one per line.
    #[arg(short, long, value_name = "PATH", default_value = "matches.txt")]
    out: PathBuf,

    /// Also write full context for each hit as JSON, one object per line.
    #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "matches.jsonl")]
    details: Option<PathBuf>,

    /// First point of the range. Defaults to the start of the vulnerability's space.
    #[arg(long, value_name = "N")]
    start: Option<u128>,

    /// One past the last point. Defaults to the end of the vulnerability's space.
    #[arg(long, value_name = "N")]
    end: Option<u128>,

    /// Worker threads. Defaults to the number of cores.
    #[arg(short, long, value_name = "N")]
    threads: Option<usize>,

    /// Points per work block. Defaults to a value derived from the range.
    #[arg(long, value_name = "N")]
    block: Option<u64>,

    /// Ignore any existing checkpoint and start from the beginning.
    #[arg(long)]
    restart: bool,

    /// Use the GPU. `both` also walks points on the CPU; `only` leaves the CPU to
    /// confirm what the device finds.
    #[arg(long, value_name = "MODE", num_args = 0..=1, default_missing_value = "both",
          hide = !cfg!(feature = "gpu"),
          long_help = GPU_HELP)]
    gpu: Option<keyforge::gpu::Mode>,

    /// Points per device launch. Sized from the device's memory when not given.
    #[arg(long, value_name = "N", hide = !cfg!(feature = "gpu"))]
    gpu_batch: Option<usize>,
}

/// `verify` re-derives a secret through the same walk a scan used.
///
/// This is the first triage step and it is deliberately the *same code path*: a hit is
/// only worth acting on if the addresses it implies can be reproduced, and reproducing
/// them with a second implementation would only prove the two agree. `--vuln` is
/// optional here, unlike in a scan -- a phrase in `matches.txt` is a phrase whether or
/// not you still remember which sweep produced it -- and naming one just borrows its
/// scope so triage walks exactly what the scan walked.
#[derive(Args)]
struct VerifyArgs {
    /// Candidate secrets: mnemonic phrases or 64-character private keys. Reads stdin
    /// when none are given, so `keyforge verify < matches.txt` triages a whole file.
    #[arg(value_name = "SECRET")]
    secrets: Vec<String>,

    /// Borrow a vulnerability's derivation scope, so triage walks what the scan walked.
    #[arg(long, value_name = "ID")]
    vuln: Option<String>,

    /// Derivation paths, repeatable. E.g. "m/44'/0'/0'/{0,1}/{0..9}".
    #[arg(long = "path", value_name = "PATH")]
    paths: Vec<String>,

    /// How the material became keys: bip39, bip32-seed, privkey.
    #[arg(long, value_name = "ROUTE", value_delimiter = ',')]
    routes: Option<Vec<String>>,

    /// Which hash160 forms to derive: compressed, uncompressed, p2sh-p2wpkh.
    #[arg(long, value_name = "FORM", value_delimiter = ',')]
    hash_forms: Option<Vec<String>>,

    /// Test each derived address against a filter, and report only what it passes.
    #[arg(short, long, value_name = "PATH")]
    filter: Option<PathBuf>,

    /// Print every address derived, not just the first few of each secret.
    #[arg(long)]
    all: bool,
}

/// How many addresses a secret shows before the listing is summarised.
///
/// A default scope derives a few hundred per secret, which is more than anyone reads and
/// enough to bury the next secret in a file being triaged. The first few are the ones
/// worth looking up -- receive addresses at the front of each account -- and `--all` is
/// there for when they are not.
const VERIFY_PREVIEW: usize = 12;

fn main() -> std::process::ExitCode {
    let command = Cli::parse().command;
    // Scan writes its real output to a file, so its terminal chatter goes to stderr and
    // the file is the thing you redirect. `vulns` is the opposite: its output *is* the
    // answer, so it goes to stdout and pipes.
    let ui = Ui::new(match command {
        Command::Scan(_) => Stream::Stderr,
        Command::Vulns { .. } | Command::Verify(_) => Stream::Stdout,
    });

    let result = match command {
        Command::Scan(args) => run_scan(&ui, args),
        Command::Vulns { name } => run_vulns(&ui, name),
        Command::Verify(args) => run_verify(&ui, args),
    };

    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            ui.error(&format!("{error}"));
            for cause in error.chain().skip(1) {
                ui.error_cause(&format!("{cause}"));
            }
            std::process::ExitCode::FAILURE
        }
    }
}

fn run_scan(ui: &Ui, args: ScanArgs) -> Result<()> {
    let v = args.scope.vulnerability()?;
    let scope = args.scope.to_scope(v)?;

    // A corpus vulnerability walks a file; everything else walks a number range. The two
    // need different readers, different progress and different checkpoints, so which one
    // this is gets settled here rather than being threaded through the engine.
    let corpus = matches!(v.space(), Space::Corpus);
    if corpus && args.corpus.is_none() {
        bail!(
            "`{}` walks a passphrase corpus, so it needs one: --corpus <FILE>, \
             one passphrase per line",
            v.id()
        );
    }
    if !corpus && args.corpus.is_some() {
        bail!(
            "`{}` walks a number range, not a corpus. Narrow it with --start and --end.",
            v.id()
        );
    }

    let (space_start, space_end) = match v.space() {
        Space::Integers { start, end } => (start, end),
        Space::Corpus => (0, 0),
    };
    let (start, end) = if corpus {
        (0, 0)
    } else {
        let start = args.start.unwrap_or(space_start);
        let end = args.end.unwrap_or(space_end);
        if start >= end {
            bail!("--start {start} is not below --end {end}");
        }
        if start < space_start || end > space_end {
            bail!(
                "range {start}..{end} falls outside `{}`'s space of \
                 {space_start}..{space_end}",
                v.id()
            );
        }
        (start, end)
    };

    if args.gpu.is_some() && !cfg!(feature = "gpu") {
        bail!(
            "this binary was built without GPU support. Rebuild with one of:\n    \
             cargo build --release --features metal     (Apple)\n    \
             cargo build --release --features cuda      (NVIDIA, driver 580 or newer)"
        );
    }
    if args.gpu.is_some() && corpus {
        bail!("a corpus is walked on the CPU; drop --gpu");
    }

    let threads = args.threads.unwrap_or_else(|| {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    });
    if threads == 0 {
        bail!("--threads must be at least 1");
    }

    ui.title(env!("CARGO_PKG_VERSION"));
    ui.gap();

    // The vulnerability, as a heading: what it is called and how it is classified, then
    // its own prose under it. One block, so the reader knows what is being looked for
    // before being shown how.
    ui.row(v.id(), v.classification());
    for line in v.describe() {
        ui.cont_plain(&line);
    }

    // Narrowing a range is only sound for a vulnerability whose points have structure a
    // user can reason about. Everywhere else it leaves a hole, and saying so is the
    // whole reason `range_is_narrowable` exists.
    let narrowed = !corpus && (start != space_start || end != space_end);
    if narrowed && !v.range_is_narrowable() {
        ui.warn(&format!(
            "`{}`'s points have no time structure, so a narrowed range is a hole \
             rather than a saving: the wallets outside it are just as likely.",
            v.id()
        ));
    }

    let target = load_filter(ui, &args.filter, &scope)?;

    // What is being walked. The range and its size share a row because neither is much
    // use without the other, and the per-point costs go underneath because they are what
    // turns a point count into a running time.
    ui.gap();
    match &args.corpus {
        Some(path) => ui.row("search", &path.display().to_string()),
        None => ui.row(
            "search",
            // The bounds first and unpunctuated, because they are the two numbers a
            // reader retypes into --start and --end; the count beside them is the one
            // they are reading for. Grouping the bounds with commas would make them
            // readable and unpasteable at the same time.
            &format!(
                "{start} .. {end}  ·  {}",
                plural_u128(end - start, "point")
            ),
        ),
    }
    // Named because it multiplies the work per point, and because a rate counted per
    // point and one counted per walk differ by exactly this factor -- which is an easy
    // way to think a sweep is slower than it is.
    let streams = keyforge::scan::engine::streams_of(v, start);
    ui.cont(&format!(
        "{}per point: {}, {}, {}",
        match streams {
            1 => String::new(),
            n => format!("{n} streams, each walked in full · "),
        },
        plural(scope.probes_per_point(), "probe"),
        plural(scope.ec_ops_per_point(), "key"),
        // Not pluralised: `pbkdf2s` is not a word anyone writes.
        format!("{} pbkdf2", ui::commas(scope.pbkdf2_per_point())),
    ));
    ui.row("material", &fmt(&scope.material_sizes, |s| format!("{s}B")));
    ui.row("routes", &fmt(&scope.routes, |r| r.as_str().to_string()));
    // Only shown when something actually walks them. A privkey-only scope carries the
    // default path set and never touches it, and listing paths a scan does not walk
    // invites exactly the wrong conclusion about what a clean pass covered.
    let forms = fmt(&scope.forms, |f| f.as_str().to_string());
    if scope.routes.iter().any(|r| r.derives()) {
        for (i, path) in scope.paths.iter().enumerate() {
            ui.row(if i == 0 { "paths" } else { "" }, &path.to_string());
        }
        // The forms are a *cross product* with the paths, not a property of them: every
        // leaf of every path is hashed every way the scope asks for. Said here, under the
        // paths, because listing them as their own row reads as a second axis that lines
        // up with the purposes above -- and the natural conclusion, that m/44' is the
        // legacy row and m/49' the p2sh one, is not what this derives. It is worth
        // knowing which way round it is: `--hash-forms` is the flag that cuts the work
        // when the correspondence does hold for what you are looking for.
        ui.cont(&format!("each leaf hashed as {forms}"));
    } else {
        // Nothing walks a tree, so there is no path for a form to be a product with.
        ui.row("hash forms", &forms);
    }

    // What it is being walked *against*. This block used to be missing entirely: a sweep
    // named the vulnerability, the scope and the device, and said nothing at all about
    // the file that decides every verdict it reaches. A filter that is the wrong size,
    // covers the wrong forms, or admits strangers at a rate that will bury triage is a
    // thing to find out here rather than three days in.
    ui.gap();
    describe_filter(ui, &target, &args.filter);
    ui.row("cpu", &plural(threads as u64, "thread"));
    ui.row("output", &args.out.display().to_string());
    ui.gap();

    let state_path = args.filter.with_extension("state");
    let fingerprint_parts = {
        let mut parts = args.scope.fingerprint_input(&scope);
        parts.push(args.filter.display().to_string());
        parts.push(start.to_string());
        parts.push(end.to_string());
        // A corpus is identified by its contents, not its path: the same name holding a
        // different file would make a resume skip lines it never walked.
        if let Some(path) = &args.corpus {
            parts.push(engine::corpus_fingerprint(path)?);
        }
        parts
    };
    let refs: Vec<&str> = fingerprint_parts.iter().map(|s| s.as_str()).collect();
    let fingerprint = State::fingerprint(&refs);

    let config = ScanConfig {
        filter: args.filter.clone(),
        corpus: args.corpus.clone(),
        out: args.out,
        details: args.details,
        start,
        end,
        threads,
        block: args.block,
        restart: args.restart,
        gpu: args.gpu,
        gpu_batch: args.gpu_batch,
    };

    let report = if corpus {
        engine::run_corpus(ui, v, &scope, &target, &config, &fingerprint, &state_path)?
    } else {
        engine::run(ui, v, &scope, &target, &config, &fingerprint, &state_path)?
    };

    ui.gap();
    let rate = report.points_done as f64 / report.elapsed.as_secs_f64().max(1e-9);
    ui.row_strong(
        if report.finished { "finished" } else { "stopped" },
        &format!(
            "{} in {} ({:.0}/s{})",
            plural(report.points_done, if corpus { "passphrase" } else { "point" }),
            ui::duration(report.elapsed.as_secs_f64()),
            rate,
            // Both numbers, when they differ. A point is the unit of coverage; a walk is
            // the unit of work, and it is the one a per-draw counter reports.
            match report.streams {
                1 => String::new(),
                n => format!(", {:.0} walks/s", rate * n as f64),
            }
        ),
    );
    ui.row(
        "candidates",
        &format!(
            "{} secrets from {} matching positions",
            ui::commas(report.candidates),
            ui::commas(report.locations)
        ),
    );
    // Should always be zero. See `ScanReport::unconfirmed`: the device is a filter and
    // the CPU is the oracle, so this counts records the two disagree about -- which is the
    // only symptom a silently-wrong kernel has.
    if let Some(profile) = &report.gpu_profile {
        ui.gap();
        for line in profile.lines() {
            ui.cont_verbatim(line);
        }
    }
    if report.unconfirmed > 0 {
        ui.warn(&format!(
            "{} device records could not be reproduced on the CPU. The two \
             implementations have diverged; treat this sweep as incomplete and please \
             report it.",
            ui::commas(report.unconfirmed)
        ));
    }
    if !report.finished {
        ui.cont("re-run the same command to pick up where this stopped");
    }
    if report.candidates > 0 {
        ui.cont("these are candidates, not confirmed funds: check them on chain");
    }
    Ok(())
}

/// What a secret on a line of `matches.txt` turns out to be.
///
/// `matches.txt` holds one importable secret per line and nothing else, which is exactly
/// the two shapes here. Telling them apart is done by looking rather than by asking the
/// user, because the file does not record which is which and a triage run over a mixed
/// file is the normal case.
enum Secret {
    /// A BIP39 mnemonic, walked as a tree.
    Phrase(String),
    /// 32 raw bytes, which are the key itself.
    PrivKey([u8; 32]),
}

impl Secret {
    /// Classify one line, or say why it is neither shape.
    ///
    /// The checksum is verified rather than assumed: a mistyped phrase derives a
    /// perfectly valid but completely different wallet, and reporting its addresses as
    /// "the addresses behind what you pasted" is the one failure triage must not have.
    fn parse(text: &str) -> Result<Secret> {
        let text = text.trim();
        if text.is_empty() {
            bail!("empty secret");
        }
        // A private key is 64 hex characters. Checked first because it is unambiguous:
        // no BIP39 phrase is a single whitespace-free 64-character hex string.
        if text.len() == 64 && text.chars().all(|c| c.is_ascii_hexdigit()) {
            let mut bytes = [0u8; 32];
            for (i, byte) in bytes.iter_mut().enumerate() {
                *byte = u8::from_str_radix(&text[i * 2..i * 2 + 2], 16)
                    .expect("64 hex digits parse two at a time");
            }
            return Ok(Secret::PrivKey(bytes));
        }
        let words = text.split_whitespace().count();
        if keyforge::wallet::bip39::VALID_ENTROPY_SIZES.contains(&(words * 4 / 3))
            && words.is_multiple_of(3)
        {
            if !keyforge::wallet::bip39::mnemonic_is_valid(text) {
                bail!(
                    "this is {words} words, but its BIP39 checksum does not check out. \
                     That is a typo rather than a different wallet: a phrase with a bad \
                     checksum still derives addresses, just not the ones you meant."
                );
            }
            return Ok(Secret::Phrase(text.to_string()));
        }
        bail!(
            "this is neither a BIP39 mnemonic (12, 15, 18, 21 or 24 words) nor a \
             64-character hex private key. It is {words} word(s), {} characters.",
            text.chars().count()
        )
    }

    /// The label the report leads with.
    fn kind(&self) -> String {
        match self {
            Secret::Phrase(p) => {
                format!("BIP39 mnemonic, {} words", p.split_whitespace().count())
            }
            Secret::PrivKey(_) => "private key, 32 bytes".to_string(),
        }
    }
}

/// Re-derive candidate secrets and show the addresses behind them.
fn run_verify(ui: &Ui, args: VerifyArgs) -> Result<()> {
    // The scope: a vulnerability's defaults if one was named, the scanner's otherwise,
    // then any explicit override. Built through the same `ScopeArgs` a scan uses so the
    // two cannot drift -- a `--path` means the same thing in both.
    let named = match &args.vuln {
        Some(name) => Some(vuln::find(name).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown vulnerability `{name}`\n\navailable: {}",
                vuln::names().join(", ")
            )
        })?),
        None => None,
    };
    let mut scope = Scope::default();
    if let Some(v) = named {
        v.defaults().apply(&mut scope);
    }
    // A `verify` walks routes the scan's own scope may have excluded, so the overrides
    // are applied through the same parsers rather than re-spelled here.
    let overrides = ScopeArgs {
        vuln: args.vuln.clone().unwrap_or_default(),
        material: None,
        routes: args.routes.clone(),
        paths: args.paths.clone(),
        hash_forms: args.hash_forms.clone(),
    };
    overrides.override_scope(&mut scope)?;

    let target = match &args.filter {
        Some(path) => Some(load_filter(ui, path, &scope)?),
        None => None,
    };

    let lines = read_secrets(&args.secrets)?;
    if lines.is_empty() {
        bail!(
            "no secrets to verify. Pass them as arguments, or pipe a file of them: \
             keyforge verify < matches.txt"
        );
    }

    ui.title(env!("CARGO_PKG_VERSION"));
    ui.gap();

    let mut deriver = keyforge::scan::derive::Deriver::new();
    let mut bad = 0usize;
    let mut confirmed = 0usize;

    for (index, (line_no, line)) in lines.iter().enumerate() {
        if index > 0 {
            ui.gap();
        }
        let secret = match Secret::parse(line) {
            Ok(secret) => secret,
            Err(error) => {
                bad += 1;
                ui.warn(&format!("line {line_no}: {error}"));
                continue;
            }
        };

        // Rows of six, so a 24-word phrase is four readable lines rather than one that
        // runs off the side of the terminal.
        let rows = ui::phrase_lines(line);
        ui.row("secret", &ui.data(&rows[0]));
        for row in &rows[1..] {
            ui.cont_verbatim(&ui.data(row));
        }
        ui.row("type", &secret.kind());

        // Every hash the scope derives, in walk order, with its path and form. Collected
        // rather than streamed because the summary counts are what decides how much of
        // it to print.
        let mut found: Vec<(String, &'static str, String, bool)> = Vec::new();
        let mut derived = 0usize;
        {
            let mut visit = keyforge::scan::derive::All(
                |location: &keyforge::scan::derive::Location, hash: &[u8; 20], _: &str| {
                    derived += 1;
                    let hit = target.as_ref().is_some_and(|t| t.contains(hash));
                    // With a filter, only what passes it is worth a line; without one,
                    // the listing *is* the answer.
                    if target.is_none() || hit {
                        found.push((
                            location.path(&scope).unwrap_or_else(|| "-".to_string()),
                            location.form.as_str(),
                            keyforge::wallet::address::encode(location.form, hash),
                            hit,
                        ));
                    }
                },
            );
            match &secret {
                Secret::Phrase(phrase) => deriver.walk_phrase(phrase, &scope, &mut visit),
                // The privkey route takes the material as the key, which is what the
                // 32 bytes on the line are. A scope that excludes that route would derive
                // nothing at all, so it is added rather than assumed.
                Secret::PrivKey(key) => {
                    let mut scope = scope.clone();
                    scope.routes = vec![Route::PrivKey];
                    scope.material_sizes = vec![32];
                    deriver.walk_batch(&[*key], &scope, &mut visit);
                }
            }
        }

        match &target {
            Some(_) => {
                ui.row(
                    "addresses",
                    &format!("{} derived, {} passing the filter", ui::commas(derived as u64), ui::commas(found.len() as u64)),
                );
                if found.is_empty() {
                    ui.cont(
                        "the filter rules every one of them out, which is definite: a \
                         bloom filter has no false negatives. This secret's wallets are \
                         not in it.",
                    );
                } else {
                    confirmed += 1;
                }
            }
            None => ui.row("addresses", &format!("{} derived", ui::commas(derived as u64))),
        }

        let shown = if args.all { found.len() } else { found.len().min(VERIFY_PREVIEW) };
        // The path column is sized to what is in this listing, and dropped entirely when
        // nothing in it has a path: a raw private key is not derived from anything, and a
        // column of `-` invites the reader to look for the meaning of the dash.
        let paths = found[..shown].iter().any(|(p, ..)| p != "-");
        let width = found[..shown].iter().map(|(p, ..)| p.len()).max().unwrap_or(0);
        for (path, form, address, _) in &found[..shown] {
            let path = if paths { format!("{path:<width$}  ") } else { String::new() };
            ui.cont_verbatim(&format!(
                "{path}{}  {}",
                ui.dim(&format!("{form:<13}")),
                ui.data(address)
            ));
        }
        if found.len() > shown {
            ui.cont(&format!(
                "and {} more; pass --all to list them",
                ui::commas((found.len() - shown) as u64)
            ));
        }
    }

    ui.gap();
    if target.is_some() {
        ui.row_strong(
            "passing",
            &format!("{} of {} secrets", ui::commas(confirmed as u64), ui::commas(lines.len() as u64)),
        );
        ui.cont(
            "a filter says \"probably\", never \"yes\". Look these addresses up on chain \
             before treating any of them as funds.",
        );
    } else {
        ui.row_strong(
            "verified",
            &format!("{} of {} secrets", ui::commas((lines.len() - bad) as u64), ui::commas(lines.len() as u64)),
        );
        // Only when there is something to look up. Advising the reader to check addresses
        // on chain after a run that derived none is the kind of line that makes a tool
        // feel like it is not reading its own output.
        if bad < lines.len() {
            ui.cont("look these addresses up on chain to see whether they hold anything");
        }
    }
    if bad > 0 {
        bail!("{bad} line(s) were not a mnemonic or a private key");
    }
    Ok(())
}

/// The secrets to verify: the arguments, or every non-blank line of stdin.
///
/// Reading stdin when there are no arguments is what makes `keyforge verify < matches.txt`
/// the whole triage step rather than a shell loop. Blank lines and `#` comments are
/// skipped so a hand-annotated shortlist still works.
fn read_secrets(args: &[String]) -> Result<Vec<(usize, String)>> {
    if !args.is_empty() {
        return Ok(args.iter().cloned().enumerate().map(|(i, s)| (i + 1, s)).collect());
    }
    use std::io::{BufRead, IsTerminal};
    if std::io::stdin().is_terminal() {
        // Waiting for input nobody is typing looks exactly like a hang.
        // Written as concatenated pieces rather than one continued literal: the example
        // lines have to reach the terminal with their leading indent intact, and a `\`
        // line-continuation strips exactly that.
        bail!(concat!(
            "no secrets given, and stdin is a terminal.\n\n",
            "    keyforge verify \"<phrase>\"       one secret\n",
            "    keyforge verify < matches.txt   a whole file",
        ));
    }
    let mut out = Vec::new();
    // Numbered as the file numbers them, counting the lines that are skipped: a warning
    // that says "line 2" has to mean the second line of the file the reader is looking
    // at, not the second line that happened to survive the filter.
    for (index, line) in std::io::stdin().lock().lines().enumerate() {
        let line = line.context("reading secrets from stdin")?;
        let trimmed = line.trim();
        if !trimmed.is_empty() && !trimmed.starts_with('#') {
            out.push((index + 1, trimmed.to_string()));
        }
    }
    Ok(out)
}

/// The target block of the banner: the file every verdict is reached against.
///
/// Four facts, in the order they matter. The size is what has to fit in RAM. The forms
/// are what can match at all -- a filter without P2SH entries answers "no" to a third of
/// what a default scope derives, whatever the wallets behind them hold. The rate is how
/// much of `matches.txt` will turn out to be nothing, which is the difference between an
/// afternoon of triage and a week of it. The companion is what cuts that rate down.
fn describe_filter(ui: &Ui, target: &Target, path: &std::path::Path) {
    let primary = target.primary();
    ui.row(
        "filter",
        &format!(
            "{}, {}",
            path.display(),
            ui::bytes(primary.size_bytes() as u64)
        ),
    );
    let forms = keyforge::target::bloom::kind_names(primary.kinds());
    ui.cont(&format!(
        "{} layout, covering {}",
        primary.layout().name(),
        if forms.is_empty() { "nothing it will say".to_string() } else { forms.join(", ") }
    ));
    // Sampled rather than computed from a declared entry count, because the header does
    // not carry one and a filter holding far fewer entries than it was sized for is
    // common -- and reads as a much better rate than the file actually delivers.
    ui.cont(&format!(
        "false positives: {}",
        ui::one_in(primary.false_positive_rate(FILL_SAMPLES))
    ));
    match target.verify() {
        Some(companion) => ui.cont(&format!(
            "{} alongside, ruling most of those out ({} between them)",
            keyforge::target::verify_path(path).display(),
            ui::one_in(
                primary.false_positive_rate(FILL_SAMPLES)
                    * companion.false_positive_rate(FILL_SAMPLES)
            )
        )),
        None => ui.cont(
            "no verification filter alongside it, so every false positive reaches \
             matches.txt for triage to rule out",
        ),
    }
}

/// Words sampled to estimate a filter's fill, and from it its false-positive rate.
///
/// The bits are far too many to count -- a 7.6 GB filter is 60 billion of them -- and
/// they are uniformly distributed by construction, so a stride sample converges quickly.
/// Ten thousand is far past where the estimate stops moving and still costs nothing
/// beside the read that just loaded the file.
const FILL_SAMPLES: usize = 10_000;

/// Total physical RAM, where the platform will say.
#[cfg(target_os = "linux")]
fn physical_ram() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

#[cfg(target_os = "macos")]
fn physical_ram() -> Option<u64> {
    let mut bytes: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: the name is a NUL-terminated C string, and the out-buffer is a live u64
    // whose size is what `len` reports.
    let rc = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&raw mut bytes).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0).then_some(bytes)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn physical_ram() -> Option<u64> {
    None
}

/// Refuse a filter pair that cannot fit in RAM, before spending twenty minutes proving it.
///
/// Filters are read into memory whole rather than mapped, and deliberately: probes are
/// uniform over the whole bit space, so anything short of resident degrades to a page
/// fault per probe and a sweep that would have taken days takes years. The consequence is
/// that the pair has to fit, and a pair that does not produces swapping rather than an
/// error -- the process stays alive, makes almost no progress, and gives no clue why.
///
/// A verification companion is optional, so when only *it* pushes the total over the
/// edge the useful thing is to say so and carry on without it: a scan with no companion
/// works fine, it just reports more false positives for triage to rule out.
fn check_memory(ui: &Ui, filter: &std::path::Path) -> Result<bool> {
    let size = |p: &std::path::Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    let primary = size(filter);
    let companion_path = keyforge::target::verify_path(filter);
    let companion = if companion_path.exists() { size(&companion_path) } else { 0 };

    let Some(ram) = physical_ram() else {
        return Ok(true);
    };
    // Leave a margin for the derivation buffers, the page cache and the rest of the
    // machine. A filter that exactly fills RAM still swaps.
    let usable = ram - (ram / 8).min(2 << 30);

    if primary > usable {
        bail!(
            "the filter is {} and this machine has {} of RAM.\n\
             Filters are held resident on purpose -- probes are uniform over the whole \
             bit space, so a filter that has to be paged in makes a sweep hundreds of \
             times slower rather than a little slower.\n\
             Use a smaller filter, or a machine with more memory.",
            ui::bytes(primary),
            ui::bytes(ram)
        );
    }
    if companion > 0 && primary + companion > usable {
        ui.warn(&format!(
            "the filter and its verification companion are {} together, which does not \
             fit in {} of RAM. Scanning with the filter alone: candidates will still be \
             correct, there will just be more false positives for triage to rule out.",
            ui::bytes(primary + companion),
            ui::bytes(ram)
        ));
        return Ok(false);
    }
    Ok(true)
}

/// Open the filter, and say plainly if it cannot answer for part of the scope.
fn load_filter(ui: &Ui, path: &std::path::Path, scope: &Scope) -> Result<Target> {
    let with_companion = check_memory(ui, path)?;
    let primary = BloomFilter::open(path)
        .with_context(|| format!("opening the filter {}", path.display()))?;
    let verify = if with_companion { Target::open_verify(path)? } else { None };
    let target = Target::new(primary, verify);

    // A filter built without, say, P2SH entries will never match a form the scan spends
    // a third of its time deriving. Silence here is what turns that into a clean pass
    // that proves less than it appears to.
    let unmatchable = target.unmatchable_forms(&scope.forms);
    if !unmatchable.is_empty() {
        ui.warn(&format!(
            "this filter holds no entries for {}, so those forms can never match; \
             drop them with --hash-forms to stop deriving them",
            fmt(&unmatchable, |f| f.as_str().to_string())
        ));
    }
    Ok(target)
}

/// A count and its unit, with the `s` only where it belongs.
///
/// `1 keys` in the banner of a scan that is otherwise carefully aligned is small, and
/// exactly the kind of small that makes a tool read as unfinished. The narrow scopes hit
/// it constantly: `low-int` derives one key per point.
fn plural_u128(count: u128, unit: &str) -> String {
    match count {
        1 => format!("1 {unit}"),
        n => format!("{} {unit}s", ui::commas_u128(n)),
    }
}

fn plural(count: u64, unit: &str) -> String {
    match count {
        1 => format!("1 {unit}"),
        n => format!("{} {unit}s", ui::commas(n)),
    }
}

fn fmt<T>(items: &[T], show: impl Fn(&T) -> String) -> String {
    items.iter().map(show).collect::<Vec<_>>().join(", ")
}

fn run_vulns(ui: &Ui, name: Option<String>) -> Result<()> {
    match name {
        None => {
            list_vulnerabilities(ui);
            Ok(())
        }
        Some(name) => match vuln::find(&name) {
            Some(v) => {
                print!("{}", vuln::render_guide(v, ui::terminal_width().min(88)));
                Ok(())
            }
            None => bail!(
                "unknown vulnerability `{name}`\n\navailable: {}",
                vuln::names().join(", ")
            ),
        },
    }
}

/// One line per vulnerability: the name to type, what it costs, whether a GPU can help,
/// and what it is.
///
/// The space size is shown because it is the number that decides whether a sweep is an
/// afternoon or a fortnight, and it is the first thing worth knowing. The GPU column is
/// shown because it is the second: it is the difference between nine days and nine hours,
/// and finding out that the one vulnerability you picked is the one with no kernel should
/// not require starting a sweep.
fn list_vulnerabilities(ui: &Ui) {
    let rows: Vec<_> = vuln::registry()
        .iter()
        .map(|v| {
            let space = match v.space().len() {
                Some(n) if n >= 1 << 20 => format!("2^{:.0}", (n as f64).log2()),
                Some(n) => ui::commas_u128(n),
                None => "corpus".to_string(),
            };
            let gpu = if v.kernel().is_some() { "gpu" } else { "cpu" };
            (v.id(), space, gpu, v.guide().what)
        })
        .collect();

    let id_width = rows.iter().map(|(id, ..)| id.len()).max().unwrap_or(0);
    let space_width = rows.iter().map(|(_, s, ..)| s.len()).max().unwrap_or(0).max(5);
    // Whatever the terminal has left after the three fixed columns and their separators
    // goes to the summary, so a wide window shows more of each description rather than
    // the same 62 characters with empty space beside them.
    let used = 2 + id_width + 2 + space_width + 2 + 3 + 2;
    let budget = ui::terminal_width().min(110).saturating_sub(used).max(24);

    println!();
    println!(
        "  {}  {}  {}  {}",
        ui.dim(&format!("{:<id_width$}", "vulnerability")),
        ui.dim(&format!("{:>space_width$}", "space")),
        ui.dim("run"),
        ui.dim("what went wrong")
    );
    for (id, space, gpu, what) in &rows {
        println!(
            "  {}  {}  {}  {}",
            ui.headline(&format!("{id:<id_width$}")),
            format_args!("{space:>space_width$}"),
            ui.dim(gpu),
            summarise(what, budget)
        );
    }
    println!();
    println!("  {}", ui.dim("`gpu` means the sweep has a device kernel; `cpu` means it does not."));
    println!();
    println!("  keyforge vulns <name>            the full guide for one");
    println!("  keyforge scan --vuln <name> -f funded.bf");
}

/// The opening of a guide's `what`, trimmed to fit one terminal line.
///
/// A listing is an index rather than documentation: it has to be scannable down the
/// left edge, so every row gets the same budget and the guide holds the rest. Cuts on a
/// word boundary, because a name sliced in half reads as a different name.
fn summarise(text: &str, budget: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.len() <= budget {
        return flat;
    }
    // Walk characters rather than bytes. Guide text is prose, and a single em dash
    // straddling the budget would make byte slicing panic the listing instead of
    // shortening a line. `char_indices` also gives the last space for free.
    let mut cut = 0;
    let mut last_space = None;
    for (i, c) in flat.char_indices() {
        if i >= budget {
            break;
        }
        cut = i + c.len_utf8();
        if c == ' ' {
            last_space = Some(i);
        }
    }
    let cut = last_space.unwrap_or(cut);
    format!("{}...", flat[..cut].trim_end_matches([',', ';', ':', '-']).trim_end())
}
