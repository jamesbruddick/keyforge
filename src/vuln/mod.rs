//! Which vulnerability a sweep is looking for.
//!
//! Every vulnerability here has the same shape -- something that should have produced an
//! unguessable secret produced a guessable one instead -- so they share this program's
//! whole pipeline and differ only in two things: which points exist to try, and what
//! secret material each point produces. A [`Vulnerability`] answers both, and everything
//! downstream of [`expand`](Vulnerability::expand) is shared. That is what makes adding
//! one a small, local piece of work.
//!
//! # A vulnerability is not a byte stream
//!
//! Several of these reduce to the same stream. Trust Wallet's `rng() & 0x000000ff`,
//! PHP 7.1's `mt_rand(0, 255)` and a macOS-built `bx` all mask the low byte of a
//! conforming MT19937 word, so they produce byte-identical entropy from the same seed;
//! `mt19937::tests::modern_php_and_trust_wallet_are_the_libcxx_stream` is that claim,
//! checked. Giving each its own stream would double a sweep's cost to rediscover the
//! same wallets, so what differs between them is recorded in the *scope* and the
//! *label* rather than in the generator.
//!
//! # What a vulnerability sets are defaults
//!
//! Each one narrows the scan to what the affected software could actually have produced
//! -- Trust Wallet Core only ever made 12-word BIP39 wallets, so two thirds of a `bx`
//! entropy scope is dead work against it -- and every default stays overridable, because
//! the boundary of what a program produced is itself an assumption.
//! [`Vulnerability::describe`] is what the banner prints, so a narrowed run says out
//! loud what it is walking past, and [`Vulnerability::guide`] is what the README and
//! `keyforge vulns <id>` print, so nobody has to read this file to use one.

pub mod brainwallet;
pub mod glibc;
pub mod java;
pub mod mt19937;
pub mod python;
pub mod truncated;
pub mod weak_key;

use crate::scan::derive::{Route, Scope};
use crate::wallet::path::PathSpec;

/// The search space a vulnerability leaves behind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Space {
    /// A contiguous range of integers, walked exhaustively.
    ///
    /// `u128` rather than `u64` because the interesting spaces already span 2^32
    /// (a Unix timestamp), 2^48 (`java.util.Random`'s LCG state) and 2^64, and a space
    /// that does not fit is one that silently gets truncated.
    Integers { start: u128, end: u128 },
    /// Externally supplied inputs, streamed: a passphrase corpus for a brainwallet.
    /// There is no arithmetic that enumerates these, so the scan reads them.
    Corpus,
}

impl Space {
    /// How many points the space holds, or `None` for a corpus, whose size is not known
    /// until the file is read.
    pub fn len(&self) -> Option<u128> {
        match self {
            Space::Integers { start, end } => Some(end.saturating_sub(*start)),
            Space::Corpus => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == Some(0)
    }
}

/// One point of a search space, ready to expand.
#[derive(Clone, Copy, Debug)]
pub enum Point<'a> {
    /// The `n`th integer of an [`Space::Integers`] space.
    Integer(u128),
    /// One line of a corpus, borrowed from the reader's buffer.
    Input(&'a [u8]),
}

/// What a point expands to.
///
/// There is deliberately no `Material` enum here distinguishing entropy from a private
/// key from a mnemonic. That distinction already exists, one layer down, as
/// [`Route`]: raw bytes that could have become a wallet three ways are
/// `Route::Bip39 | Bip32Seed | PrivKey`, and a brainwallet hash that is simply a key is
/// `Route::PrivKey` on its own. A vulnerability says which by what it puts in
/// [`Defaults::routes`], so expanding a point only has to produce **bytes**.
///
/// Keeping it to a fixed-size array is also what keeps `expand` off the allocator: the
/// hot loop calls it billions of times.
pub type Expanded = [u8; 32];

/// Defaults a vulnerability narrows the scan to. Every field is overridable by a flag.
#[derive(Clone, Debug)]
pub struct Defaults {
    /// Prefix lengths of the material to try, in bytes.
    pub material_sizes: Vec<usize>,
    /// How the material could have become keys.
    pub routes: Vec<Route>,
    /// Derivation paths worth walking. Empty means "use the scanner's default set".
    pub paths: Vec<PathSpec>,
}

impl Defaults {
    /// Apply these to a scope, leaving anything the vulnerability does not narrow.
    pub fn apply(&self, scope: &mut Scope) {
        if !self.material_sizes.is_empty() {
            scope.material_sizes = self.material_sizes.clone();
        }
        if !self.routes.is_empty() {
            scope.routes = self.routes.clone();
        }
        if !self.paths.is_empty() {
            scope.paths = self.paths.clone();
        }
    }
}

/// The user-facing "how to scan this" entry.
///
/// Five fields, the same five for every vulnerability, because a reader comparing two of
/// them should not have to work out which parts correspond. The struct exists so this
/// text cannot drift from the code: `keyforge vulns <id>` prints it, the README carries
/// the same words, and a test asserts every registered vulnerability fills it in.
#[derive(Clone, Debug)]
pub struct Guide {
    /// What went wrong, in one or two sentences, assuming no cryptography.
    pub what: &'static str,
    /// The software and versions affected, and the CVE if there is one.
    pub affected: &'static str,
    /// One copy-pasteable command that works as written.
    pub command: &'static str,
    /// How long a full sweep takes, on real hardware.
    pub time: &'static str,
    /// What a hit looks like and the one next step.
    pub hit: &'static str,
}

/// A vulnerability that produces guessable Bitcoin keys.
pub trait Vulnerability: Send + Sync {
    /// The name the CLI knows this by. Lowercase, hyphenated, stable: it goes in
    /// checkpoints and scripts.
    fn id(&self) -> &'static str;

    /// Other accepted spellings, for names that have more than one common form.
    fn aliases(&self) -> &'static [&'static str] {
        &[]
    }

    /// The CVE, where there is one. Its absence is itself worth saying: it usually means
    /// a class of bug rather than one shipped product.
    fn cve(&self) -> Option<&'static str> {
        None
    }

    /// The banner lines: what this is, and what its scope assumes.
    fn describe(&self) -> Vec<String>;

    /// The README and `keyforge vulns <id>` entry.
    fn guide(&self) -> Guide;

    /// The space to walk.
    fn space(&self) -> Space;

    /// Scope narrowing that follows from what the affected software could produce.
    fn defaults(&self) -> Defaults;

    /// Whether narrowing the range is sound.
    ///
    /// This is where vulnerabilities genuinely differ in kind. `bx` seeded from a 32-bit
    /// **timestamp**, so the plausible range is the years the tool was in use and
    /// `--start`/`--end` are a real economy. A seed drawn from a weak `random_device` has
    /// no time structure, is spread across the whole space, and a narrowed range is a
    /// hole rather than a saving. Getting this wrong in the second direction is the
    /// expensive mistake, so the default is the safe one.
    fn range_is_narrowable(&self) -> bool {
        false
    }

    /// Turn one point into the secret material it produces.
    ///
    /// Several, in general: a generator with more than one plausible byte mapping
    /// produces one material per mapping, and they are materials for the *same* point
    /// rather than separate passes over the range. The order must be stable, since it is
    /// how a caller maps a material back to the stream that produced it.
    ///
    /// `out` is a caller-owned buffer, cleared before each call, so this allocates
    /// nothing on the hot path. How the bytes are then interpreted is
    /// [`Defaults::routes`]' business, not this method's.
    fn expand(&self, point: Point<'_>, out: &mut Vec<Expanded>);
}

/// Every vulnerability the binary knows about.
///
/// A slice rather than a map: there are a couple of dozen at most, `id` lookups happen
/// once per run, and a slice keeps the listing order meaningful -- the ones with
/// published populations first.
pub fn registry() -> &'static [&'static dyn Vulnerability] {
    &[
        &mt19937::MilkSad,
        &mt19937::TrustWallet,
        &mt19937::PhpMt,
        &python::PythonRandom,
        &glibc::GlibcRand,
        &java::JavaUtilRandom,
        &brainwallet::Brainwallet,
        &truncated::TruncatedEntropy,
        &weak_key::LowInteger,
        &weak_key::RepeatedByte,
    ]
}

/// Look one up by id or alias.
pub fn find(name: &str) -> Option<&'static dyn Vulnerability> {
    registry()
        .iter()
        .copied()
        .find(|v| v.id() == name || v.aliases().contains(&name))
}

/// Render one vulnerability's guide as the block `keyforge vulns <id>` prints.
///
/// The README carries the same five parts in the same order for every vulnerability, so
/// a reader comparing two of them never has to work out which bits correspond, and
/// `readme_documents_every_vulnerability` holds the file to it.
pub fn render_guide(v: &dyn Vulnerability) -> String {
    let g = v.guide();
    let mut s = String::new();
    let title = match v.cve() {
        Some(cve) => format!("{} ({cve})", v.id()),
        None => v.id().to_string(),
    };
    s.push_str(&format!("{title}\n\n"));
    s.push_str(&format!("What went wrong\n  {}\n\n", wrap(g.what, 2)));
    s.push_str(&format!("Who is affected\n  {}\n\n", wrap(g.affected, 2)));
    s.push_str(&format!("Scan it with\n  {}\n\n", g.command));
    s.push_str(&format!("How long it takes\n  {}\n\n", wrap(g.time, 2)));
    s.push_str(&format!("What a hit looks like\n  {}\n", wrap(g.hit, 2)));
    s
}

/// Reflow prose that was written as an indented Rust string literal, where the source
/// indentation is not part of the text.
fn wrap(text: &str, indent: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut out = String::new();
    let mut col = indent;
    for word in words {
        if col + word.len() + 1 > 78 && col > indent {
            out.push('\n');
            out.push_str(&" ".repeat(indent));
            col = indent;
        } else if col > indent {
            out.push(' ');
            col += 1;
        }
        out.push_str(word);
        col += word.len();
    }
    out
}

/// Every accepted name, for an error message that lists the alternatives.
pub fn names() -> Vec<&'static str> {
    registry().iter().map(|v| v.id()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ids and aliases must be unique across the registry, and every one must resolve
    /// back to the vulnerability that declared it. Two plugins answering to the same
    /// name would make which one a scan walked depend on registry order.
    #[test]
    fn every_name_resolves_to_exactly_one_vulnerability() {
        let mut seen: Vec<&str> = Vec::new();
        for v in registry() {
            for name in std::iter::once(v.id()).chain(v.aliases().iter().copied()) {
                assert!(!seen.contains(&name), "`{name}` is claimed twice");
                seen.push(name);
                assert_eq!(find(name).map(|f| f.id()), Some(v.id()));
            }
        }
        assert_eq!(find("nonsense").map(|v| v.id()), None);
    }

    /// Ids go into checkpoints and scripts, so they have to be stable, typeable, and
    /// free of the shell's opinions.
    #[test]
    fn ids_are_lowercase_and_hyphenated() {
        for v in registry() {
            let id = v.id();
            assert!(!id.is_empty());
            assert!(
                id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
                "`{id}` is not a plain lowercase-hyphenated id"
            );
        }
    }

    /// A narrowed scope has to say what it is walking past, so a clean pass over a
    /// preset never reads as a clean pass over the keyspace.
    #[test]
    fn every_vulnerability_describes_its_assumption() {
        for v in registry() {
            let text = v.describe().join(" ");
            assert!(text.contains(v.id()), "{} does not name itself", v.id());
            assert!(text.len() > 80, "{} says too little", v.id());
        }
    }

    /// Every vulnerability must be documented, in the same five parts, before it can
    /// ship. This is the test that keeps `keyforge vulns` and the README honest.
    #[test]
    fn every_vulnerability_has_a_complete_guide() {
        for v in registry() {
            let g = v.guide();
            for (field, text) in [
                ("what", g.what),
                ("affected", g.affected),
                ("command", g.command),
                ("time", g.time),
                ("hit", g.hit),
            ] {
                assert!(
                    !text.trim().is_empty(),
                    "{} has an empty `{field}` in its guide",
                    v.id()
                );
            }
            // The command has to actually name this vulnerability, or it documents
            // something else.
            assert!(
                g.command.contains("keyforge") && g.command.contains(v.id()),
                "{}'s guide command does not invoke it: {}",
                v.id(),
                g.command
            );
            assert!(g.what.len() > 40, "{}'s `what` is too terse to help", v.id());
        }
    }

    /// A point of whatever kind this vulnerability's space is made of.
    ///
    /// Feeding an integer to a corpus vulnerability is a programming error the
    /// `expand` implementations assert against, so a registry-wide test has to ask the
    /// space what shape its points are rather than assuming.
    fn sample_point(v: &dyn Vulnerability) -> Point<'static> {
        match v.space() {
            Space::Integers { start, .. } => Point::Integer(start),
            Space::Corpus => Point::Input(b"correct horse battery staple"),
        }
    }

    /// No vulnerability may expand a point to the same material twice: that would derive
    /// the same wallets a second time and report nothing new. The three products that
    /// share the libc++ byte stream are the reason this is worth asserting rather than
    /// assuming.
    #[test]
    fn no_vulnerability_expands_a_point_to_repeated_material() {
        for v in registry() {
            let mut out = Vec::new();
            v.expand(sample_point(*v), &mut out);
            assert!(!out.is_empty(), "{} expanded to nothing", v.id());

            let mut sorted = out.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), out.len(), "{} repeats material", v.id());
        }
    }

    /// Every vulnerability's first point must expand to something its own routes can
    /// actually use, or that point is silently dropped by the walk while still being
    /// counted as scanned.
    ///
    /// What "usable" means depends on the route, and conflating the two is a mistake
    /// worth spelling out. All-zero *entropy* is a perfectly good BIP39 wallet -- it is
    /// the published `abandon abandon ... about` vector, and `truncated-entropy` starts
    /// there deliberately. All-zero *key material* is not a valid secp256k1 scalar and
    /// never derives anything. So the check is only applied to vulnerabilities whose
    /// defaults actually put the bytes on the privkey route.
    #[test]
    fn the_first_point_of_every_space_expands_to_something_its_routes_can_use() {
        for v in registry() {
            let mut out = Vec::new();
            v.expand(sample_point(*v), &mut out);
            assert!(!out.is_empty(), "{} expanded to nothing", v.id());

            if !v.defaults().routes.contains(&Route::PrivKey) {
                continue;
            }
            for material in &out {
                assert!(
                    secp256k1::SecretKey::from_byte_array(*material).is_ok(),
                    "{}'s first point is not a valid private key, but its defaults put \
                     it on the privkey route",
                    v.id()
                );
            }
        }
    }

    /// The README must carry a section for every registered vulnerability, and that
    /// section must show the command the plugin itself documents.
    ///
    /// This is what stops a new plugin shipping undocumented. It checks the parts that
    /// have to be identical -- the id and the runnable command -- rather than the prose,
    /// which is regenerated by `cargo run --example dump_guides` and free to be reworded.
    #[test]
    fn readme_documents_every_vulnerability() {
        let readme = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("README.md"),
        )
        .expect("README.md is part of the crate");

        for v in registry() {
            let heading = format!("### `{}`", v.id());
            assert!(
                readme.contains(&heading),
                "README has no `{heading}` section; regenerate with \
                 `cargo run --example dump_guides`"
            );
            assert!(
                readme.contains(v.guide().command),
                "README does not show {}'s documented command: {}",
                v.id(),
                v.guide().command
            );
        }
    }

    /// A vulnerability that walks nothing would report a clean pass having done no work.
    #[test]
    fn no_vulnerability_has_an_empty_space() {
        for v in registry() {
            assert!(!v.space().is_empty(), "{} walks nothing", v.id());
        }
    }

    /// Defaults must produce a scope that actually probes something.
    #[test]
    fn defaults_leave_a_scope_that_derives_something() {
        for v in registry() {
            let mut scope = Scope::default();
            v.defaults().apply(&mut scope);
            assert!(
                scope.probes_per_point() > 0,
                "{} narrows the scope to nothing",
                v.id()
            );
        }
    }
}
