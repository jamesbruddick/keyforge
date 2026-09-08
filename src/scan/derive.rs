//! The BIP32 walk: secret material in, hash160s out.
//!
//! This rolls its own BIP32 rather than using the `bitcoin` crate, so the hot loop
//! neither allocates nor parses derivation paths -- at ~90 derivations per point and
//! billions of points, that overhead is the whole budget. Correctness is not taken on
//! trust: the tests hold every path in the default scope against `bitcoin`'s own
//! `Xpriv`, plus the published BIP32 vectors.
//!
//! The walk is breadth-first, and that is a performance decision rather than a
//! stylistic one. Both of the expensive layers underneath want work in bulk:
//! `crate::crypto::ec` turns a whole level's scalars into public keys behind a single
//! field inversion, and `crate::crypto::pbkdf2` runs the batch's mnemonic stretching
//! `LANES` streams at a time. Walking one path to its leaf before starting the next
//! would hand each of them a single item and give up both. [`POINTS_PER_BATCH`] points
//! are therefore in flight at once, and a level of the tree is a flat list rather than a
//! stack frame.
//!
//! # One level, many shapes
//!
//! The scanner this grew out of walked exactly one tree shape -- three hardened levels,
//! then a chain level, then an index level -- so a level was a homogeneous thing. A
//! [`PathSpec`] set is not: `m/44'/0'/0'/{0,1}/{0..9}` and `m/{0,1}/{0..9}` reach their
//! leaves at different depths, and both are in the default scope.
//!
//! The answer is that **a level is a flat list of cursors from every spec at once**, and
//! each cursor draws its children from its own spec at its own depth. Concatenating is
//! load-bearing rather than tidy: `crate::crypto::ec::Batch` runs 256 scalars through one
//! digit position per inversion, and a single spec's shallow levels are only
//! `POINTS_PER_BATCH x sizes x routes` wide -- two dozen, nowhere near enough to pay for
//! an inversion. Pooled across specs and depths they are.
//!
//! # A finished cursor is a leaf
//!
//! A cursor whose spec has run out of segments moves to the leaf pool, and that single
//! rule replaces a special case. The old walk carried raw private keys -- `bx ec-new`,
//! where the entropy *is* the key -- in a separate list, spliced in after the last level
//! so their public keys could still ride the leaf level's shared inversion. Here such a
//! key is simply a cursor born with no segments left, and it joins the pool by the same
//! rule as a BIP44 leaf. There is no second path through this file for it.
//!
//! Taproot is deliberately absent. P2TR commits to an x-only key with no hash160
//! anywhere in the output, so a hash160 filter structurally cannot test it.

use crate::crypto::ec::{self, Ge};
use crate::crypto::hash::hash160;
use crate::crypto::pbkdf2::{self, HmacSha512};
use crate::target::bloom::PROBE_CHUNK;
use crate::wallet::address::HashForm;
use crate::wallet::bip39;
use crate::wallet::path::PathSpec;
use secp256k1::{Scalar, SecretKey};

/// How the secret material became key material.
///
/// A vulnerability decides what bytes exist; this decides what a wallet did with them,
/// and the resulting keys are completely different. A sweep that only covers the BIP39
/// route misses real wallets, which is why this is an axis of the scope rather than a
/// property of the vulnerability.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Route {
    /// The bytes are BIP39 entropy: entropy -> mnemonic -> PBKDF2 -> BIP32 master.
    /// The headline Milk Sad path, and the only one Trust Wallet Core had.
    Bip39,
    /// The bytes are the BIP32 seed directly, no BIP39 involved -- `bx hd-new`.
    Bip32Seed,
    /// The bytes are a private key directly -- `bx ec-new`. Only defined where the
    /// material is exactly 32 bytes.
    PrivKey,
}

impl Route {
    pub fn as_str(self) -> &'static str {
        match self {
            Route::Bip39 => "bip39",
            Route::Bip32Seed => "bip32-seed",
            Route::PrivKey => "privkey",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "bip39" => Some(Route::Bip39),
            // The old scanner's spellings, kept so a documented command still runs.
            "bip32-seed" | "raw-master" => Some(Route::Bip32Seed),
            "privkey" | "raw-privkey" => Some(Route::PrivKey),
            _ => None,
        }
    }

    /// Whether this route derives a tree at all. `PrivKey` does not: it has no path,
    /// and multiplying it by the path set would derive the same key once per spec.
    pub fn derives(self) -> bool {
        self != Route::PrivKey
    }
}

/// Where a hash160 came from. All-`Copy` so the visitor costs nothing per address.
///
/// The path is stored as `(spec, leaf)` rather than as indices, because the walk emits
/// hundreds of these per point and a `Vec<u32>` here would allocate on the hot path.
/// [`PathSpec::leaf_path`] expands it, on the cold path, only for something being
/// reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Location {
    /// Which point of the batch this came from. The walk handles several at once and
    /// only the caller knows how they are numbered, so this indexes back into the slice
    /// it passed in.
    pub batch_index: u32,
    /// Bytes of secret material this was derived from.
    pub material_len: u8,
    pub route: Route,
    /// Index into [`Scope::paths`]. Meaningless for [`Route::PrivKey`], which has no path.
    pub spec: u16,
    /// Index of the leaf within that spec, in walk order.
    pub leaf: u32,
    pub form: HashForm,
}

impl Location {
    /// The BIP32 path, or `None` for a route that has no derivation.
    pub fn path(&self, scope: &Scope) -> Option<String> {
        if !self.route.derives() {
            return None;
        }
        scope.paths.get(self.spec as usize)?.leaf_path(self.leaf as u64)
    }
}

/// What a walk hands its hash160s to.
///
/// The walk produces hashes in chunks, because the only consumer that matters -- a
/// bloom filter far larger than cache -- is far cheaper per hash when it gets a chunk
/// than when it gets one at a time. `select` is where that happens: it sees a whole
/// chunk and says which of them are worth hearing about, and only those come back
/// through `visit`.
///
/// The default `select` keeps everything, so a plain closure is still a complete
/// visitor and `verify` and the tests see every hash the scope derives.
pub trait Visitor {
    /// Given a chunk's hash160s, append the indices worth reporting to `hits`.
    ///
    /// Indices must be appended in increasing order; `visit` is then called for each
    /// one in that order, with the hash at that position.
    fn select(&mut self, hashes: &[[u8; 20]], hits: &mut Vec<u32>) {
        hits.extend(0..hashes.len() as u32);
    }

    fn visit(&mut self, location: &Location, hash: &[u8; 20], phrase: &str);
}

/// A closure as a `Visitor` that keeps the default `select`, i.e. sees every hash the
/// walk derives.
///
/// That is what `verify` and the tests want: there is no filter to consult, or the
/// whole point is to enumerate. The scan implements `Visitor` itself so that it can
/// batch, which is the only reason the trait is not just a closure.
pub struct All<F>(pub F);

impl<F> Visitor for All<F>
where
    F: FnMut(&Location, &[u8; 20], &str),
{
    fn visit(&mut self, location: &Location, hash: &[u8; 20], phrase: &str) {
        (self.0)(location, hash, phrase)
    }
}

/// What to derive for each point of the search space.
#[derive(Clone, Debug)]
pub struct Scope {
    /// Prefix lengths of the secret material to try, in bytes. For BIP39 these are the
    /// entropy sizes: 16, 24 and 32 give 12-, 18- and 24-word phrases.
    pub material_sizes: Vec<usize>,
    pub routes: Vec<Route>,
    pub paths: Vec<PathSpec>,
    pub forms: Vec<HashForm>,
}

impl Default for Scope {
    /// The four shapes the old scanner's `--purposes 44,49,84,none` expanded to.
    fn default() -> Self {
        Self {
            material_sizes: vec![16, 24, 32],
            routes: vec![Route::Bip39, Route::Bip32Seed, Route::PrivKey],
            paths: ["m/44'/0'/0'/{0,1}/{0..9}",
                    "m/49'/0'/0'/{0,1}/{0..9}",
                    "m/84'/0'/0'/{0,1}/{0..9}",
                    "m/{0,1}/{0..9}"]
                .iter()
                .map(|p| PathSpec::parse(p).expect("built-in path"))
                .collect(),
            forms: vec![
                HashForm::Compressed,
                HashForm::Uncompressed,
                HashForm::P2shP2wpkh,
            ],
        }
    }
}

impl Scope {
    /// Check that this scope derives what the user thinks it does.
    ///
    /// Every case here is one that otherwise fails *quietly*: the scan runs to
    /// completion, reports a clean pass, and has derived less than the user asked for --
    /// or the same thing twice. None of them are caught by the type system, so they are
    /// caught here, once, before any work starts.
    pub fn validate(&self) -> Result<(), ScopeError> {
        if self.material_sizes.is_empty() {
            return Err(ScopeError::Empty("material sizes"));
        }
        if self.routes.is_empty() {
            return Err(ScopeError::Empty("routes"));
        }
        if self.forms.is_empty() {
            return Err(ScopeError::Empty("hash forms"));
        }
        // An empty path set is only harmless if nothing walks a tree. With a deriving
        // route in scope it means every BIP39 and BIP32-seed wallet is skipped, which
        // looks exactly like a completed sweep.
        if self.paths.is_empty() && self.tree_routes() > 0 {
            return Err(ScopeError::NoPaths);
        }
        // Two identical specs derive every address twice: double the work, double the
        // candidate count, and one secret written to `matches.txt` under two paths.
        for (i, a) in self.paths.iter().enumerate() {
            if let Some(b) = self.paths[..i].iter().find(|b| *b == a) {
                return Err(ScopeError::DuplicatePath(b.to_string()));
            }
        }
        for (i, a) in self.material_sizes.iter().enumerate() {
            if self.material_sizes[..i].contains(a) {
                return Err(ScopeError::DuplicateSize(*a));
            }
        }
        if self.probes_per_point() == 0 {
            return Err(ScopeError::NothingToProbe);
        }
        Ok(())
    }

    /// Routes that walk a tree -- i.e. everything but [`Route::PrivKey`].
    fn tree_routes(&self) -> usize {
        self.routes.iter().filter(|r| r.derives()).count()
    }

    /// Material sizes that can be a private key on their own. Only 32 bytes is a
    /// scalar, so shorter material contributes nothing to the `PrivKey` route.
    fn privkey_sizes(&self) -> usize {
        if self.routes.contains(&Route::PrivKey) {
            self.material_sizes.iter().filter(|s| **s == 32).count()
        } else {
            0
        }
    }

    /// Public keys derived per point. The dominant cost, and what the ETA is built
    /// from.
    ///
    /// Per spec: a level's parents need a public key only if the segment below them is
    /// normal -- a hardened child is derived from the parent's *private* key and costs
    /// no EC work at all -- and then every leaf needs one for its hash. So it is the
    /// sum of the widths above each normal segment, plus the leaf count.
    pub fn ec_ops_per_point(&self) -> u64 {
        let per_tree: u64 = self
            .paths
            .iter()
            .map(|spec| {
                let mut width = 1u64;
                let mut ops = 0u64;
                for seg in spec.segments() {
                    if !seg.hardened {
                        ops += width;
                    }
                    width *= seg.len() as u64;
                }
                ops + width
            })
            .sum();
        let trees = self.material_sizes.len() as u64 * self.tree_routes() as u64;
        trees * per_tree + self.privkey_sizes() as u64
    }

    /// Filter probes issued per point, i.e. addresses times hash forms.
    pub fn probes_per_point(&self) -> u64 {
        let leaves: u64 = self.paths.iter().map(|s| s.width()).sum();
        let trees = self.material_sizes.len() as u64 * self.tree_routes() as u64;
        (trees * leaves + self.privkey_sizes() as u64) * self.forms.len() as u64
    }

    /// PBKDF2 invocations per point -- one per material size, only if BIP39 is scanned.
    pub fn pbkdf2_per_point(&self) -> u64 {
        if self.routes.contains(&Route::Bip39) {
            self.material_sizes.len() as u64
        } else {
            0
        }
    }
}

/// A scope that would have scanned less than it appears to.
#[derive(Debug, PartialEq, Eq)]
pub enum ScopeError {
    Empty(&'static str),
    NoPaths,
    DuplicatePath(String),
    DuplicateSize(usize),
    NothingToProbe,
}

impl std::fmt::Display for ScopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScopeError::Empty(what) => write!(f, "no {what} selected"),
            ScopeError::NoPaths => f.write_str(
                "no derivation paths selected, but a route that walks a tree is in scope; \
                 every BIP39 and BIP32-seed wallet would be skipped",
            ),
            ScopeError::DuplicatePath(p) => {
                write!(f, "derivation path `{p}` is listed twice; it would be walked twice")
            }
            ScopeError::DuplicateSize(n) => {
                write!(f, "material size {n} is listed twice; it would be walked twice")
            }
            ScopeError::NothingToProbe => f.write_str("this scope derives no addresses"),
        }
    }
}

impl std::error::Error for ScopeError {}

/// A BIP32 extended private key: the scalar and its chain code.
#[derive(Clone, Copy)]
struct Node {
    key: SecretKey,
    chain_code: [u8; 32],
}

/// A node of the derivation tree, plus enough context to name what it produces.
///
/// The walk is breadth-first, so a level is a flat list of these and the tree structure
/// lives only in how one level expands into the next.
#[derive(Clone, Copy)]
struct Cursor {
    node: Node,
    /// Index into the batch's phrase list. Every route reports the BIP39 phrase for its
    /// (point, material size) pair, so phrases are written once and shared.
    phrase: u32,
    /// How many segments of this cursor's spec have been consumed.
    depth: u16,
    /// The leaf index built up so far, mixed-radix, most significant level first. When
    /// the spec runs out this is the final leaf index.
    leaf: u32,
    location: Location,
}

/// How many points are walked together.
///
/// Batching is what makes the two expensive layers cheap. A whole level of the tree
/// shares one field inversion in `crate::crypto::ec`, and the batch's PBKDF2 streams are
/// run `LANES` at a time by `crate::crypto::pbkdf2`. Four points keeps every level
/// comfortably wide while the working set still fits in L1.
pub const POINTS_PER_BATCH: usize = 4;

/// Derivation state reused across batches by one thread.
///
/// Every buffer here exists to keep the per-point path free of allocation; they are
/// swapped and refilled, never reallocated.
#[derive(Default)]
pub struct Deriver {
    phrases: Vec<String>,
    seeds: Vec<[u8; 64]>,
    level: Vec<Cursor>,
    next: Vec<Cursor>,
    leaves: Vec<Cursor>,
    /// Cursors at the current depth whose next segment is normal, and so need a public
    /// key before they can expand. Kept apart from the hardened ones so a level's
    /// inversion covers exactly the cursors that need it.
    normal: Vec<Cursor>,
    scalars: Vec<[u8; 32]>,
    points: Vec<Ge>,
    batch: ec::Batch,
    /// The chunk of leaf hash160s waiting to be offered to the visitor, and where each
    /// of them came from.
    hashes: Vec<[u8; 20]>,
    pending: Vec<(Location, u32)>,
    hits: Vec<u32>,
}

/// The BIP32 master node for some seed material, or `None` if it is not a valid key.
fn master(seed_material: &[u8]) -> Option<Node> {
    split(&HmacSha512::new(b"Bitcoin seed").mac(seed_material))
}

fn split(i: &[u8; 64]) -> Option<Node> {
    let key = SecretKey::from_byte_array(i[..32].try_into().unwrap()).ok()?;
    let mut chain_code = [0u8; 32];
    chain_code.copy_from_slice(&i[32..]);
    Some(Node { key, chain_code })
}

/// CKDpriv for a hardened child. Needs no public key, so it costs no EC work.
fn derive_hardened(parent: &Node, index: u32) -> Option<Node> {
    let mut data = [0u8; 37];
    data[0] = 0;
    data[1..33].copy_from_slice(&parent.key.secret_bytes());
    data[33..].copy_from_slice(&index.to_be_bytes());
    ckd(parent, &data)
}

/// CKDpriv for a normal child, one child at a time.
///
/// The scan does not go through here: `walk_levels` derives every sibling of a parent
/// together and hoists the two things they share out of the loop. This is the unhoisted
/// form, kept because the tests check the tree one node at a time and that is exactly
/// the independent second opinion they are for.
#[cfg(test)]
fn derive_normal(parent: &Node, parent_pub: &Ge, index: u32) -> Option<Node> {
    use crate::wallet::path::HARDENED;
    debug_assert!(index < HARDENED);
    let mut data = [0u8; 37];
    data[..33].copy_from_slice(&parent_pub.serialize());
    data[33..].copy_from_slice(&index.to_be_bytes());
    ckd(parent, &data)
}

fn ckd(parent: &Node, data: &[u8; 37]) -> Option<Node> {
    ckd_keyed(parent, &HmacSha512::new(&parent.chain_code), data)
}

/// The same, for a caller that already has the parent's chain-code HMAC.
///
/// Worth separating because `HmacSha512::new` is not cheap relative to what follows: it
/// absorbs the two 128-byte pad blocks, so it costs two SHA-512 compressions against the
/// two `mac` then spends. The key is the *parent's* chain code, identical for every one
/// of its children, so building it per child makes half of a level's compressions the
/// same two computed ten times over. See `walk_levels`.
fn ckd_keyed(parent: &Node, hmac: &HmacSha512, data: &[u8; 37]) -> Option<Node> {
    let i = hmac.mac(data);
    // child = parse256(IL) + kpar (mod n). Both the "IL >= n" and "child == 0" cases are
    // invalid per BIP32; both are ~2^-127 events, and skipping the child is the right
    // response for a scanner.
    let tweak = Scalar::from_be_bytes(i[..32].try_into().unwrap()).ok()?;
    let key = parent.key.add_tweak(&tweak).ok()?;
    let mut chain_code = [0u8; 32];
    chain_code.copy_from_slice(&i[32..]);
    Some(Node { key, chain_code })
}

impl Deriver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Emit every hash160 the scope asks for, for a batch of secret material.
    ///
    /// `visit` is called once per (address, hash form). It is generic rather than a
    /// trait object so the filter probe inlines into the derivation loop.
    pub fn walk_batch<V: Visitor + ?Sized>(
        &mut self,
        materials: &[[u8; 32]],
        scope: &Scope,
        visit: &mut V,
    ) {
        let sizes = scope.material_sizes.len();
        let mut phrases = std::mem::take(&mut self.phrases);
        let mut level = std::mem::take(&mut self.level);
        let mut leaves = std::mem::take(&mut self.leaves);
        level.clear();
        leaves.clear();

        // One phrase per (point, material size), written before anything else because
        // every route reports it and the BIP39 route hashes it.
        let wanted = materials.len() * sizes;
        while phrases.len() < wanted {
            phrases.push(String::with_capacity(bip39::MAX_PHRASE_LEN));
        }
        for (i, material) in materials.iter().enumerate() {
            for (j, &size) in scope.material_sizes.iter().enumerate() {
                bip39::write_mnemonic(&material[..size], &mut phrases[i * sizes + j]);
            }
        }

        // The batch's PBKDF2 work, run as one job so the streams can be paired.
        if scope.routes.contains(&Route::Bip39) {
            pbkdf2::bip39_seeds(&phrases[..wanted], &mut self.seeds);
        }

        for (i, material) in materials.iter().enumerate() {
            for (j, &size) in scope.material_sizes.iter().enumerate() {
                let phrase = (i * sizes + j) as u32;
                for &route in &scope.routes {
                    let location = Location {
                        batch_index: i as u32,
                        material_len: size as u8,
                        route,
                        spec: 0,
                        leaf: 0,
                        form: HashForm::Compressed,
                    };
                    let node = match route {
                        Route::Bip39 => master(&self.seeds[phrase as usize]),
                        Route::Bip32Seed => master(&material[..size]),
                        // No derivation at all: the material is the key. It is born a
                        // leaf, and rides the leaf level's shared inversion like any
                        // other -- there is no separate list for it.
                        Route::PrivKey if size == 32 => SecretKey::from_byte_array(*material)
                            .ok()
                            .map(|key| Node { key, chain_code: [0u8; 32] }),
                        Route::PrivKey => continue,
                    };
                    let Some(node) = node else { continue };

                    if route.derives() {
                        // One root per spec: the same master node heads every path.
                        for (spec, _) in scope.paths.iter().enumerate() {
                            let mut location = location;
                            location.spec = spec as u16;
                            level.push(Cursor { node, phrase, depth: 0, leaf: 0, location });
                        }
                    } else {
                        leaves.push(Cursor { node, phrase, depth: 0, leaf: 0, location });
                    }
                }
            }
        }

        self.walk_levels(level, leaves, &phrases, scope, visit);
        self.phrases = phrases;
    }

    /// Walk the tree below a BIP39 phrase supplied directly, for `verify`.
    ///
    /// Takes the same path as a scan hit so triage cannot disagree with the scan that
    /// produced it.
    pub fn walk_phrase<V: Visitor + ?Sized>(&mut self, phrase: &str, scope: &Scope, visit: &mut V) {
        // Word count back to material size, purely for reporting.
        let material_len = match phrase.split_whitespace().count() {
            12 => 16,
            18 => 24,
            _ => 32,
        };
        let phrases = vec![phrase.to_string()];
        let mut level = std::mem::take(&mut self.level);
        level.clear();
        if let Some(node) = master(&pbkdf2::bip39_seed(phrase)) {
            for (spec, _) in scope.paths.iter().enumerate() {
                level.push(Cursor {
                    node,
                    phrase: 0,
                    depth: 0,
                    leaf: 0,
                    location: Location {
                        batch_index: 0,
                        material_len,
                        route: Route::Bip39,
                        spec: spec as u16,
                        leaf: 0,
                        form: HashForm::Compressed,
                    },
                });
            }
        }
        self.walk_levels(level, Vec::new(), &phrases, scope, visit);
    }

    /// Expand a level of roots down to leaves, then emit.
    ///
    /// Each level is fully materialised before the next begins, which is the whole
    /// reason this is breadth-first: it lets one call to `crate::crypto::ec` turn a
    /// level's worth of scalars into public keys behind a single field inversion.
    /// Walking depth-first would need one inversion per key.
    fn walk_levels<V: Visitor + ?Sized>(
        &mut self,
        mut level: Vec<Cursor>,
        mut leaves: Vec<Cursor>,
        phrases: &[String],
        scope: &Scope,
        visit: &mut V,
    ) {
        let mut next = std::mem::take(&mut self.next);
        let mut normal = std::mem::take(&mut self.normal);

        while !level.is_empty() {
            next.clear();
            normal.clear();

            // Split the level three ways: finished cursors become leaves, hardened
            // children are derived immediately because they need no public key, and the
            // rest are pooled so one inversion covers all of them.
            for cursor in &level {
                let spec = &scope.paths[cursor.location.spec as usize];
                let Some(seg) = spec.segments().get(cursor.depth as usize) else {
                    let mut done = *cursor;
                    done.location.leaf = done.leaf;
                    leaves.push(done);
                    continue;
                };
                if seg.hardened {
                    for (n, child) in seg.children().enumerate() {
                        if let Some(node) = derive_hardened(&cursor.node, child) {
                            next.push(Cursor {
                                node,
                                phrase: cursor.phrase,
                                depth: cursor.depth + 1,
                                leaf: cursor.leaf * seg.len() as u32 + n as u32,
                                location: cursor.location,
                            });
                        }
                    }
                } else {
                    normal.push(*cursor);
                }
            }

            // Every normal-segment parent in this level, behind one inversion.
            self.public_keys(&normal);
            for (cursor, point) in normal.iter().zip(&self.points) {
                let spec = &scope.paths[cursor.location.spec as usize];
                let seg = &spec.segments()[cursor.depth as usize];

                // Everything a child needs from its parent, built once for all of them:
                // the chain-code HMAC's two pad midstates, and the 37-byte data block
                // with only the trailing index still to fill in. At ten indices per
                // parent this is the difference between four SHA-512 compressions per
                // child and just over two -- see `ckd_keyed`.
                let hmac = HmacSha512::new(&cursor.node.chain_code);
                let mut data = [0u8; 37];
                data[..33].copy_from_slice(&point.serialize());

                for (n, child) in seg.children().enumerate() {
                    data[33..].copy_from_slice(&child.to_be_bytes());
                    let Some(node) = ckd_keyed(&cursor.node, &hmac, &data) else {
                        continue;
                    };
                    next.push(Cursor {
                        node,
                        phrase: cursor.phrase,
                        depth: cursor.depth + 1,
                        leaf: cursor.leaf * seg.len() as u32 + n as u32,
                        location: cursor.location,
                    });
                }
            }
            std::mem::swap(&mut level, &mut next);
        }

        // The leaves, in chunks. `Visitor::select` is where a bloom filter gets to
        // overlap its memory misses, and it can only do that with several hashes in
        // hand, so hashes accumulate until a chunk is full rather than going out one at
        // a time.
        let mut hashes = std::mem::take(&mut self.hashes);
        let mut pending = std::mem::take(&mut self.pending);
        let mut hits = std::mem::take(&mut self.hits);
        hashes.clear();
        pending.clear();

        self.public_keys(&leaves);
        for (cursor, point) in leaves.iter().zip(&self.points) {
            emit_forms(point, scope, &cursor.location, cursor.phrase, &mut hashes, &mut pending);
            if hashes.len() + scope.forms.len() > PROBE_CHUNK {
                flush(&mut hashes, &mut pending, &mut hits, phrases, visit);
            }
        }
        flush(&mut hashes, &mut pending, &mut hits, phrases, visit);

        self.level = level;
        self.next = next;
        self.leaves = leaves;
        self.normal = normal;
        self.hashes = hashes;
        self.pending = pending;
        self.hits = hits;
    }

    /// Public keys for a whole level, behind one field inversion.
    fn public_keys(&mut self, level: &[Cursor]) {
        self.scalars.clear();
        self.scalars
            .extend(level.iter().map(|c| c.node.key.secret_bytes()));
        self.batch.public_keys(&self.scalars, &mut self.points);
    }
}

/// The private key of one leaf, re-derived from the material that produced it.
///
/// This is the **cold path**: it walks a single path one node at a time, with no shared
/// inversion and no hoisted HMAC, and it allocates. That is fine, because the only
/// caller is the match sink, and a match happens roughly once in a billion points. Doing
/// it this way keeps the hot loop from carrying a 32-byte key through every chunk for
/// the sake of the one record in a billion that needs it.
///
/// Returns `None` for material a route cannot use -- a `PrivKey` route on anything but
/// 32 bytes, or the ~2^-127 cases where a derivation step lands outside the curve order.
pub fn leaf_private_key(
    material: &[u8],
    route: Route,
    spec: Option<&PathSpec>,
    leaf: u64,
) -> Option<[u8; 32]> {
    let node = match route {
        Route::PrivKey => {
            let bytes: [u8; 32] = material.try_into().ok()?;
            return SecretKey::from_byte_array(bytes).ok().map(|k| k.secret_bytes());
        }
        Route::Bip39 => master(&pbkdf2::bip39_seed(&bip39::mnemonic(material)))?,
        Route::Bip32Seed => master(material)?,
    };

    let mut node = node;
    for child in spec?.leaf(leaf)? {
        node = if child & crate::wallet::path::HARDENED != 0 {
            derive_hardened(&node, child)?
        } else {
            let public = ec::public_key(&node.key.secret_bytes());
            let mut data = [0u8; 37];
            data[..33].copy_from_slice(&public.serialize());
            data[33..].copy_from_slice(&child.to_be_bytes());
            ckd(&node, &data)?
        };
    }
    Some(node.key.secret_bytes())
}

/// Append every requested hash form for one public key to the pending chunk.
#[inline]
fn emit_forms(
    public: &Ge,
    scope: &Scope,
    location: &Location,
    phrase: u32,
    hashes: &mut Vec<[u8; 20]>,
    pending: &mut Vec<(Location, u32)>,
) {
    // Two of the three forms are built from the compressed hash, so it is computed once
    // and shared -- but only if the scope actually asks for one of them. A sweep
    // restricted to `--hash-forms uncompressed` would otherwise pay a SHA-256 and a
    // RIPEMD-160 per key for a value it never looks at.
    let mut compressed = None;
    let mut compressed_hash = || *compressed.get_or_insert_with(|| hash160(&public.serialize()));

    for &form in &scope.forms {
        let hash = match form {
            HashForm::Compressed => compressed_hash(),
            HashForm::Uncompressed => hash160(&public.serialize_uncompressed()),
            HashForm::P2shP2wpkh => {
                let mut script = [0u8; 22];
                script[0] = 0x00; // OP_0
                script[1] = 0x14; // push 20 bytes
                script[2..].copy_from_slice(&compressed_hash());
                hash160(&script)
            }
        };
        let mut here = *location;
        here.form = form;
        hashes.push(hash);
        pending.push((here, phrase));
    }
}

/// Offer a full chunk to the visitor and report back whatever it selects.
fn flush<V: Visitor + ?Sized>(
    hashes: &mut Vec<[u8; 20]>,
    pending: &mut Vec<(Location, u32)>,
    hits: &mut Vec<u32>,
    phrases: &[String],
    visit: &mut V,
) {
    if hashes.is_empty() {
        return;
    }
    hits.clear();
    visit.select(hashes, hits);
    for &i in hits.iter() {
        let (location, phrase) = pending[i as usize];
        visit.visit(&location, &hashes[i as usize], &phrases[phrase as usize]);
    }
    hashes.clear();
    pending.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::pbkdf2::bip39_seed;
    use crate::vuln::mt19937::entropy_for_seed;
    use crate::wallet::path::HARDENED;
    use bitcoin::bip32::{DerivationPath, Xpriv};
    use bitcoin::hashes::Hash;
    use bitcoin::network::Network;
    use bitcoin::secp256k1::Secp256k1 as RefSecp;

    /// Derive one leaf a node at a time, the unhoisted way.
    ///
    /// This is deliberately *not* how `walk_levels` does it -- no shared inversion, no
    /// hoisted parent HMAC, one path walked to its end. That is the point: it is the
    /// independent second opinion the batched walk is checked against.
    fn derive_leaf(master: &Node, spec: &PathSpec, leaf: u64) -> Option<Node> {
        let mut node = *master;
        for child in spec.leaf(leaf)? {
            node = if child & HARDENED != 0 {
                derive_hardened(&node, child)?
            } else {
                let pk = ec::public_key(&node.key.secret_bytes());
                derive_normal(&node, &pk, child)?
            };
        }
        Some(node)
    }

    /// BIP32 test vector 1: the published chain from a known 16-byte seed, which pins
    /// master generation and both hardened and normal CKDpriv independently of any
    /// other library.
    #[test]
    fn matches_the_published_bip32_vectors() {
        let seed = hex::decode("000102030405060708090a0b0c0d0e0f").unwrap();
        let master = master(&seed).unwrap();
        assert_eq!(
            hex::encode(master.key.secret_bytes()),
            "e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35"
        );
        assert_eq!(
            hex::encode(master.chain_code),
            "873dff81c02f525623fd1fe5167eac3a55a049de3d314bb42ee227ffed37d508"
        );

        // m/0'
        let h0 = derive_hardened(&master, HARDENED).unwrap();
        assert_eq!(
            hex::encode(h0.key.secret_bytes()),
            "edb2e14f9ee77d26dd93b4ecede8d16ed408ce149b6cd80b0715a2d911a0afea"
        );

        // m/0'/1
        let pk = ec::public_key(&h0.key.secret_bytes());
        let n1 = derive_normal(&h0, &pk, 1).unwrap();
        assert_eq!(
            hex::encode(n1.key.secret_bytes()),
            "3c6cb8d0f6a264c91ea8b5030fadaa8e538b020f0a387421a12de9319dc93368"
        );

        // And the same leaf reached through the path grammar, which is the form the
        // scan actually walks.
        let spec = PathSpec::parse("m/0'/1").unwrap();
        let viaspec = derive_leaf(&master, &spec, 0).unwrap();
        assert_eq!(viaspec.key.secret_bytes(), n1.key.secret_bytes());
    }

    /// Every path in the default scope, held against the `bitcoin` crate's BIP32.
    ///
    /// This now exercises the path grammar as well: the path string handed to the
    /// reference implementation is rendered by `PathSpec::leaf_path`, so a grammar that
    /// enumerated leaves in a different order than it named them would fail here.
    #[test]
    fn agrees_with_the_reference_bip32_over_the_default_scope() {
        let scope = Scope::default();
        let ref_secp = RefSecp::new();

        for seed in 0..8u32 {
            let entropy = entropy_for_seed(seed);
            for &size in &scope.material_sizes {
                let phrase = bip39::mnemonic(&entropy[..size]);
                let bip39_material = bip39_seed(&phrase);

                let materials: [(Route, &[u8]); 2] = [
                    (Route::Bip39, &bip39_material),
                    (Route::Bip32Seed, &entropy[..size]),
                ];
                for (route, material) in materials {
                    let ours_master = master(material).unwrap();
                    let theirs_master = Xpriv::new_master(Network::Bitcoin, material).unwrap();
                    assert_eq!(
                        ours_master.key.secret_bytes(),
                        theirs_master.private_key.secret_bytes(),
                        "master mismatch, seed {seed} size {size} {route:?}"
                    );

                    for spec in &scope.paths {
                        for leaf in 0..spec.width() {
                            let path = spec.leaf_path(leaf).unwrap();
                            let theirs = theirs_master
                                .derive_priv(
                                    &ref_secp,
                                    &path.parse::<DerivationPath>().unwrap(),
                                )
                                .unwrap();
                            let ours = derive_leaf(&ours_master, spec, leaf).unwrap();

                            assert_eq!(
                                ours.key.secret_bytes(),
                                theirs.private_key.secret_bytes(),
                                "{path} mismatch, seed {seed} size {size} {route:?}"
                            );
                            assert_eq!(
                                ours.chain_code.as_slice(),
                                AsRef::<[u8]>::as_ref(&theirs.chain_code),
                                "{path} chain code mismatch"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Paths that interleave hardened and normal levels, against the reference BIP32.
    ///
    /// The default scope is all one shape -- a hardened prefix, then normal levels -- so
    /// it cannot catch a walk that handles the two kinds in the wrong order. The path
    /// grammar makes other shapes expressible for the first time, and `walk_levels`
    /// splits each level into hardened and normal cursors, so a hardened segment sitting
    /// *below* a normal one is exactly the case that generalization could get wrong.
    #[test]
    fn agrees_with_the_reference_bip32_on_interleaved_hardened_levels() {
        let ref_secp = RefSecp::new();
        let seed = b"keyforge interleaved path vector";
        let ours_master = master(seed).unwrap();
        let theirs_master = Xpriv::new_master(Network::Bitcoin, seed).unwrap();

        for text in [
            "m",                       // no derivation at all
            "m/0",                     // one normal level
            "m/0'",                    // one hardened level
            "m/0/1'",                  // hardened below normal
            "m/0'/1/2'/3",             // alternating, four deep
            "m/{0,1}/{2,3}'",          // sets on both, hardened below normal
            "m/44'/0'/0'/{0,1}/{0..3}",// the familiar shape, for contrast
            "m/2147483647'/0/1",       // the largest hardenable index
        ] {
            let spec = PathSpec::parse(text).unwrap();
            for leaf in 0..spec.width() {
                let path = spec.leaf_path(leaf).unwrap();
                let theirs = theirs_master
                    .derive_priv(&ref_secp, &path.parse::<DerivationPath>().unwrap())
                    .unwrap();
                let ours = derive_leaf(&ours_master, &spec, leaf).unwrap();
                assert_eq!(
                    ours.key.secret_bytes(),
                    theirs.private_key.secret_bytes(),
                    "{path} (from `{text}`) disagrees with the reference"
                );
            }
        }
    }

    /// The same shapes through the *batched* walk, which is where the hardened/normal
    /// split actually happens. `derive_leaf` above walks one node at a time and would
    /// not exercise it.
    #[test]
    fn the_batched_walk_handles_interleaved_hardened_levels() {
        let ref_secp = RefSecp::new();
        let entropy = entropy_for_seed(3);

        for text in ["m/0/1'", "m/0'/1/2'/3", "m/{0,1}/{2,3}'", "m"] {
            let spec = PathSpec::parse(text).unwrap();
            let scope = Scope {
                material_sizes: vec![32],
                routes: vec![Route::Bip32Seed],
                paths: vec![spec.clone()],
                forms: vec![HashForm::Compressed],
            };
            assert_eq!(scope.validate(), Ok(()));

            let mut got: Vec<(String, String)> = Vec::new();
            Deriver::new().walk_batch(
                std::slice::from_ref(&entropy),
                &scope,
                &mut All(|loc: &Location, hash: &[u8; 20], _: &str| {
                    got.push((loc.path(&scope).unwrap(), hex::encode(hash)));
                }),
            );

            let theirs_master = Xpriv::new_master(Network::Bitcoin, &entropy).unwrap();
            let mut want: Vec<(String, String)> = (0..spec.width())
                .map(|leaf| {
                    let path = spec.leaf_path(leaf).unwrap();
                    let child = theirs_master
                        .derive_priv(&ref_secp, &path.parse::<DerivationPath>().unwrap())
                        .unwrap();
                    let pk = child.private_key.public_key(&ref_secp);
                    (path, hex::encode(hash160(&pk.serialize())))
                })
                .collect();

            got.sort();
            want.sort();
            assert_eq!(got, want, "the batched walk disagrees for `{text}`");
        }
    }

    /// The three hash160 forms must equal what the reference address types commit to.
    #[test]
    fn hash_forms_match_the_reference_address_types() {
        use bitcoin::{Address, CompressedPublicKey, PublicKey as RefPublicKey};

        let master = master(b"keyforge hash form vector").unwrap();
        let public = ec::public_key(&master.key.secret_bytes());

        let ref_pk = RefPublicKey::from_slice(&public.serialize()).unwrap();
        let ref_compressed = CompressedPublicKey::from_slice(&public.serialize()).unwrap();

        // P2PKH compressed and P2WPKH commit to the same 20 bytes.
        let compressed = hash160(&public.serialize());
        assert_eq!(compressed, ref_pk.pubkey_hash().to_byte_array());
        assert_eq!(compressed, ref_compressed.wpubkey_hash().to_byte_array());

        // Uncompressed P2PKH. Built from the uncompressed serialisation, which is also
        // how it crosses the two secp256k1 major versions in this dependency graph --
        // the reference crate pulls in its own.
        let uncompressed_pk = RefPublicKey::from_slice(&public.serialize_uncompressed()).unwrap();
        assert!(!uncompressed_pk.compressed);
        assert_eq!(
            hash160(&public.serialize_uncompressed()),
            uncompressed_pk.pubkey_hash().to_byte_array()
        );

        // P2SH-P2WPKH: the hash160 of the wrapped-segwit redeem script.
        let mut script = [0u8; 22];
        script[0] = 0x00;
        script[1] = 0x14;
        script[2..].copy_from_slice(&compressed);
        let p2sh = hash160(&script);
        let ref_addr = Address::p2shwpkh(&ref_compressed, Network::Bitcoin);
        assert!(
            ref_addr.to_string().starts_with('3'),
            "expected a wrapped-segwit address"
        );
        assert_eq!(
            hex::encode(p2sh),
            hex::encode(&ref_addr.script_pubkey().as_bytes()[2..22])
        );
    }

    /// The whole walk, end to end: the visitor must see exactly the hashes the
    /// reference implementation produces for the same scope.
    #[test]
    fn the_walk_visits_exactly_the_expected_addresses() {
        let mut d = Deriver::new();
        let scope = Scope::default();
        let entropy = entropy_for_seed(42);

        let mut seen: Vec<(String, String)> = Vec::new();
        d.walk_batch(
            std::slice::from_ref(&entropy),
            &scope,
            &mut All(|loc: &Location, hash: &[u8; 20], _phrase: &str| {
                seen.push((
                    format!(
                        "{}/{}/{}/{}",
                        loc.material_len,
                        loc.route.as_str(),
                        loc.path(&scope).unwrap_or_else(|| "-".into()),
                        loc.form.as_str()
                    ),
                    hex::encode(hash),
                ));
            }),
        );

        assert_eq!(seen.len() as u64, scope.probes_per_point());

        // Every entry must be distinct: a bug that reuses one node for every index
        // would otherwise pass silently.
        let unique: std::collections::HashSet<_> = seen.iter().map(|(_, h)| h).collect();
        assert_eq!(unique.len(), seen.len(), "duplicate hashes in the walk");

        // Spot-check one address against the reference implementation.
        let ref_secp = RefSecp::new();
        let phrase = bip39::mnemonic(&entropy[..16]);
        let master = Xpriv::new_master(Network::Bitcoin, &bip39_seed(&phrase)).unwrap();
        let child = master
            .derive_priv(
                &ref_secp,
                &"m/84'/0'/0'/0/3".parse::<DerivationPath>().unwrap(),
            )
            .unwrap();
        let pk = child.private_key.public_key(&ref_secp);
        let expected = hex::encode(hash160(&pk.serialize()));
        assert!(
            seen.contains(&("16/bip39/m/84'/0'/0'/0/3/compressed".to_string(), expected)),
            "the walk did not produce the reference address for m/84'/0'/0'/0/3"
        );
    }

    /// The batched walk and the one-at-a-time walk must agree, address for address.
    ///
    /// This is the test that guards the two optimisations `walk_levels` exists for --
    /// the shared field inversion and the hoisted parent HMAC. Both produce the right
    /// answer only if a level's cursors are matched to their own spec and depth, which
    /// is exactly what got harder when levels stopped being homogeneous.
    #[test]
    fn the_batched_walk_agrees_with_walking_one_path_at_a_time() {
        let scope = Scope::default();
        let entropy = entropy_for_seed(7);

        let mut single: Vec<(String, String)> = Vec::new();
        for &size in &scope.material_sizes {
            let phrase = bip39::mnemonic(&entropy[..size]);
            for &route in &scope.routes {
                let node = match route {
                    Route::Bip39 => master(&bip39_seed(&phrase)),
                    Route::Bip32Seed => master(&entropy[..size]),
                    Route::PrivKey if size == 32 => SecretKey::from_byte_array(entropy)
                        .ok()
                        .map(|key| Node { key, chain_code: [0u8; 32] }),
                    Route::PrivKey => continue,
                };
                let Some(node) = node else { continue };
                if !route.derives() {
                    let pk = ec::public_key(&node.key.secret_bytes());
                    single.push((
                        format!("{size}/{}/-", route.as_str()),
                        hex::encode(hash160(&pk.serialize())),
                    ));
                    continue;
                }
                for spec in &scope.paths {
                    for leaf in 0..spec.width() {
                        let child = derive_leaf(&node, spec, leaf).unwrap();
                        let pk = ec::public_key(&child.key.secret_bytes());
                        single.push((
                            format!("{size}/{}/{}", route.as_str(), spec.leaf_path(leaf).unwrap()),
                            hex::encode(hash160(&pk.serialize())),
                        ));
                    }
                }
            }
        }

        // One form only, so each address appears once and the two lists line up.
        // Compared as sets: the batched walk emits leaves grouped by the depth at which
        // their spec ran out, which is a different order and not part of the contract.
        let mut batched: Vec<(String, String)> = Vec::new();
        Deriver::new().walk_batch(
            std::slice::from_ref(&entropy),
            &Scope { forms: vec![HashForm::Compressed], ..scope.clone() },
            &mut All(|loc: &Location, hash: &[u8; 20], _: &str| {
                batched.push((
                    format!("{}/{}/{}", loc.material_len, loc.route.as_str(),
                            loc.path(&scope).unwrap_or_else(|| "-".into())),
                    hex::encode(hash),
                ));
            }),
        );
        batched.sort();
        single.sort();
        assert_eq!(batched.len(), single.len(), "the two walks derived different counts");
        assert_eq!(batched, single);
    }

    /// The accounting the ETA relies on must match what the walk actually does.
    ///
    /// The numbers are the ones the old scanner's README recorded for the same scope --
    /// 1,443 probes, 553 EC ops, 3 PBKDF2 per seed -- which makes this an equivalence
    /// check against the tool this replaces as well as an internal one.
    #[test]
    fn cost_accounting_matches_the_walk() {
        let scope = Scope::default();
        assert_eq!(scope.probes_per_point(), 1443);
        assert_eq!(scope.ec_ops_per_point(), 553);
        assert_eq!(scope.pbkdf2_per_point(), 3);

        // Spelled out as the formula, so the assertion reads as what it checks. The
        // lone `+ 1` in each is the privkey route, which has no derivation path.
        assert_eq!(scope.probes_per_point(), (3 * 2 * (4 * 20) + 1) * 3);
        assert_eq!(scope.ec_ops_per_point(), 3 * 2 * (4 * (1 + 2 * (1 + 10))) + 1);

        // And the walk actually emits that many.
        let mut n = 0u64;
        Deriver::new().walk_batch(
            &[entropy_for_seed(1)],
            &scope,
            &mut All(|_: &Location, _: &[u8; 20], _: &str| n += 1),
        );
        assert_eq!(n, scope.probes_per_point());
    }

    /// Every way a scope can quietly scan less than it looks like it does.
    ///
    /// All of these ran to completion before `validate` existed. The dangerous ones are
    /// the empty path set -- which skips every BIP39 and BIP32-seed wallet while looking
    /// like a finished sweep -- and the duplicated spec, which derives every address
    /// twice and inflates the candidate count.
    #[test]
    fn validate_rejects_scopes_that_would_quietly_scan_less() {
        let base = Scope::default();
        assert_eq!(base.validate(), Ok(()));

        let empty_paths = Scope { paths: vec![], ..base.clone() };
        assert_eq!(empty_paths.validate(), Err(ScopeError::NoPaths));

        // ...unless nothing walks a tree, in which case there is nothing to skip.
        let privkey_only = Scope {
            paths: vec![],
            routes: vec![Route::PrivKey],
            material_sizes: vec![32],
            ..base.clone()
        };
        assert_eq!(privkey_only.validate(), Ok(()));

        let dup = Scope {
            paths: vec![
                PathSpec::parse("m/0").unwrap(),
                PathSpec::parse("m/0").unwrap(),
            ],
            ..base.clone()
        };
        assert!(matches!(dup.validate(), Err(ScopeError::DuplicatePath(_))));

        let dup_size = Scope { material_sizes: vec![16, 16], ..base.clone() };
        assert_eq!(dup_size.validate(), Err(ScopeError::DuplicateSize(16)));

        for empty in [
            Scope { material_sizes: vec![], ..base.clone() },
            Scope { routes: vec![], ..base.clone() },
            Scope { forms: vec![], ..base.clone() },
        ] {
            assert!(matches!(empty.validate(), Err(ScopeError::Empty(_))));
        }
    }

    /// A duplicated spec is not merely wasteful -- it emits the same address twice, so a
    /// hit would be written under two paths and counted twice. This is the behaviour
    /// `validate` exists to prevent, pinned here so the cost is on record.
    #[test]
    fn a_duplicated_path_would_derive_every_address_twice() {
        let scope = Scope {
            paths: vec![
                PathSpec::parse("m/0").unwrap(),
                PathSpec::parse("m/0").unwrap(),
            ],
            routes: vec![Route::Bip39],
            material_sizes: vec![16],
            forms: vec![HashForm::Compressed],
        };
        let mut emitted = 0usize;
        let mut distinct = std::collections::HashSet::new();
        Deriver::new().walk_batch(
            &[[7u8; 32]],
            &scope,
            &mut All(|_: &Location, h: &[u8; 20], _: &str| {
                emitted += 1;
                distinct.insert(*h);
            }),
        );
        assert_eq!(emitted, 2);
        assert_eq!(distinct.len(), 1, "the duplicate derived a different address");
        assert!(scope.validate().is_err(), "validate should have caught this");
    }

    /// Every registered vulnerability's default scope must pass validation, or the tool
    /// ships a preset that scans less than it claims.
    #[test]
    fn every_vulnerability_default_scope_validates() {
        for v in crate::vuln::registry() {
            let mut scope = Scope::default();
            v.defaults().apply(&mut scope);
            assert_eq!(scope.validate(), Ok(()), "{} has an invalid default scope", v.id());
        }
    }

    /// The four canary wallets the Milk Sad team funded and watched get emptied, from
    /// research update 3. These are the strongest end-to-end oracle available: real
    /// published (seed, path, address) triples, produced by `bx` itself rather than by
    /// anyone's reimplementation, and confirmed on-chain by the theft.
    ///
    /// All four are the 24-word BIP39 route. Two sit at `m/0/0`, which is why the bare
    /// `m/{0,1}/{0..9}` spec is in the default scope -- without it the scan walks
    /// straight past half of the only wallets anyone has published ground truth for.
    #[test]
    fn reproduces_the_published_canary_wallets() {
        let canaries = [
            (500u32, "m/44'/0'/0'/0/0", "13KqxkrmsPKy8gyYwochCQTuPHC7Lp8bFU"),
            (500, "m/0/0", "1NxkqwmsQMTqv4SrggPv4vGHDzJKR52S2f"),
            (u32::MAX, "m/44'/0'/0'/0/0", "1HQR3nKaDahAFrPHMoDVdWiMNFGFb7cHA5"),
            (u32::MAX, "m/0/0", "16pQhPkBa5puwEzudZVyKtsrugLtA87cy"),
        ];

        let scope = Scope::default();
        let mut deriver = Deriver::new();
        for (seed, path, address) in canaries {
            let mut found = false;
            deriver.walk_batch(
                &[entropy_for_seed(seed)],
                &scope,
                &mut All(|location: &Location, hash: &[u8; 20], _: &str| {
                    found |= location.material_len == 32
                        && location.route == Route::Bip39
                        && location.form == HashForm::Compressed
                        && location.path(&scope).as_deref() == Some(path)
                        // The compressed form encodes as "P2PKH / P2WPKH"; the published
                        // canaries are the P2PKH half.
                        && crate::wallet::address::encode(location.form, hash)
                            .split(' ')
                            .next()
                            == Some(address);
                }),
            );
            assert!(found, "seed {seed} did not produce {address} at {path}");
        }
    }
}
