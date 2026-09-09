//! How a `Scope` becomes buffer sizes, and how a leaf slot names the point it came from.
//!
//! The GPU walks the same trees as `derive::walk_levels`, but where the CPU builds each
//! level as a `Vec` that *compacts* -- an invalid node is simply not pushed -- the GPU
//! keeps every level dense and marks the dead slots. That is not a stylistic difference: a
//! GPU thread's index is its position, so a level has to have a fixed shape before the
//! kernel launches. Dense-and-marked is the only form that gives one.
//!
//! # Regions
//!
//! The scanner this grew out of walked exactly one tree shape, so its leaf array was a
//! single mixed-radix number and a leaf slot could be unwound into a full `Location`. A
//! path *set* has no such shape: `m/44'/0'/0'/{0,1}/{0..9}` and `m/{0,1}/{0..9}` reach
//! their leaves at different depths and contribute different numbers of them.
//!
//! So the leaf array is a run of **regions**, one per path spec plus one for keys that have
//! no path at all, each holding `per_point` leaves for every point of the launch:
//!
//! ```text
//!   [ spec 0 ........ ][ spec 1 ........ ]...[ no path ]
//!     point 0, point 1, ...                    point 0, ...
//! ```
//!
//! # What a slot has to yield, and what it does not
//!
//! Only the **point**. The host takes a device record, re-derives that whole point through
//! `derive::Deriver`, and it is that walk which produces every phrase, key and path a user
//! ever sees. A wrong answer here therefore cannot put a wrong secret in the output file;
//! it can only stop the host confirming a record, which it counts and reports.
//!
//! That is why this module no longer reconstructs a `Location`. The old one did, through a
//! six-deep mixed-radix unwind that had already produced one bug where two leaves decoded
//! to the same place -- carried for a value that was then thrown away. Dividing within a
//! region is the whole of what is needed, and it is hard to get subtly wrong.
//!
//! Nothing here needs a GPU and the tests do not use one. That is deliberate: this is the
//! piece most likely to be quietly wrong, and it should be falsifiable on any machine.

use crate::scan::derive::{Route, Scope};
use crate::wallet::address::HashForm;
use crate::wallet::path::PathSpec;

/// One contiguous run of the leaf array.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    /// The path spec these leaves came from, or `None` for keys with no derivation.
    pub spec: Option<usize>,
    /// Leaves this region holds for each point of a launch.
    pub per_point: usize,
    /// Where this region starts, per point of the launch. Multiply by the launch size for
    /// the slot itself -- the boundary moves with the launch, and conflating the two with
    /// the capacity is exactly how a short launch decodes to the wrong point.
    pub base_per_point: usize,
}

/// One round of the lockstep walk: every spec that acts at this depth.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Round {
    /// Nodes per point this round's parents occupy in the level buffer.
    pub width: usize,
    /// Nodes per point at the front of the buffer that need public keys -- the specs
    /// taking a normal step. Zero when every step this round is hardened.
    pub normal_width: usize,
    pub steps: Vec<Step>,
}

/// One spec's move within a round.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    /// Index into [`Layout::specs`].
    pub spec: usize,
    /// Which of that spec's segments this walks.
    pub segment: usize,
    pub hardened: bool,
    /// Parent nodes per point this step reads.
    pub width: usize,
    /// Whether this is the spec's first round, so its parents are still the master nodes
    /// and have to be copied into the shared level buffer before it can walk with them.
    pub joins: bool,
    /// Whether this is the spec's last segment, whose children are leaves.
    pub last: bool,
    /// Where this step's parents sit in the level buffer, per point.
    pub at: usize,
}

/// The shape of one launch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    /// The largest launch this was sized for, which is what the buffers are allocated
    /// against. **Not** the size of any particular launch: the last launch of a range is
    /// usually short, and the methods that depend on it take it as an argument.
    pub capacity: usize,
    pub material_sizes: Vec<usize>,
    /// The routes that derive a tree, in scope order. `PrivKey` is excluded: it has no
    /// derivation and joins at the leaf level instead.
    pub tree_routes: Vec<Route>,
    /// Material sizes that can be a private key on their own -- only 32 bytes is a scalar.
    pub raw_sizes: Vec<usize>,
    pub specs: Vec<PathSpec>,
    pub forms: Vec<HashForm>,
    pub regions: Vec<Region>,
}

impl Layout {
    pub fn new(scope: &Scope, capacity: usize) -> Self {
        let tree_routes: Vec<Route> =
            scope.routes.iter().copied().filter(|r| r.derives()).collect();
        let raw_sizes: Vec<usize> = if scope.routes.contains(&Route::PrivKey) {
            scope.material_sizes.iter().copied().filter(|s| *s == 32).collect()
        } else {
            Vec::new()
        };
        let trees = scope.material_sizes.len() * tree_routes.len();

        let mut regions = Vec::new();
        let mut base = 0usize;
        for (i, spec) in scope.specs_or_empty(&tree_routes).iter().enumerate() {
            let per_point = trees * spec.width() as usize;
            if per_point == 0 {
                continue;
            }
            regions.push(Region { spec: Some(i), per_point, base_per_point: base });
            base += per_point;
        }
        if !raw_sizes.is_empty() {
            regions.push(Region {
                spec: None,
                per_point: raw_sizes.len(),
                base_per_point: base,
            });
        }

        Self {
            capacity,
            material_sizes: scope.material_sizes.clone(),
            tree_routes,
            raw_sizes,
            specs: scope.paths.clone(),
            forms: scope.forms.clone(),
            regions,
        }
    }

    /// Master nodes per point: one per (material size, tree route).
    pub fn masters_per_point(&self) -> usize {
        self.material_sizes.len() * self.tree_routes.len()
    }

    /// Every leaf key per point, across all regions.
    pub fn leaves_per_point(&self) -> usize {
        self.regions.iter().map(|r| r.per_point).sum()
    }

    /// Hashes per point: one per (leaf, form). This is `Scope::probes_per_point`, and the
    /// test below holds the two to each other.
    pub fn hashes_per_point(&self) -> usize {
        self.leaves_per_point() * self.forms.len()
    }

    pub fn leaves(&self, points: usize) -> usize {
        points * self.leaves_per_point()
    }

    /// The widest level the shared buffer has to hold, per point.
    ///
    /// The **last** segment of a spec writes straight into the leaf array rather than into
    /// a level buffer, so its output does not count here -- which is the widest level of
    /// all, and including it sized both buffers about ten times too large on the default
    /// scope. That is memory taken directly out of the batch on the very device where
    /// memory is what limits it.
    ///
    /// It is a *sum* across specs rather than a maximum, because the specs walk in
    /// lockstep and share one buffer -- see [`rounds`](Self::rounds) for why.
    pub fn level_capacity(&self, points: usize) -> usize {
        self.rounds().iter().map(|r| r.width).max().unwrap_or(0) * points
    }

    /// The walk, one round at a time, right-aligned so every spec's last segment falls on
    /// the last round.
    ///
    /// The specs used to be walked one after another, each ping-ponging through the level
    /// buffers on its own. That made every normal segment its own `public_keys` -- a k*G
    /// and a batched inversion, six dispatches -- so the default scope issued eight of
    /// them per launch where one shared walk needs two. Dispatches are not free and these
    /// were small, and it measured as a 7% loss against the scanner this grew out of,
    /// which fused its four fixed purposes into one level by construction.
    ///
    /// Right-alignment is what makes sharing possible for specs of *different depths*:
    /// a spec with `n` segments simply starts `maxdepth - n` rounds in, so
    /// `m/44'/0'/0'/{0,1}/{0..9}` and `m/{0,1}/{0..9}` reach their chain and index levels
    /// on the same rounds. Aligning from the front would leave the bare spec at its leaf
    /// level while the others were still in hardened prefix, sharing nothing.
    ///
    /// Within a round the specs taking a **normal** step are placed first, so one
    /// `public_keys` over the prefix `[0, normal_width)` serves all of them and each
    /// spec's public key sits at the same offset as its parent node.
    pub fn rounds(&self) -> Vec<Round> {
        let trees = self.masters_per_point();
        if trees == 0 {
            return Vec::new();
        }
        // A spec with no segments at all -- `m`, the master node itself -- has no round to
        // take: its masters are its leaves and are copied straight across.
        let walked: Vec<(usize, &PathSpec)> = self
            .specs
            .iter()
            .enumerate()
            .filter(|(_, spec)| !spec.segments().is_empty())
            .collect();
        let depth = walked.iter().map(|(_, s)| s.segments().len()).max().unwrap_or(0);

        let mut out = Vec::new();
        for round in 0..depth {
            let mut steps: Vec<Step> = Vec::new();
            for (spec, path) in &walked {
                let segments = path.segments();
                // Right-aligned: this spec joins once its remaining segments fit.
                let Some(seg) = (round + segments.len()).checked_sub(depth) else {
                    continue;
                };
                if seg >= segments.len() {
                    continue;
                }
                // Nodes per point at the start of this segment.
                let width: usize =
                    trees * segments[..seg].iter().map(|s| s.len()).product::<usize>();
                steps.push(Step {
                    spec: *spec,
                    segment: seg,
                    hardened: segments[seg].hardened,
                    width,
                    joins: seg == 0,
                    last: seg + 1 == segments.len(),
                    at: 0,
                });
            }
            // Normal steps first, so `public_keys` covers a prefix. Stable within each
            // group, so a spec's placement is a function of the scope and nothing else.
            steps.sort_by_key(|s| (s.hardened, s.spec));
            let mut at = 0usize;
            let mut normal = 0usize;
            for step in &mut steps {
                step.at = at;
                at += step.width;
                if !step.hardened {
                    normal = at;
                }
            }
            out.push(Round { width: at, normal_width: normal, steps });
        }
        out
    }

    /// Which point a leaf slot belongs to, for a launch of `points` points.
    ///
    /// `None` for a slot outside the launch rather than a panic or a wrapped answer: the
    /// value arrives from device memory, and a corrupt one must surface as a counted
    /// anomaly rather than take the scan down.
    pub fn point_of(&self, points: usize, leaf: usize) -> Option<usize> {
        for region in &self.regions {
            let lo = region.base_per_point * points;
            let hi = lo + region.per_point * points;
            if leaf >= lo && leaf < hi {
                return Some((leaf - lo) / region.per_point);
            }
        }
        None
    }
}

impl Scope {
    /// The path specs that are actually walked: none at all when nothing derives a tree.
    ///
    /// A privkey-only scope carries the default path set and never touches it. Sizing
    /// regions for those paths would allocate buffers for a tree the launch never builds.
    pub(crate) fn specs_or_empty(&self, tree_routes: &[Route]) -> Vec<PathSpec> {
        if tree_routes.is_empty() {
            Vec::new()
        } else {
            self.paths.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every leaf slot of a launch must name the point that produced it, and every point
    /// must own exactly the slots it should.
    ///
    /// This is the test the module exists for. It is pure arithmetic and needs no GPU, so
    /// it fails on any machine that breaks it.
    #[test]
    fn every_leaf_slot_names_its_point() {
        for scope in [Scope::default(), narrow(), privkey_only()] {
            let points = 3;
            let layout = Layout::new(&scope, points);
            let total = layout.leaves(points);
            assert!(total > 0, "a scope that derives nothing: {scope:?}");

            let mut counts = vec![0usize; points];
            for leaf in 0..total {
                let point = layout.point_of(points, leaf).expect("slot in range");
                assert!(point < points, "slot {leaf} decoded to point {point}");
                counts[point] += 1;
            }
            for (point, n) in counts.iter().enumerate() {
                assert_eq!(
                    *n,
                    layout.leaves_per_point(),
                    "point {point} owns the wrong number of leaves"
                );
            }
        }
    }

    /// The per-point hash count must be the same quantity `Scope` reports, which is what
    /// the progress bar and the false-positive estimate are built from. Two derivations of
    /// one number that could drift apart.
    #[test]
    fn hashes_per_point_agrees_with_the_scope() {
        for scope in [Scope::default(), narrow(), privkey_only()] {
            let layout = Layout::new(&scope, 1);
            assert_eq!(
                layout.hashes_per_point() as u64,
                scope.probes_per_point(),
                "layout and scope disagree for {scope:?}"
            );
        }
    }

    /// A slot past the end is `None`, not a wrapped or panicking answer.
    ///
    /// The short-launch case is the one that matters: region boundaries move with the
    /// launch size, and decoding a short launch against the capacity is exactly the bug
    /// this argument exists to prevent.
    #[test]
    fn an_out_of_range_slot_is_rejected() {
        let scope = Scope::default();
        let layout = Layout::new(&scope, 4);
        assert!(layout.point_of(4, layout.leaves(4)).is_none());
        assert!(layout.point_of(4, layout.leaves(4) - 1).is_some());

        let short = 1;
        assert!(layout.point_of(short, layout.leaves(short) - 1).is_some());
        assert!(layout.point_of(short, layout.leaves(short)).is_none());
        // And the last region's leaves still decode to point 0 in a one-point launch.
        let last = layout.regions.last().unwrap();
        assert_eq!(layout.point_of(short, last.base_per_point * short), Some(0));
    }

    /// Regions must tile the leaf array with no gap and no overlap, or a slot in the seam
    /// would decode to the wrong point or to none at all.
    #[test]
    fn regions_tile_the_leaf_array() {
        for scope in [Scope::default(), narrow(), privkey_only()] {
            let layout = Layout::new(&scope, 1);
            let mut next = 0usize;
            for region in &layout.regions {
                assert_eq!(region.base_per_point, next, "regions leave a seam");
                assert!(region.per_point > 0, "an empty region");
                next += region.per_point;
            }
            assert_eq!(next, layout.leaves_per_point());
        }
    }

    /// A privkey-only scope must not size regions for paths it never walks.
    #[test]
    fn a_privkey_only_scope_walks_no_paths() {
        let layout = Layout::new(&privkey_only(), 1);
        assert_eq!(layout.regions.len(), 1);
        assert_eq!(layout.regions[0].spec, None);
        assert_eq!(layout.leaves_per_point(), 1);
        assert_eq!(layout.masters_per_point(), 0);
    }

    /// Level buffers are sized to the widest level, not the sum: the walk reuses them
    /// between specs, and a sum would allocate several times what a launch needs on the
    /// The level buffer holds the widest *round*, which is a sum across specs -- they
    /// walk in lockstep and share it. It must still be far smaller than the leaf level,
    /// which is the whole reason the last segment writes past it.
    #[test]
    fn level_capacity_is_the_widest_round() {
        let scope = Scope::default();
        let layout = Layout::new(&scope, 1);
        // The widest round is the one every spec spends at its chain level, feeding the
        // index level below: 3 sizes x 2 tree routes x 2 chains, for each of the four
        // specs. The index level itself is wider again and does *not* count, because it
        // writes into the leaf array.
        assert_eq!(layout.level_capacity(1), 4 * (3 * 2 * 2));
        assert!(
            layout.level_capacity(1) * 5 < layout.leaves_per_point(),
            "level buffers are being sized like the leaf level"
        );
        // And it scales with the launch.
        assert_eq!(layout.level_capacity(8), 8 * layout.level_capacity(1));
    }

    /// The round plan is what lets one `public_keys` serve every spec, so the properties
    /// it has to have are worth stating rather than inferring from a throughput number.
    #[test]
    fn the_round_plan_is_right_aligned_and_shareable() {
        let layout = Layout::new(&Scope::default(), 1);
        let rounds = layout.rounds();
        // The deepest spec is m/44'/0'/0'/{0,1}/{0..9}: five segments, five rounds.
        assert_eq!(rounds.len(), 5);

        for (index, round) in rounds.iter().enumerate() {
            // Normal steps come first, so the prefix `public_keys` covers exactly them.
            let first_hardened = round
                .steps
                .iter()
                .position(|s| s.hardened)
                .unwrap_or(round.steps.len());
            assert!(
                round.steps[first_hardened..].iter().all(|s| s.hardened),
                "round {index} interleaves normal and hardened steps"
            );
            // Offsets tile the round without gap or overlap.
            let mut next = 0;
            for step in &round.steps {
                assert_eq!(step.at, next, "round {index} leaves a seam");
                next += step.width;
            }
            assert_eq!(next, round.width);
            assert_eq!(
                round.normal_width,
                round.steps.iter().take(first_hardened).map(|s| s.width).sum::<usize>()
            );
        }

        // Right-aligned: every spec finishes on the last round, which is what puts all
        // four at their index level together.
        let last = rounds.last().unwrap();
        assert_eq!(last.steps.len(), 4, "every spec should act on the last round");
        assert!(last.steps.iter().all(|s| s.last && !s.hardened));
        assert_eq!(last.normal_width, last.width, "the last round is all normal steps");

        // The bare spec has two segments, so it joins two rounds from the end and needs
        // its masters copied in when it does.
        let joiners: Vec<_> =
            rounds.iter().enumerate().flat_map(|(i, r)| r.steps.iter().filter(|s| s.joins).map(move |s| (i, s.spec))).collect();
        assert!(joiners.contains(&(3, 3)), "the bare spec joins at round 3, got {joiners:?}");
    }

    /// A step's children must land where the same spec reads them from next round --
    /// the one thing that, if wrong, silently derives a different tree.
    #[test]
    fn a_steps_children_land_where_it_next_reads_them() {
        for scope in [Scope::default(), narrow()] {
            let layout = Layout::new(&scope, 1);
            let rounds = layout.rounds();
            for (index, round) in rounds.iter().enumerate() {
                for step in &round.steps {
                    if step.last {
                        continue;
                    }
                    let next = rounds[index + 1]
                        .steps
                        .iter()
                        .find(|s| s.spec == step.spec)
                        .expect("an unfinished spec acts again");
                    // It reads exactly the children this step wrote.
                    let children = layout.specs[step.spec].segments()[step.segment].len() as usize;
                    assert_eq!(next.width, step.width * children);
                    assert_eq!(next.segment, step.segment + 1);
                }
            }
        }
    }

    fn narrow() -> Scope {
        Scope {
            material_sizes: vec![32],
            routes: vec![Route::Bip39],
            paths: vec![PathSpec::parse("m/0'/1/{0..4}").unwrap()],
            forms: vec![HashForm::Compressed],
        }
    }

    fn privkey_only() -> Scope {
        Scope {
            material_sizes: vec![32],
            routes: vec![Route::PrivKey],
            paths: vec![],
            forms: vec![HashForm::Compressed],
        }
    }
}
