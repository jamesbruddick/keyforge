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

    /// The widest intermediate level any spec produces, which is what the level buffers
    /// have to be sized for.
    ///
    /// Levels are walked one spec at a time and the buffers are reused between them, so
    /// this is a maximum rather than a sum. Taking the sum would allocate several times
    /// what is needed on a device where memory is the thing that limits the batch.
    pub fn level_capacity(&self, points: usize) -> usize {
        let trees = self.masters_per_point();
        let mut widest = trees;
        for spec in &self.specs {
            let mut width = 1usize;
            for seg in spec.segments() {
                width *= seg.len();
                widest = widest.max(trees * width);
            }
        }
        widest * points
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
    /// very device where memory is the limit.
    #[test]
    fn level_capacity_is_the_widest_level_not_the_sum() {
        let scope = Scope::default();
        let layout = Layout::new(&scope, 1);
        // 3 sizes x 2 tree routes x 2 chains x 10 indices is the widest level any of the
        // default specs reaches.
        assert_eq!(layout.level_capacity(1), 3 * 2 * 2 * 10);
        assert!(layout.level_capacity(1) < layout.leaves_per_point() * 4);
        // And it scales with the launch.
        assert_eq!(layout.level_capacity(8), 8 * layout.level_capacity(1));
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
