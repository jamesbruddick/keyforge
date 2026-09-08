//! Derivation paths, and the sets of them a scan walks.
//!
//! The scanner this grew out of had no path grammar. It had `--purposes 44,49,84,none`
//! crossed with `--chains 0,1` and `--indices 10`, which is exactly the shape
//! libbitcoin's `bx` could produce and nothing else: an account level fixed at
//! `m/purpose'/0'/account'`, two levels below it, and no way to say `m/48'/0'/0'/2'`
//! at all. Widening that enum every time a wallet standard appears is the wrong move,
//! so a path is parsed instead:
//!
//! ```text
//!   m/44'/0'/0'/{0,1}/{0..9}      BIP44 and its segwit successors
//!   m/{0,1}/{0..9}                bare -- `bx hd-private -i N`, and two of the four
//!                                 published Milk Sad canary wallets
//!   m/48'/0'/0'/2'/{0,1}/{0..9}   BIP48 multisig
//!   m                             the master node itself
//! ```
//!
//! A segment is one fixed index, a set, or a range, optionally hardened. That covers
//! every standard the scanner has needed and, more usefully, the ones it has not met.
//!
//! # The hardened bit is never data
//!
//! BIP32 marks a hardened child by setting the high bit of the index, so `44'` and
//! `2147483692` are the same derivation. Accepting the second spelling would make
//! `--path m/2147483648` silently mean `m/0'`, and a scan that thought it was walking
//! normal children would walk hardened ones and report nothing. So a literal value with
//! the high bit set is a **parse error**, and `'` is the only way to say hardened.
//! [`tests::rejects_a_raw_hardened_bit`] is that claim.
//!
//! # Why a leaf is a number
//!
//! The walk emits hundreds of addresses per seed and has to name where each came from.
//! Carrying a `Vec<u32>` per address would allocate on the hot path, so a leaf is
//! identified by its **index within the spec**, and [`PathSpec::leaf`] expands that back
//! into a path only when something is actually being reported. The expansion is a
//! mixed-radix unwind, the same arithmetic the GPU layout uses to decode a thread index,
//! which is why [`tests::leaf_indices_enumerate_in_walk_order`] pins the two orders
//! together: rightmost segment varies fastest.

use std::fmt;

/// BIP32 marks hardened children by setting the high bit of the index.
pub const HARDENED: u32 = 0x8000_0000;

/// The most leaves one spec may derive.
///
/// This is a **type-safety bound, not a matter of taste**. A leaf is identified by a
/// `u32` in `Location`, which is what keeps that struct `Copy` and cheap on a hot path
/// walked billions of times, so a spec wider than `u32::MAX` would wrap two different
/// leaves onto the same index and report one of them under the other's path.
///
/// Rejecting at parse rather than clamping is deliberate. The product of a few wide
/// ranges overflows quietly -- `m/{0..2147483647}/{0..2147483647}/{0..2147483647}` has a
/// `u64` width of exactly zero -- and a spec with width zero derives *nothing*, so a
/// scan would finish, report a clean pass, and have checked none of it. That failure is
/// invisible; a parse error is not.
pub const MAX_SPEC_WIDTH: u64 = u32::MAX as u64;

/// What one level of a path can be.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Values {
    /// A single index: `44'`, `0`.
    Fixed(u32),
    /// An explicit set: `{0,1}`. Kept in the order written, and de-duplicated at parse
    /// time -- `{0,0}` would otherwise derive and report the same address twice.
    Set(Vec<u32>),
    /// An inclusive range: `{0..9}` is ten indices. Inclusive because address ranges
    /// are written the way people count them, and `{0..9}` meaning nine addresses is
    /// the kind of off-by-one that silently shortens a sweep.
    Range { start: u32, end: u32 },
}

impl Values {
    /// How many indices this level expands to.
    pub fn len(&self) -> usize {
        match self {
            Values::Fixed(_) => 1,
            Values::Set(v) => v.len(),
            Values::Range { start, end } => (*end - *start) as usize + 1,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `n`th index at this level, without the hardened bit.
    pub fn get(&self, n: usize) -> Option<u32> {
        match self {
            Values::Fixed(v) => (n == 0).then_some(*v),
            Values::Set(v) => v.get(n).copied(),
            Values::Range { start, end } => {
                let v = start.checked_add(n as u32)?;
                (v <= *end).then_some(v)
            }
        }
    }
}

/// One level of a path: which indices, and whether they are hardened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    pub values: Values,
    pub hardened: bool,
}

impl Segment {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The `n`th child index at this level, hardened bit applied -- i.e. the number
    /// CKDpriv actually takes.
    pub fn child(&self, n: usize) -> Option<u32> {
        let v = self.values.get(n)?;
        Some(if self.hardened { v | HARDENED } else { v })
    }

    /// Every child index at this level, in order.
    pub fn children(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.len()).map(|n| self.child(n).expect("n < len"))
    }
}

/// One derivation path with sets at any level -- a rectangular family of paths.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathSpec {
    segments: Vec<Segment>,
}

impl PathSpec {
    /// `m` -- the master node itself, one leaf and no derivation.
    pub fn master() -> Self {
        PathSpec { segments: Vec::new() }
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// Levels below the master. `m` is depth 0.
    pub fn depth(&self) -> usize {
        self.segments.len()
    }

    /// How many leaves this spec derives per master node: the product of every level.
    ///
    /// Never overflows: [`PathSpec::parse`] refuses anything wider than
    /// [`MAX_SPEC_WIDTH`], and `master()` is empty. The checked arithmetic is kept
    /// anyway so a future constructor cannot reintroduce a silent zero.
    pub fn width(&self) -> u64 {
        Self::checked_width(&self.segments).expect("width is bounded at parse")
    }

    /// The width of a segment list, or `None` if it overflows `u64` on the way.
    fn checked_width(segments: &[Segment]) -> Option<u64> {
        segments
            .iter()
            .try_fold(1u64, |acc, s| acc.checked_mul(s.len() as u64))
    }

    /// Whether any level is hardened, which decides whether the walk can skip EC work
    /// for this spec's upper levels.
    pub fn has_hardened(&self) -> bool {
        self.segments.iter().any(|s| s.hardened)
    }

    /// Expand a leaf index back into the child indices that reach it.
    ///
    /// The rightmost segment varies fastest, matching the order the walk emits leaves
    /// and the order the GPU layout decodes a thread index. Returns `None` for an index
    /// at or past [`PathSpec::width`] rather than wrapping, because on the GPU side this
    /// value arrives from device memory and a corrupt one must surface as a counted
    /// anomaly rather than a plausible wrong answer.
    pub fn leaf(&self, mut index: u64) -> Option<Vec<u32>> {
        if index >= self.width() {
            return None;
        }
        let mut out = vec![0u32; self.segments.len()];
        for (i, seg) in self.segments.iter().enumerate().rev() {
            let n = (index % seg.len() as u64) as usize;
            index /= seg.len() as u64;
            out[i] = seg.child(n).expect("n < len");
        }
        Some(out)
    }

    /// Render a leaf as a path string: `m/44'/0'/0'/0/5`.
    pub fn leaf_path(&self, index: u64) -> Option<String> {
        let children = self.leaf(index)?;
        let mut s = String::from("m");
        for child in children {
            s.push('/');
            if child & HARDENED != 0 {
                s.push_str(&(child & !HARDENED).to_string());
                s.push('\'');
            } else {
                s.push_str(&child.to_string());
            }
        }
        Some(s)
    }

    /// Parse `m/44'/0'/0'/{0,1}/{0..9}`.
    ///
    /// A leading `m/` is required rather than optional: without it `44'/0'` and
    /// `m/44'/0'` would both parse, and one of them is someone who has mistyped a path
    /// that means something else.
    pub fn parse(text: &str) -> Result<Self, ParseError> {
        let text = text.trim();
        let rest = match text.strip_prefix("m/") {
            Some(rest) => rest,
            None if text == "m" => return Ok(PathSpec::master()),
            None => return Err(ParseError::NoMaster(text.to_string())),
        };

        let mut segments = Vec::new();
        for part in rest.split('/') {
            segments.push(parse_segment(part)?);
        }

        // See `MAX_SPEC_WIDTH`: too wide is a parse error, because the alternative is a
        // wrapped leaf index or a silently empty sweep.
        match Self::checked_width(&segments) {
            Some(w) if w <= MAX_SPEC_WIDTH => Ok(PathSpec { segments }),
            _ => Err(ParseError::TooWide(text.to_string())),
        }
    }
}

impl fmt::Display for PathSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("m")?;
        for seg in &self.segments {
            f.write_str("/")?;
            match &seg.values {
                Values::Fixed(v) => write!(f, "{v}")?,
                Values::Set(v) => {
                    f.write_str("{")?;
                    for (i, x) in v.iter().enumerate() {
                        if i > 0 {
                            f.write_str(",")?;
                        }
                        write!(f, "{x}")?;
                    }
                    f.write_str("}")?;
                }
                Values::Range { start, end } => write!(f, "{{{start}..{end}}}")?,
            }
            if seg.hardened {
                f.write_str("'")?;
            }
        }
        Ok(())
    }
}

fn parse_segment(part: &str) -> Result<Segment, ParseError> {
    let part = part.trim();
    if part.is_empty() {
        return Err(ParseError::EmptySegment);
    }

    // `'`, `h` and `H` are all in use across wallet software for the same thing.
    let (body, hardened) = match part.strip_suffix(['\'', 'h', 'H']) {
        Some(body) => (body, true),
        None => (part, false),
    };
    if body.is_empty() {
        return Err(ParseError::EmptySegment);
    }

    let values = if let Some(inner) = body.strip_prefix('{').and_then(|b| b.strip_suffix('}')) {
        // `{}` before anything else: it would otherwise reach the set branch, split into
        // one empty item, and be reported as `NotAnIndex("")`, which describes the
        // symptom rather than the mistake.
        if inner.trim().is_empty() {
            return Err(ParseError::EmptySegment);
        }
        if let Some((lo, hi)) = inner.split_once("..") {
            let start = parse_index(lo)?;
            let end = parse_index(hi)?;
            if start > end {
                return Err(ParseError::BackwardsRange { start, end });
            }
            Values::Range { start, end }
        } else {
            let mut set = Vec::new();
            for item in inner.split(',') {
                let v = parse_index(item)?;
                // `{0,0}` would derive and report the same address twice, which inflates
                // a candidate count and wastes a probe. Written order is kept.
                if !set.contains(&v) {
                    set.push(v);
                }
            }
            Values::Set(set)
        }
    } else if body.starts_with('{') || body.ends_with('}') {
        return Err(ParseError::UnbalancedBraces(part.to_string()));
    } else {
        Values::Fixed(parse_index(body)?)
    };

    Ok(Segment { values, hardened })
}

/// Parse one index and refuse the hardened bit as data -- see the module note.
fn parse_index(text: &str) -> Result<u32, ParseError> {
    let text = text.trim();
    let value: u32 = text
        .parse()
        .map_err(|_| ParseError::NotAnIndex(text.to_string()))?;
    if value >= HARDENED {
        return Err(ParseError::HardenedBitSet(value));
    }
    Ok(value)
}

#[derive(Debug, PartialEq, Eq)]
pub enum ParseError {
    NoMaster(String),
    EmptySegment,
    NotAnIndex(String),
    HardenedBitSet(u32),
    BackwardsRange { start: u32, end: u32 },
    UnbalancedBraces(String),
    TooWide(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::NoMaster(t) => {
                write!(f, "path `{t}` must start with `m/` (or be exactly `m`)")
            }
            ParseError::EmptySegment => f.write_str("path has an empty level"),
            ParseError::NotAnIndex(t) => write!(f, "`{t}` is not a derivation index"),
            ParseError::HardenedBitSet(v) => write!(
                f,
                "index {v} has the hardened bit set; write `{}'` instead",
                v & !HARDENED
            ),
            ParseError::BackwardsRange { start, end } => {
                write!(f, "range {{{start}..{end}}} counts backwards")
            }
            ParseError::UnbalancedBraces(t) => write!(f, "`{t}` has unbalanced braces"),
            ParseError::TooWide(t) => write!(
                f,
                "path `{t}` derives more than {MAX_SPEC_WIDTH} addresses per key; \
                 narrow a range or split it across several --path arguments"
            ),
        }
    }
}

impl std::error::Error for ParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_shapes_the_old_scanner_hardcoded() {
        let bip44 = PathSpec::parse("m/44'/0'/0'/{0,1}/{0..9}").unwrap();
        assert_eq!(bip44.depth(), 5);
        // 2 chains x 10 indices, which is what `--chains 0,1 --indices 10` meant.
        assert_eq!(bip44.width(), 20);
        assert!(bip44.has_hardened());

        let bare = PathSpec::parse("m/{0,1}/{0..9}").unwrap();
        assert_eq!(bare.depth(), 2);
        assert_eq!(bare.width(), 20);
        assert!(!bare.has_hardened());

        let master = PathSpec::parse("m").unwrap();
        assert_eq!(master.depth(), 0);
        // The master node is one leaf, not none: an empty product is 1, and a spec that
        // derived nothing would silently drop `bx ec-new` keys from a sweep.
        assert_eq!(master.width(), 1);
    }

    /// A literal high-bit index must not be a quiet synonym for hardened -- see the
    /// module note. This is the whole reason `parse_index` exists.
    #[test]
    fn rejects_a_raw_hardened_bit() {
        assert_eq!(
            PathSpec::parse("m/2147483648"),
            Err(ParseError::HardenedBitSet(2147483648))
        );
        // And the error says what to write instead, because the fix is not obvious.
        assert!(
            ParseError::HardenedBitSet(2147483692)
                .to_string()
                .contains("44'")
        );
        // The legitimate spelling of the same derivation still works.
        assert_eq!(
            PathSpec::parse("m/44'").unwrap().segments()[0].child(0),
            Some(44 | HARDENED)
        );
    }

    /// Leaf indices must enumerate in the order the walk emits them -- rightmost level
    /// fastest -- because the GPU decodes a thread index with the same arithmetic and
    /// the two orders must not drift.
    #[test]
    fn leaf_indices_enumerate_in_walk_order() {
        let spec = PathSpec::parse("m/44'/{7,8}/{0..2}").unwrap();
        assert_eq!(spec.width(), 6);

        let got: Vec<String> = (0..spec.width())
            .map(|i| spec.leaf_path(i).unwrap())
            .collect();
        assert_eq!(
            got,
            vec![
                "m/44'/7/0", "m/44'/7/1", "m/44'/7/2",
                "m/44'/8/0", "m/44'/8/1", "m/44'/8/2",
            ]
        );
    }

    /// A leaf past the end is `None`, not a wrapped or panicking answer: on the GPU path
    /// this index arrives from device memory.
    #[test]
    fn an_out_of_range_leaf_is_rejected() {
        let spec = PathSpec::parse("m/{0,1}/{0..9}").unwrap();
        assert!(spec.leaf(spec.width() - 1).is_some());
        assert!(spec.leaf(spec.width()).is_none());
        assert!(spec.leaf(u64::MAX).is_none());
    }

    #[test]
    fn round_trips_through_display() {
        for text in [
            "m",
            "m/0",
            "m/44'/0'/0'/{0,1}/{0..9}",
            "m/48'/0'/0'/2'/{0,1}/{0..9}",
            "m/{1,4,9}'/0",
        ] {
            let spec = PathSpec::parse(text).unwrap();
            assert_eq!(spec.to_string(), text, "`{text}` did not survive a round trip");
            assert_eq!(PathSpec::parse(&spec.to_string()).unwrap(), spec);
        }
    }

    #[test]
    fn accepts_the_h_spellings_of_hardened() {
        let apostrophe = PathSpec::parse("m/44'").unwrap();
        for spelling in ["m/44h", "m/44H"] {
            assert_eq!(PathSpec::parse(spelling).unwrap(), apostrophe);
        }
    }

    #[test]
    fn rejects_malformed_paths() {
        use ParseError::*;
        assert!(matches!(PathSpec::parse("44'/0'"), Err(NoMaster(_))));
        assert!(matches!(PathSpec::parse("m/"), Err(EmptySegment)));
        assert!(matches!(PathSpec::parse("m/0//1"), Err(EmptySegment)));
        assert!(matches!(PathSpec::parse("m/{}"), Err(EmptySegment)));
        assert!(matches!(PathSpec::parse("m/abc"), Err(NotAnIndex(_))));
        assert!(matches!(PathSpec::parse("m/-1"), Err(NotAnIndex(_))));
        assert!(matches!(PathSpec::parse("m/{0,1"), Err(UnbalancedBraces(_))));
        assert!(matches!(
            PathSpec::parse("m/{9..0}"),
            Err(BackwardsRange { start: 9, end: 0 })
        ));
    }

    /// A spec wide enough to wrap a `u32` leaf index must be refused at parse.
    ///
    /// Both halves of this were real. `m/{0..2147483647}` cubed has a `u64` width of
    /// exactly *zero*, so every leaf lookup returns `None` and the spec derives nothing
    /// while the scan reports a clean pass; and a width of ten billion exceeds the `u32`
    /// a leaf index is stored in, so two different addresses collide onto one index and
    /// one gets reported under the other's path.
    #[test]
    fn rejects_a_spec_too_wide_for_a_u32_leaf_index() {
        // Overflows u64 outright, and used to produce width 0.
        let cubed = "m/{0..2147483647}/{0..2147483647}/{0..2147483647}";
        assert!(matches!(PathSpec::parse(cubed), Err(ParseError::TooWide(_))));

        // Fits u64 but not u32: 10^10 leaves.
        let wide = "m/{0..99999}/{0..99999}";
        assert!(matches!(PathSpec::parse(wide), Err(ParseError::TooWide(_))));

        // The widest a single level can be is 2^31, since an index may not have the
        // hardened bit set. That is accepted, and its width is exact.
        let widest = PathSpec::parse("m/{0..2147483647}").unwrap();
        assert_eq!(widest.width(), 1 << 31);
        assert!(widest.leaf((1 << 31) - 1).is_some());
        assert!(widest.leaf(1 << 31).is_none());

        // Doubling it steps one past what a u32 leaf index can address, and is refused.
        assert!(matches!(
            PathSpec::parse("m/{0..2147483647}/{0,1}"),
            Err(ParseError::TooWide(_))
        ));

        // Anything that parses has a width a u32 leaf index can address, which is the
        // property the walk relies on.
        for text in ["m", "m/0", "m/44'/0'/0'/{0,1}/{0..9}", "m/{0..2147483647}"] {
            let spec = PathSpec::parse(text).unwrap();
            assert!(spec.width() <= u32::MAX as u64, "`{text}` overflows a u32 leaf");
        }
    }

    /// A repeated index in a set derives the same address twice, inflating the candidate
    /// count and wasting a probe per duplicate.
    #[test]
    fn deduplicates_a_repeated_index() {
        let spec = PathSpec::parse("m/{0,1,0,1}").unwrap();
        assert_eq!(spec.width(), 2);
    }
}
