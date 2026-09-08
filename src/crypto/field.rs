//! Arithmetic in the secp256k1 base field, GF(2^256 - 2^32 - 977).
//!
//! This exists so the scanner can run its own fixed-base multiplier (`crate::crypto::ec`)
//! instead of libsecp256k1's, which is constant-time and pays for that on every one
//! of the hundreds of public keys per seed. Nothing here is secret: every scalar is derived
//! from a published 32-bit timestamp, and every key is one an attacker could compute
//! from the same public information. So these routines are branchy and data-dependent
//! by design.
//!
//! **Do not reuse this module for anything holding a real secret.** It leaks its
//! operands through timing and is not written to resist that.
//!
//! Elements are kept fully reduced in four 64-bit limbs, least significant first,
//! rather than in the lazily-reduced 5x52 form libsecp256k1 uses. Full reduction
//! costs a conditional subtract per add and a slightly longer carry chain per
//! multiply; it buys an invariant simple enough to state in one line and test
//! directly, which is the right trade for a reimplementation whose only defence is
//! its tests.
//!
//! That trade has been priced rather than assumed. Deleting the canonical-reduction
//! tail from `reduce_wide` outright -- incorrect, but an upper bound on what dropping
//! the invariant could ever be worth -- moves a curve-only scope from 1,922 to 2,068
//! seeds/s, 7.6%, or about 4% of a default-scope sweep. A *correct* weak-reduction
//! form gives some of that straight back: with operands allowed up to 2^256, `sub`
//! loses its guarantee that a borrow leaves a value above `C` and needs a second
//! conditional, and the hot path runs `sub` six times per point addition against three
//! multiplies. What is left over is not worth an invariant that no longer holds, whose
//! violations arrive with probability around 2^-224 and which therefore no test can
//! produce. The real version of this change is libsecp256k1's 5x52 form, and that is a
//! rewrite of the module, not an edit to it.

/// p = 2^256 - C.
const P: [u64; 4] = [
    0xffff_fffe_ffff_fc2f,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
];

/// The 33-bit constant that makes the field's reduction cheap: 2^256 = C (mod p).
const C: u64 = 0x1_0000_03d1;

/// A field element, always fully reduced: the represented value is below `P`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Fe(pub [u64; 4]);

pub const ZERO: Fe = Fe([0, 0, 0, 0]);
pub const ONE: Fe = Fe([1, 0, 0, 0]);
/// The curve's b: y^2 = x^3 + 7. Used only to check points in tests.
#[cfg(test)]
pub const B: Fe = Fe([7, 0, 0, 0]);

#[inline(always)]
fn add_limbs(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], bool) {
    let mut r = [0u64; 4];
    let mut carry = false;
    for i in 0..4 {
        let (s, c1) = a[i].overflowing_add(b[i]);
        let (s, c2) = s.overflowing_add(carry as u64);
        r[i] = s;
        carry = c1 | c2;
    }
    (r, carry)
}

#[inline(always)]
fn sub_limbs(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], bool) {
    let mut r = [0u64; 4];
    let mut borrow = false;
    for i in 0..4 {
        let (d, b1) = a[i].overflowing_sub(b[i]);
        let (d, b2) = d.overflowing_sub(borrow as u64);
        r[i] = d;
        borrow = b1 | b2;
    }
    (r, borrow)
}

/// Is `a` at or above the modulus, and therefore in need of one subtraction?
#[inline(always)]
fn needs_reduction(a: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] != P[i] {
            return a[i] > P[i];
        }
    }
    true
}

/// Add `v` at the bottom limb and propagate, returning the carry out of the top.
#[inline(always)]
fn add_wide_at_zero(r: &mut [u64; 4], v: u128) -> u64 {
    let mut carry = v;
    for limb in r.iter_mut() {
        let t = (*limb as u128) + (carry & u64::MAX as u128);
        *limb = t as u64;
        carry = (carry >> 64) + (t >> 64);
    }
    carry as u64
}

/// Reduce a 512-bit product modulo p.
///
/// Folds the high half down with `2^256 = C`, twice, then once more for any carry
/// the folds themselves produced. Each fold shrinks the excess by 223 bits, so the
/// loop runs at most once and the result is below 2^256 before the final subtract.
#[inline(always)]
fn reduce_wide(prod: &[u64; 8]) -> Fe {
    // r = low || 0 + high * C, which is at most 289 bits, i.e. five limbs.
    let mut r = [0u64; 5];
    let mut carry: u128 = 0;
    for i in 0..4 {
        let t = (prod[4 + i] as u128) * (C as u128) + (prod[i] as u128) + carry;
        r[i] = t as u64;
        carry = t >> 64;
    }
    r[4] = carry as u64;

    // Fold the fifth limb back in, and any carry that produces after it.
    let mut s = [r[0], r[1], r[2], r[3]];
    let mut carry = (r[4] as u128) * (C as u128);
    while carry != 0 {
        carry = (add_wide_at_zero(&mut s, carry) as u128) * (C as u128);
    }

    if needs_reduction(&s) {
        s = sub_limbs(&s, &P).0;
    }
    Fe(s)
}

impl Fe {
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.0 == [0, 0, 0, 0]
    }

    #[inline]
    pub fn is_odd(&self) -> bool {
        self.0[0] & 1 == 1
    }

    #[inline(always)]
    pub fn add(&self, other: &Fe) -> Fe {
        let (mut s, carry) = add_limbs(&self.0, &other.0);
        // The sum is below 2p, so at most one reduction is needed -- but a carry out
        // of the top limb means 2^256 was dropped, which is congruent to C.
        if carry {
            s = add_limbs(&s, &[C, 0, 0, 0]).0;
        }
        if needs_reduction(&s) {
            s = sub_limbs(&s, &P).0;
        }
        Fe(s)
    }

    #[inline(always)]
    pub fn sub(&self, other: &Fe) -> Fe {
        let (mut d, borrow) = sub_limbs(&self.0, &other.0);
        // Borrowing added an implicit 2^256; p = 2^256 - C, so subtracting C turns
        // that into the +p the result wanted. The value left is always above C.
        if borrow {
            d = sub_limbs(&d, &[C, 0, 0, 0]).0;
        }
        Fe(d)
    }

    #[inline(always)]
    pub fn negate(&self) -> Fe {
        if self.is_zero() {
            ZERO
        } else {
            Fe(sub_limbs(&P, &self.0).0)
        }
    }

    // Small multiples, spelled out one by one because the curve formulas need only
    // these four and a generic routine would cost an extra addition on each.

    #[inline(always)]
    pub fn mul2(&self) -> Fe {
        self.add(self)
    }

    #[inline(always)]
    pub fn mul3(&self) -> Fe {
        self.mul2().add(self)
    }

    #[inline(always)]
    pub fn mul4(&self) -> Fe {
        self.mul2().mul2()
    }

    #[inline(always)]
    pub fn mul8(&self) -> Fe {
        self.mul4().mul2()
    }

    #[inline(always)]
    pub fn mul(&self, other: &Fe) -> Fe {
        let mut prod = [0u64; 8];
        for i in 0..4 {
            let mut carry: u64 = 0;
            for j in 0..4 {
                let t = (self.0[i] as u128) * (other.0[j] as u128)
                    + (prod[i + j] as u128)
                    + (carry as u128);
                prod[i + j] = t as u64;
                carry = (t >> 64) as u64;
            }
            prod[i + 4] = carry;
        }
        reduce_wide(&prod)
    }

    /// Squaring, which the curve formulas use about as often as multiplication.
    ///
    /// The off-diagonal terms of the product appear twice, so half of them are
    /// computed and the partial sum is doubled before the diagonal is folded in.
    #[inline(always)]
    pub fn sqr(&self) -> Fe {
        let a = &self.0;
        let mut prod = [0u64; 8];

        // Off-diagonal half: a[i]*a[j] for j > i.
        for i in 0..4 {
            let mut carry: u64 = 0;
            for j in (i + 1)..4 {
                let t = (a[i] as u128) * (a[j] as u128) + (prod[i + j] as u128) + (carry as u128);
                prod[i + j] = t as u64;
                carry = (t >> 64) as u64;
            }
            prod[i + 4] = carry;
        }

        // Double it. The top limb cannot overflow: the off-diagonal sum is below
        // 2^446, so doubling stays inside 512 bits.
        let mut carry = 0u64;
        for limb in prod.iter_mut() {
            let t = ((*limb as u128) << 1) + (carry as u128);
            *limb = t as u64;
            carry = (t >> 64) as u64;
        }
        debug_assert_eq!(carry, 0);

        // Then the diagonal a[i]^2 at position 2i.
        let mut carry: u128 = 0;
        for i in 0..4 {
            let t = (a[i] as u128) * (a[i] as u128) + (prod[2 * i] as u128) + carry;
            prod[2 * i] = t as u64;
            let t = (prod[2 * i + 1] as u128) + (t >> 64);
            prod[2 * i + 1] = t as u64;
            carry = t >> 64;
        }
        debug_assert_eq!(carry, 0);

        reduce_wide(&prod)
    }

    /// Modular inverse, as `self^(p-2)`.
    ///
    /// The addition chain is the one libsecp256k1 uses: 255 squarings and 15
    /// multiplications, exploiting the long run of set bits in p-2. Inversion is by
    /// far the most expensive field operation, which is why `crate::crypto::ec` batches all
    /// of a level's points through a single one.
    pub fn inv(&self) -> Fe {
        // Powers named for the number of trailing set bits they carry.
        let x1 = *self;
        let x2 = x1.sqr().mul(&x1);
        let x3 = x2.sqr().mul(&x1);
        let x6 = repeat_sqr(&x3, 3).mul(&x3);
        let x9 = repeat_sqr(&x6, 3).mul(&x3);
        let x11 = repeat_sqr(&x9, 2).mul(&x2);
        let x22 = repeat_sqr(&x11, 11).mul(&x11);
        let x44 = repeat_sqr(&x22, 22).mul(&x22);
        let x88 = repeat_sqr(&x44, 44).mul(&x44);
        let x176 = repeat_sqr(&x88, 88).mul(&x88);
        let x220 = repeat_sqr(&x176, 44).mul(&x44);
        let x223 = repeat_sqr(&x220, 3).mul(&x3);

        // p - 2 = x223 || 23 zero-ish bits || the tail 0b...111111 0 1101101 11
        let mut r = repeat_sqr(&x223, 23).mul(&x22);
        r = repeat_sqr(&r, 5).mul(&x1);
        r = repeat_sqr(&r, 3).mul(&x2);
        repeat_sqr(&r, 2).mul(&x1)
    }

    /// Square root by `self^((p+1)/4)`, valid because p = 3 (mod 4).
    ///
    /// Returns a root whose square may not be `self` -- the caller must check, since
    /// only half the field has one. Used by the tests, not the hot path.
    #[cfg(test)]
    pub fn sqrt(&self) -> Fe {
        let x1 = *self;
        let x2 = x1.sqr().mul(&x1);
        let x3 = x2.sqr().mul(&x1);
        let x6 = repeat_sqr(&x3, 3).mul(&x3);
        let x9 = repeat_sqr(&x6, 3).mul(&x3);
        let x11 = repeat_sqr(&x9, 2).mul(&x2);
        let x22 = repeat_sqr(&x11, 11).mul(&x11);
        let x44 = repeat_sqr(&x22, 22).mul(&x22);
        let x88 = repeat_sqr(&x44, 44).mul(&x44);
        let x176 = repeat_sqr(&x88, 88).mul(&x88);
        let x220 = repeat_sqr(&x176, 44).mul(&x44);
        let x223 = repeat_sqr(&x220, 3).mul(&x3);

        let mut r = repeat_sqr(&x223, 23).mul(&x22);
        r = repeat_sqr(&r, 6).mul(&x2);
        repeat_sqr(&r, 2)
    }

    /// Big-endian bytes. The canonical encoding, since elements are always reduced.
    #[inline]
    pub fn to_bytes(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (i, limb) in self.0.iter().enumerate() {
            out[24 - i * 8..32 - i * 8].copy_from_slice(&limb.to_be_bytes());
        }
        out
    }

    /// Big-endian bytes, rejecting anything at or above the modulus.
    ///
    /// Nothing in the scan parses a field element -- they are only ever computed --
    /// so this is the tests' way in.
    #[cfg(test)]
    pub fn from_bytes(bytes: &[u8; 32]) -> Option<Fe> {
        let mut limbs = [0u64; 4];
        for i in 0..4 {
            limbs[i] = u64::from_be_bytes(bytes[24 - i * 8..32 - i * 8].try_into().unwrap());
        }
        if needs_reduction(&limbs) {
            return None;
        }
        Some(Fe(limbs))
    }
}

#[inline]
fn repeat_sqr(x: &Fe, times: u32) -> Fe {
    let mut r = *x;
    for _ in 0..times {
        r = r.sqr();
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;
    use rand::RngExt;

    fn modulus() -> BigUint {
        BigUint::from_bytes_be(&[
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe,
            0xff, 0xff, 0xfc, 0x2f,
        ])
    }

    fn big(f: &Fe) -> BigUint {
        BigUint::from_bytes_be(&f.to_bytes())
    }

    fn fe(v: &BigUint) -> Fe {
        let bytes = v.to_bytes_be();
        let mut padded = [0u8; 32];
        padded[32 - bytes.len()..].copy_from_slice(&bytes);
        Fe::from_bytes(&padded).unwrap()
    }

    /// Random field elements, plus the edge cases a carry bug hides in: zero, one,
    /// p-1, and values straddling a limb boundary.
    fn samples(count: usize) -> Vec<Fe> {
        let p = modulus();
        let mut out = vec![
            ZERO,
            ONE,
            fe(&(p.clone() - 1u32)),
            fe(&(p.clone() - 2u32)),
            Fe([u64::MAX, u64::MAX, u64::MAX, 0]),
            Fe([0, 0, 0, u64::MAX]),
            Fe([u64::MAX, 0, 0, 0]),
        ];
        let mut rng = rand::rng();
        while out.len() < count {
            let bytes: [u8; 32] = rng.random();
            if let Some(f) = Fe::from_bytes(&bytes) {
                out.push(f);
            }
        }
        out
    }

    /// Every arithmetic operation against big-integer modular arithmetic, over both
    /// random elements and the boundary values. A reduction that is off by one p, or
    /// a carry dropped between limbs, cannot survive this.
    #[test]
    fn arithmetic_matches_big_integer_modular_arithmetic() {
        let p = modulus();
        let values = samples(64);

        for a in &values {
            let ba = big(a);
            assert_eq!(big(&a.negate()), (&p - &ba) % &p, "negate");
            assert_eq!(big(&a.sqr()), (&ba * &ba) % &p, "sqr");
            assert_eq!(big(&a.mul2()), (&ba * 2u32) % &p, "mul2");
            assert_eq!(big(&a.mul3()), (&ba * 3u32) % &p, "mul3");
            assert_eq!(big(&a.mul4()), (&ba * 4u32) % &p, "mul4");
            assert_eq!(big(&a.mul8()), (&ba * 8u32) % &p, "mul8");

            for b in &values {
                let bb = big(b);
                assert_eq!(big(&a.add(b)), (&ba + &bb) % &p, "add");
                assert_eq!(big(&a.sub(b)), (&ba + &p - &bb) % &p, "sub");
                assert_eq!(big(&a.mul(b)), (&ba * &bb) % &p, "mul");
            }
        }
    }

    /// `inv` is the longest addition chain here and the easiest to get subtly wrong,
    /// so it is checked as a round trip rather than against a transcribed constant.
    #[test]
    fn inverse_round_trips() {
        assert_eq!(ZERO.inv(), ZERO, "0^(p-2) is 0, and callers rely on it");
        for a in samples(48) {
            if a.is_zero() {
                continue;
            }
            assert_eq!(a.inv().mul(&a), ONE, "a * a^-1 must be 1");
        }
    }

    /// The square root chain, used to rebuild points from x in the tests.
    #[test]
    fn square_root_inverts_squaring() {
        for a in samples(32) {
            let root = a.sqr().sqrt();
            assert!(root == a || root == a.negate(), "sqrt(a^2) must be +/-a");
        }
    }

    /// Serialisation is canonical: it round trips, and it refuses the values at or
    /// above p that a non-canonical encoding would otherwise smuggle in.
    #[test]
    fn byte_encoding_is_canonical() {
        for a in samples(32) {
            assert_eq!(Fe::from_bytes(&a.to_bytes()), Some(a));
        }
        let mut at_p = [0u8; 32];
        at_p.copy_from_slice(&fe(&(modulus() - 1u32)).to_bytes());
        assert!(Fe::from_bytes(&at_p).is_some());
        assert_eq!(Fe::from_bytes(&[0xff; 32]), None, "2^256-1 is above p");
    }
}
