// secp256k1 scalar multiplication against the fixed generator.
//
// This mirrors src/ec.rs in what it computes and **deliberately not in how**. The CPU
// runs 256 scalars through one digit position at a time so that a single field inversion
// can be shared across the whole position, which makes affine addition (3 field muls)
// affordable in place of mixed Jacobian addition (11). That trade is right for a core
// whose issue slots would otherwise sit idle waiting on a dependent chain.
//
// It is wrong here, and the arithmetic says so plainly. A Fermat inversion is ~270
// *sequential* field operations on one thread. Sharing one across a threadgroup buys each
// lane ~5 muls of parallel work per digit row and costs 270 mul-times of stall, seventeen
// times per scalar. Widening the inversion group until the groups themselves parallelise
// fixes the stall but needs barriers, threadgroup memory per row, and a divergence-coupled
// digit loop -- to save 165 muls against 220. 1.33x, for all of that.
//
// So: **one thread, one scalar, Jacobian accumulation, no communication.** The level's
// points are converted to affine afterwards by a single batched inversion (see the
// `invert_*` kernels), which is where the sharing belongs on this machine.
//
// A consequence worth naming: the CPU's `ABANDONED` lane machinery has no analogue here.
// It exists because a degenerate addition inside a shared-inversion chunk poisons the
// whole chunk, so those lanes are pulled out and redone. A Jacobian lane just takes the
// branch -- one warp's divergence, on a ~2^-128 event.

#define EC_BASE    (1u << EC_WINDOW)
#define EC_ROW_LEN (EC_BASE / 2)

typedef struct { Fe x; Fe y; } Ge;
typedef struct { Fe x; Fe y; Fe z; bool infinity; } Gej;

// Infinity is (0, 0) in affine form rather than a flag, exactly as on the CPU: x = 0 gives
// y^2 = 7, whose roots are not zero, so the point is not on the curve and the encoding is
// free to mean something else.
INLINE bool ge_is_infinity(Ge p) { return fe_is_zero(p.x) && fe_is_zero(p.y); }

INLINE Ge ge_negate(Ge p) { Ge r; r.x = p.x; r.y = fe_neg(p.y); return r; }

INLINE Gej gej_infinity() {
    Gej r;
    r.x = fe_zero(); r.y = fe_zero(); r.z = fe_zero(); r.infinity = true;
    return r;
}

INLINE Gej gej_from_ge(Ge p) {
    Gej r;
    r.x = p.x; r.y = p.y; r.z = fe_one(); r.infinity = false;
    return r;
}

INLINE Fe fe_mul2(Fe a) { return fe_add(a, a); }
INLINE Fe fe_mul3(Fe a) { return fe_add(fe_mul2(a), a); }
INLINE Fe fe_mul4(Fe a) { return fe_mul2(fe_mul2(a)); }
INLINE Fe fe_mul8(Fe a) { return fe_mul2(fe_mul4(a)); }

INLINE Gej gej_double(Gej p) {
    if (p.infinity || fe_is_zero(p.y)) return gej_infinity();
    Fe a = fe_sqr(p.x);
    Fe b = fe_sqr(p.y);
    Fe c = fe_sqr(b);
    Fe d = fe_mul2(fe_sub(fe_sub(fe_sqr(fe_add(p.x, b)), a), c));
    Fe e = fe_mul3(a);
    Fe f = fe_sqr(e);

    Gej r;
    r.x = fe_sub(f, fe_mul2(d));
    r.y = fe_sub(fe_mul(e, fe_sub(d, r.x)), fe_mul8(c));
    r.z = fe_mul2(fe_mul(p.y, p.z));
    r.infinity = false;
    return r;
}

// Mixed addition, `madd-2007-bl` from the EFD: Jacobian plus affine.
//
// Branchy by design -- see the note in src/field.rs about there being no secret here.
INLINE Gej gej_add_ge(Gej p, Ge q) {
    if (ge_is_infinity(q)) return p;
    if (p.infinity) return gej_from_ge(q);

    Fe z1z1 = fe_sqr(p.z);
    Fe u2 = fe_mul(q.x, z1z1);
    Fe s2 = fe_mul(fe_mul(q.y, p.z), z1z1);
    Fe h = fe_sub(u2, p.x);
    Fe r = fe_mul2(fe_sub(s2, p.y));

    if (fe_is_zero(h)) {
        // Same x: either the same point, needing the doubling formula, or a point and
        // its negation, which sum to infinity.
        return fe_is_zero(r) ? gej_double(p) : gej_infinity();
    }

    Fe hh = fe_sqr(h);
    Fe i = fe_mul4(hh);
    Fe j = fe_mul(h, i);
    Fe v = fe_mul(p.x, i);

    Gej out;
    out.x = fe_sub(fe_sub(fe_sqr(r), j), fe_mul2(v));
    out.y = fe_sub(fe_mul(r, fe_sub(v, out.x)), fe_mul2(fe_mul(p.y, j)));
    out.z = fe_sub(fe_sub(fe_sqr(fe_add(p.z, h)), z1z1), hh);
    out.infinity = false;
    return out;
}

// Recode a big-endian scalar into signed base-2^EC_WINDOW digits.
//
// A plain split would need EC_BASE table entries per row; borrowing from the next digit
// whenever one exceeds EC_BASE/2 halves that, because -d is a free negation of d's point.
// The borrow can run off the top, which is what the extra row in EC_ROWS is for.
// One digit, computed where it is used rather than stored.
//
// The array form -- recode all EC_ROWS digits, then walk them -- is what src/ec.rs does,
// and it is right there because the CPU transposes a whole chunk of scalars. Here it costs
// an 84-byte thread-private array indexed by the loop variable, which cannot live in
// registers. The recoding carries left to right, so the carry is simply threaded through
// the row loop instead, and nothing is stored at all.
INLINE int ec_digit_at(THREAD const u32* limbs, u32 j, THREAD int* carry) {
    if (j == EC_ROWS - 1) return *carry; // the borrow that ran off the top
    u32 bit = j * EC_WINDOW;
    u32 limb = bit / 32, offset = bit % 32;
    // Read out of a 64-bit window so a digit straddling a limb boundary needs no special
    // case. Digits past bit 256 read as zero.
    u64 lo = (limb < 8) ? (u64)limbs[limb] : 0UL;
    u64 hi = (limb + 1 < 8) ? (u64)limbs[limb + 1] : 0UL;
    u64 wide = lo | (hi << 32);
    int d = (int)((wide >> offset) & (u64)(EC_BASE - 1)) + *carry;

    // Borrowing from the next digit whenever one exceeds EC_BASE/2 halves the table,
    // because -d is a free negation of d's point.
    if (d > (int)EC_ROW_LEN) {
        *carry = 1;
        return d - (int)EC_BASE;
    }
    *carry = 0;
    return d;
}

// The comb entry a signed digit selects, negated when the digit is.
//
// Row j holds d * EC_BASE^j * G for d in 1..=EC_ROW_LEN, laid out row-major -- built by
// the same `ec::build_comb` the CPU's table is, and transcoded to this file's limb order by
// `ec::comb_for_gpu`.
//
// **Not the same table, though: the GPU's window is wider.** `ec::GPU_WINDOW` is 16 against
// the CPU's 13, which is 17 rows and 35 MB here against 21 rows and 5.5 MB there. That was
// one constant until it was measured, and the measurement says the two devices want
// genuinely different answers -- see the table above `GPU_WINDOW` in src/ec.rs. The short
// version is that this gather never becomes the bottleneck, so every row removed from the
// loop below is a straight win: 13 -> 16 took `k_kmul` from 3.26s to 2.75s and the whole
// sweep from 2,646 to 2,945 seeds/s on an M1 Pro.
INLINE Ge ec_comb_entry(DEVICE const Ge* comb, u32 row, int digit) {
    u32 d = (u32)(digit < 0 ? -digit : digit);
    Ge p = comb[row * EC_ROW_LEN + d - 1];
    return digit < 0 ? ge_negate(p) : p;
}

// scalar * G, in Jacobian coordinates.
//
// **The row loop is deliberately not unrolled**, and this is the one place where that
// matters enough to spell out.
//
// Unrolling it looks attractive: it makes `bit / 32` inside `ec_digit_at` a compile-time
// constant and keeps the scalar's limbs out of memory. It was tried. The body is a full
// point addition -- ~16 field multiplies, each an unrolled 8x8 schoolbook -- so seventeen
// copies is tens of thousands of inlined instructions.
//
// Metal caps the expansion and merely runs slower for it. NVRTC honours the pragma
// literally and stops finishing: every GPU test on an RTX 5070 Ti hung rather than failed.
//
// Removing it, measured on an M1 Pro over 20,000 seeds, medians of three: a curve-only
// scope went 4,596 -> 6,182 seeds/s, **+35%**, while the default scope did not move
// (2,199 -> 2,112, inside the noise) and the kernel profile put `k_kmul` at the same 1.4s
// either way. The two do not reconcile neatly and the reason is not established; what is
// established is that removing it is never worse on Metal and is the difference between
// compiling and not on CUDA.
// Takes the `Scalar` rather than big-endian bytes, which is not a convenience: `Scalar`
// already holds little-endian 32-bit limbs and that is exactly what the digit window wants,
// so serialising to bytes and parsing straight back was the identity composed with itself
// -- through a 32-byte thread-private array, on the kernel that is 40% of the sweep.
INLINE Gej ec_mul_gen(Scalar s, DEVICE const Ge* comb) {
    // Little-endian 32-bit limbs, so digit j is the window at bit j*EC_WINDOW.
    u32 limbs[8];
    UNROLL for (u32 i = 0; i < 8; i++) limbs[i] = s.n[i];

    Gej acc = gej_infinity();
    int carry = 0;
    for (u32 row = 0; row < EC_ROWS; row++) {
        int d = ec_digit_at(limbs, row, &carry);
        if (d == 0) continue;
        acc = gej_add_ge(acc, ec_comb_entry(comb, row, d));
    }
    return acc;
}

// Jacobian to affine, given 1/z already computed. (x, y) = (X/z^2, Y/z^3).
INLINE Ge gej_to_ge(Gej p, Fe zinv) {
    Ge r;
    if (p.infinity) { r.x = fe_zero(); r.y = fe_zero(); return r; }
    Fe zi2 = fe_sqr(zinv);
    r.x = fe_mul(p.x, zi2);
    r.y = fe_mul(fe_mul(p.y, zi2), zinv);
    return r;
}

// SEC1 compressed encoding: a parity byte and x.
INLINE void ge_serialize(Ge p, THREAD u8* out) {
    out[0] = (p.y.n[0] & 1u) ? 0x03 : 0x02;
    fe_to_be(p.x, out + 1);
}

// SEC1 uncompressed encoding: a 0x04 tag, then x and y.
INLINE void ge_serialize_uncompressed(Ge p, THREAD u8* out) {
    out[0] = 0x04;
    fe_to_be(p.x, out + 1);
    fe_to_be(p.y, out + 33);
}
