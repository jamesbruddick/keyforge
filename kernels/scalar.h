// Arithmetic modulo n, the order of the secp256k1 group.
//
// Separate from field.h, which works modulo p. BIP32's CKDpriv is
// `child = parse256(IL) + k_par (mod n)`, and n is not p -- using the wrong modulus gives
// keys that are wrong by an amount too small to notice and too large to recover from.
//
// This is the whole of what the CPU reaches libsecp256k1 for: `src/derive.rs` uses
// `Scalar::from_be_bytes` and `SecretKey::add_tweak` and nothing else from it. Eight
// 32-bit limbs, same shape and same reasons as `Fe`.

// n, least significant limb first.
CONSTANT u32 SC_N[8] = {
    0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
    0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
};

typedef struct { u32 n[8]; } Scalar;

INLINE bool sc_is_zero(Scalar a) {
    u32 acc = 0;
    for (u32 i = 0; i < 8; i++) acc |= a.n[i];
    return acc == 0;
}

// Is `a` at or above n? Compared from the top down.
INLINE bool sc_ge_n(Scalar a) {
    for (int i = 7; i >= 0; i--) {
        if (a.n[i] != SC_N[i]) return a.n[i] > SC_N[i];
    }
    return true; // exactly n
}

INLINE Scalar sc_sub_n(Scalar a) {
    Scalar r;
    u64 borrow = 0;
    for (u32 i = 0; i < 8; i++) {
        u64 d = (u64)a.n[i] - (u64)SC_N[i] - borrow;
        r.n[i] = (u32)d;
        borrow = (d >> 32) & 1;
    }
    return r;
}

// (a + b) mod n, for a and b already below n.
INLINE Scalar sc_add(Scalar a, Scalar b) {
    Scalar r;
    u64 c = 0;
    for (u32 i = 0; i < 8; i++) {
        c += (u64)a.n[i] + (u64)b.n[i];
        r.n[i] = (u32)c;
        c >>= 32;
    }
    // a + b < 2n < 2^257, so one conditional subtract settles it. The carry out means the
    // sum passed 2^256, which is necessarily above n.
    if (c || sc_ge_n(r)) r = sc_sub_n(r);
    return r;
}

INLINE Scalar sc_from_be(THREAD const u8* b) {
    Scalar r;
    for (u32 i = 0; i < 8; i++) {
        u32 j = (7 - i) * 4;
        r.n[i] = ((u32)b[j] << 24) | ((u32)b[j + 1] << 16) | ((u32)b[j + 2] << 8) | (u32)b[j + 3];
    }
    return r;
}

INLINE void sc_to_be(Scalar a, THREAD u8* b) {
    for (u32 i = 0; i < 8; i++) {
        u32 j = (7 - i) * 4;
        b[j]     = (u8)(a.n[i] >> 24);
        b[j + 1] = (u8)(a.n[i] >> 16);
        b[j + 2] = (u8)(a.n[i] >> 8);
        b[j + 3] = (u8)(a.n[i]);
    }
}

// A valid secret key is in 1..n. Both failure modes are ~2^-127 events that BIP32 says to
// skip rather than clamp, and skipping is what the CPU does too.
INLINE bool sc_is_valid(Scalar a) { return !sc_is_zero(a) && !sc_ge_n(a); }
