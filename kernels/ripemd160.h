// RIPEMD-160, the second half of hash160.
//
// Only ever applied to a 32-byte SHA-256 digest, so a single block covers every call this
// program makes and there is no streaming interface. Little-endian throughout, which is
// the one thing that catches a port written from a SHA-shaped mental model.

CONSTANT u32 RMD_KL[5] = { 0x00000000u, 0x5a827999u, 0x6ed9eba1u, 0x8f1bbcdcu, 0xa953fd4eu };
CONSTANT u32 RMD_KR[5] = { 0x50a28be6u, 0x5c4dd124u, 0x6d703ef3u, 0x7a6d76e9u, 0x00000000u };

CONSTANT u8 RMD_RL[80] = {
     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
     7, 4,13, 1,10, 6,15, 3,12, 0, 9, 5, 2,14,11, 8,
     3,10,14, 4, 9,15, 8, 1, 2, 7, 0, 6,13,11, 5,12,
     1, 9,11,10, 0, 8,12, 4,13, 3, 7,15,14, 5, 6, 2,
     4, 0, 5, 9, 7,12, 2,10,14, 1, 3, 8,11, 6,15,13,
};
CONSTANT u8 RMD_RR[80] = {
     5,14, 7, 0, 9, 2,11, 4,13, 6,15, 8, 1,10, 3,12,
     6,11, 3, 7, 0,13, 5,10,14,15, 8,12, 4, 9, 1, 2,
    15, 5, 1, 3, 7,14, 6, 9,11, 8,12, 2,10, 0, 4,13,
     8, 6, 4, 1, 3,11,15, 0, 5,12, 2,13, 9, 7,10,14,
    12,15,10, 4, 1, 5, 8, 7, 6, 2,13,14, 0, 3, 9,11,
};
CONSTANT u8 RMD_SL[80] = {
    11,14,15,12, 5, 8, 7, 9,11,13,14,15, 6, 7, 9, 8,
     7, 6, 8,13,11, 9, 7,15, 7,12,15, 9,11, 7,13,12,
    11,13, 6, 7,14, 9,13,15,14, 8,13, 6, 5,12, 7, 5,
    11,12,14,15,14,15, 9, 8, 9,14, 5, 6, 8, 6, 5,12,
     9,15, 5,11, 6, 8,13,12, 5,12,13,14,11, 8, 5, 6,
};
CONSTANT u8 RMD_SR[80] = {
     8, 9, 9,11,13,15,15, 5, 7, 7, 8,11,14,14,12, 6,
     9,13,15, 7,12, 8, 9,11, 7, 7,12, 7, 6,15,13,11,
     9, 7,15,11, 8, 6, 6,14,12,13, 5,14,13,13, 7, 5,
    15, 5, 8,11,14,14, 6,14, 6, 9,12, 9,12, 5,15, 8,
     8, 5,12, 9,12, 5,14, 6, 8,13, 6, 5,15,13,11,11,
};

INLINE u32 rotl32(u32 x, u32 n) { return (x << n) | (x >> (32 - n)); }

// The five round functions, selected by round group.
INLINE u32 rmd_f(u32 j, u32 x, u32 y, u32 z) {
    if (j < 16) return x ^ y ^ z;
    if (j < 32) return (x & y) | (~x & z);
    if (j < 48) return (x | ~y) ^ z;
    if (j < 64) return (x & z) | (y & ~z);
    return x ^ (y | ~z);
}

// RIPEMD-160 of exactly 32 bytes -- the only length this program needs.
//
// The round loop is **not** unrolled, unlike the two SHA schedules. It looks like the same
// case and is not quite: both lanes index the message as `x[RMD_RL[j]]`, a subscript read
// out of a table rather than computed from the round number, so only a full eighty-round
// unroll would make `x`'s subscripts constant -- the grouped form that works in sha256.h
// cannot help here. That was tried and measured at exactly no change to `k_leaf`
// (0.12s either way, M1 Pro, interleaved A/B), so the unroll buys nothing and would only
// spend NVRTC's patience. See the note in ec.h about what that costs when it runs out.
INLINE void ripemd160_32(THREAD const u8* msg, THREAD u8* out) {
    u32 x[16];
    for (u32 i = 0; i < 8; i++) {
        x[i] = (u32)msg[i * 4] | ((u32)msg[i * 4 + 1] << 8)
             | ((u32)msg[i * 4 + 2] << 16) | ((u32)msg[i * 4 + 3] << 24);
    }
    x[8] = 0x80u;                 // terminator, immediately after 32 bytes
    for (u32 i = 9; i < 14; i++) x[i] = 0;
    x[14] = 32u * 8u;             // bit length, little-endian
    x[15] = 0;

    u32 al = 0x67452301u, bl = 0xefcdab89u, cl = 0x98badcfeu, dl = 0x10325476u, el = 0xc3d2e1f0u;
    u32 ar = al, br = bl, cr = cl, dr = dl, er = el;

    for (u32 j = 0; j < 80; j++) {
        u32 g = j / 16;
        u32 t = rotl32(al + rmd_f(j, bl, cl, dl) + x[RMD_RL[j]] + RMD_KL[g], RMD_SL[j]) + el;
        al = el; el = dl; dl = rotl32(cl, 10); cl = bl; bl = t;

        t = rotl32(ar + rmd_f(79 - j, br, cr, dr) + x[RMD_RR[j]] + RMD_KR[g], RMD_SR[j]) + er;
        ar = er; er = dr; dr = rotl32(cr, 10); cr = br; br = t;
    }

    u32 h[5];
    h[0] = 0xefcdab89u + cl + dr;
    h[1] = 0x98badcfeu + dl + er;
    h[2] = 0x10325476u + el + ar;
    h[3] = 0xc3d2e1f0u + al + br;
    h[4] = 0x67452301u + bl + cr;

    for (u32 i = 0; i < 5; i++) {
        out[i * 4]     = (u8)(h[i]);
        out[i * 4 + 1] = (u8)(h[i] >> 8);
        out[i * 4 + 2] = (u8)(h[i] >> 16);
        out[i * 4 + 3] = (u8)(h[i] >> 24);
    }
}

// RIPEMD160(SHA256(data)) -- the 20 bytes an ecloop filter holds.
INLINE void hash160(THREAD const u8* data, u32 len, THREAD u8* out) {
    u8 sha[32];
    sha256_hash(data, len, sha);
    ripemd160_32(sha, out);
}
