// SHA-256, for the BIP39 checksum and both halves of hash160.
//
// A mirror of what `sha2` does on the CPU, minus its runtime backend selection: there is
// no SHA-256 instruction to dispatch to here, so this is the straightforward FIPS 180-4
// compression and nothing else.
//
// Every message this program hashes is short and of known length -- 16/24/32 bytes of
// entropy, a 33- or 65-byte public key, a 22-byte script, a 32-byte digest -- so the
// interface takes a length rather than streaming, and two blocks is the most it will ever
// need. That bound is asserted by construction in `sha256`, not just assumed.

CONSTANT u32 SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

typedef struct { u32 h[8]; } Sha256;

INLINE Sha256 sha256_init() {
    Sha256 s;
    s.h[0] = 0x6a09e667u; s.h[1] = 0xbb67ae85u; s.h[2] = 0x3c6ef372u; s.h[3] = 0xa54ff53au;
    s.h[4] = 0x510e527fu; s.h[5] = 0x9b05688cu; s.h[6] = 0x1f83d9abu; s.h[7] = 0x5be0cd19u;
    return s;
}

INLINE u32 rotr32(u32 x, u32 n) { return (x >> n) | (x << (32 - n)); }

#define SHA256_ROUND(k, wi)                                             \
    do {                                                                \
        u32 S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);          \
        u32 ch = (e & f) ^ (~e & g);                                    \
        u32 t1 = h + S1 + ch + (k) + (wi);                              \
        u32 S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);          \
        u32 mj = (a & b) ^ (a & c) ^ (b & c);                           \
        u32 t2 = S0 + mj;                                               \
        h = g; g = f; f = e; e = d + t1;                                \
        d = c; c = b; b = a; a = t1 + t2;                               \
    } while (0)

// A rolling sixteen-word schedule, unrolled in groups of sixteen so the ring buffer's
// subscripts are compile-time constants and `w` stays in registers. See the long note in
// sha512.h -- this is the same transformation, for the same reason, over four groups
// instead of five.
INLINE Sha256 sha256_compress_w(Sha256 s, THREAD u32* w) {
    u32 a = s.h[0], b = s.h[1], c = s.h[2], d = s.h[3];
    u32 e = s.h[4], f = s.h[5], g = s.h[6], h = s.h[7];

    UNROLL for (u32 j = 0; j < 16; j++) {
        SHA256_ROUND(SHA256_K[j], w[j]);
    }
    for (u32 base = 16; base < 64; base += 16) {
        UNROLL for (u32 j = 0; j < 16; j++) {
            u32 x = w[(j + 1) & 15], y = w[(j + 14) & 15];
            u32 s0 = rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
            u32 s1 = rotr32(y, 17) ^ rotr32(y, 19) ^ (y >> 10);
            u32 wi = w[j] + s0 + w[(j + 9) & 15] + s1;
            w[j] = wi;
            SHA256_ROUND(SHA256_K[base + j], wi);
        }
    }

    s.h[0] += a; s.h[1] += b; s.h[2] += c; s.h[3] += d;
    s.h[4] += e; s.h[5] += f; s.h[6] += g; s.h[7] += h;
    return s;
}

INLINE Sha256 sha256_compress(Sha256 s, THREAD const u8* block) {
    u32 w[16];
    UNROLL for (u32 i = 0; i < 16; i++) {
        w[i] = ((u32)block[i * 4] << 24) | ((u32)block[i * 4 + 1] << 16)
             | ((u32)block[i * 4 + 2] << 8) | (u32)block[i * 4 + 3];
    }
    return sha256_compress_w(s, w);
}

// SHA-256 of a message of at most 119 bytes, i.e. one or two blocks after padding.
// The longest this program hashes is a 65-byte uncompressed public key.
//
// The staging block below looks wasteful -- 128 zeroed bytes per call, on a kernel that
// runs three times per key -- and building the schedule directly from `msg` instead was
// written and measured at 0.83s against 0.85s for the whole leaf kernel, i.e. nothing.
// The readable form stays. Unlike SHA-512's inner loop, this is not where the time is.
INLINE void sha256_hash(THREAD const u8* msg, u32 len, THREAD u8* out) {
    u8 block[128];
    for (u32 i = 0; i < 128; i++) block[i] = 0;
    for (u32 i = 0; i < len; i++) block[i] = msg[i];
    block[len] = 0x80;

    // One block if the terminator and the 8-byte length both fit under 64, else two.
    u32 blocks = (len + 1 + 8 <= 64) ? 1 : 2;
    u64 bits = (u64)len * 8;
    u32 tail = blocks * 64;
    for (u32 i = 0; i < 8; i++) block[tail - 1 - i] = (u8)(bits >> (8 * i));

    Sha256 s = sha256_init();
    for (u32 b = 0; b < blocks; b++) s = sha256_compress(s, block + b * 64);

    UNROLL for (u32 i = 0; i < 8; i++) {
        out[i * 4]     = (u8)(s.h[i] >> 24);
        out[i * 4 + 1] = (u8)(s.h[i] >> 16);
        out[i * 4 + 2] = (u8)(s.h[i] >> 8);
        out[i * 4 + 3] = (u8)(s.h[i]);
    }
}
