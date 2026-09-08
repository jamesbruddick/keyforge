// SHA-512, HMAC-SHA512, and the PBKDF2 BIP39 stretch.
//
// **This is the most performance-sensitive file in the project.** PBKDF2 is 46% of the
// GPU's work per seed -- against 23% on the CPU -- because the CPU has `sha512h` and
// `sha512su0` instructions and the GPU has no 64-bit integer ALU at all. Every ulong
// operation below lowers to a pair of 32-bit ones.
//
// The obvious response is to hand-split the state into hi/lo `uint` pairs and do the
// carries manually. **That was measured and it is slower**: 169.7 M compressions/s split
// against 178.6 M/s native, on an M1 Pro (2026-08-13, 262,144 serial chains, both
// verified against the RFC 6234 "abc" vector). The Metal compiler lowers 64-bit
// arithmetic better than the hand-written version does, so the plain spelling stays.
//
// The 64-bit operations are still routed through named helpers rather than written inline.
// That costs nothing -- they inline away -- and it means the split experiment is a
// one-file change if CUDA ever disagrees with Metal about this.
//
// **CUDA was asked and it agrees with Metal**, so both dialects keep the plain spelling.
//
// The obvious asymmetry says it should not have: NVIDIA has an instruction Apple has no
// equivalent of -- `SHF`, the funnel shift -- and a 64-bit rotate is exactly two of them,
// against the shift-shift-or-or this lowers to on a machine with no 64-bit ALU. So `rotr64`
// was written as a pair of `__funnelshift_r` calls, verified against this form for all of
// 1..=63 over edge cases and for SHA-512's own ten distances over four million random
// inputs, and measured on an RTX 5070 Ti at the default scope, 3,000,000 seeds against the
// real filter, interleaved with the order alternated each repetition:
//
//     funnel shift   107,085 seeds/s
//     plain          107,227 seeds/s
//
// Slower, and the margin is real rather than noise: unchanged configurations elsewhere in
// that sweep repeat to within 16-73 seeds/s, where these two groups sit 151 apart and are
// each internally tight. **nvcc already emits the funnel shift for the expression below.**
// The rotate distances are compile-time constants at every call site, which is exactly the
// case the peephole recognises, so spelling it out by hand added shuffling and removed
// nothing.
//
// The lesson generalises past this one function, which is why it is written down at length:
// the reason the split lost on Metal in 2026-08 and the reason it lost on CUDA are the
// same reason. Both compilers were already doing it. Reach for a dialect-specific spelling
// only after the generated code has been read, not because the hardware has an instruction
// that ought to help.
//
// The CPU's `sha512x2` two-lane scheduling is deliberately *not* ported. It exists because
// one PBKDF2 chain cannot keep an ARMv8 SHA-512 unit busy; a launch here has tens of
// thousands of independent chains, so one stream per thread already saturates.
//
// ## `k_pbkdf2` is finished, and the 784-byte spill figure is not what it looks like
//
// Profiling an RTX 5070 Ti reports `local = 784` for this kernel -- bytes of thread-local
// memory per thread -- which on the kernel that is 39% of a launch, and whose whole design
// is about keeping the message schedule in registers, reads like a defect. It is not, and
// this is written down so the number does not send anyone back here.
//
// **`local` is the static frame size: the maximum over every path, not what the hot loop
// touches.** Summing every local array in this kernel's call graph gives 900 bytes, of
// which ptxas overlaps ~116 into 784. Where they live matters more than the total:
//
// ```text
//   setup, dead before the stretch     568   phrase[216], e[32], k0[128], pad[128],
//                                            folded[64] -- all inside hmac512_new and the
//                                            scoped block in k_pbkdf2 that builds it
//   hot loop                           256   u[8], acc[8], w[16], as words
// ```
//
// The setup is 72% of the figure and runs once. `k_pbkdf2` already scopes it so it is dead
// before the 2,048 iterations begin -- that was the fix, and the fix is still in place; the
// frame is merely still *allocated* for it, which costs an address range and no traffic.
//
// **Two independent measurements say the hot loop is not paying for local memory.** First,
// capping registers to 64 -- which more than doubles resident warps and must spill heavily
// to get there -- changed the whole sweep by 0.4%, and a kernel bound on local memory
// traffic does not shrug off a large increase in local memory traffic. Second, the achieved
// rate is close to the machine: 36.9 billion compressions in 10.96s is 3.36 G/s, and at
// roughly 4,700 integer operations a compression (80 rounds at ~38, plus 64 schedule steps
// at ~26, with `ch` and `maj` folding into LOP3 and each 64-bit rotate costing two funnel
// shifts) that is ~15.8 T operations/s against this card's ~22 T. About 72%, on a kernel
// whose rounds are a serial dependency chain. Call that estimate good to a third either
// way and the conclusion does not move.
//
// So there is no structural win here. What is left is arithmetic worth ~1.8% -- peeling the
// `base == 16` group so the constant `w[8..15]` fold, discussed at length in
// `sha512_compress_w` below, measured at -3.8% on Metal and never tried on CUDA. It is
// recorded rather than done because the ceiling is small and the last Metal-to-CUDA
// extrapolation attempted in this file (the funnel-shift rotate above) was wrong.

INLINE u64 rotr64(u64 x, u32 n) { return (x >> n) | (x << (64 - n)); }
INLINE u64 shr64(u64 x, u32 n)  { return x >> n; }

CONSTANT u64 SHA512_K[80] = {
    0x428a2f98d728ae22UL, 0x7137449123ef65cdUL, 0xb5c0fbcfec4d3b2fUL, 0xe9b5dba58189dbbcUL,
    0x3956c25bf348b538UL, 0x59f111f1b605d019UL, 0x923f82a4af194f9bUL, 0xab1c5ed5da6d8118UL,
    0xd807aa98a3030242UL, 0x12835b0145706fbeUL, 0x243185be4ee4b28cUL, 0x550c7dc3d5ffb4e2UL,
    0x72be5d74f27b896fUL, 0x80deb1fe3b1696b1UL, 0x9bdc06a725c71235UL, 0xc19bf174cf692694UL,
    0xe49b69c19ef14ad2UL, 0xefbe4786384f25e3UL, 0x0fc19dc68b8cd5b5UL, 0x240ca1cc77ac9c65UL,
    0x2de92c6f592b0275UL, 0x4a7484aa6ea6e483UL, 0x5cb0a9dcbd41fbd4UL, 0x76f988da831153b5UL,
    0x983e5152ee66dfabUL, 0xa831c66d2db43210UL, 0xb00327c898fb213fUL, 0xbf597fc7beef0ee4UL,
    0xc6e00bf33da88fc2UL, 0xd5a79147930aa725UL, 0x06ca6351e003826fUL, 0x142929670a0e6e70UL,
    0x27b70a8546d22ffcUL, 0x2e1b21385c26c926UL, 0x4d2c6dfc5ac42aedUL, 0x53380d139d95b3dfUL,
    0x650a73548baf63deUL, 0x766a0abb3c77b2a8UL, 0x81c2c92e47edaee6UL, 0x92722c851482353bUL,
    0xa2bfe8a14cf10364UL, 0xa81a664bbc423001UL, 0xc24b8b70d0f89791UL, 0xc76c51a30654be30UL,
    0xd192e819d6ef5218UL, 0xd69906245565a910UL, 0xf40e35855771202aUL, 0x106aa07032bbd1b8UL,
    0x19a4c116b8d2d0c8UL, 0x1e376c085141ab53UL, 0x2748774cdf8eeb99UL, 0x34b0bcb5e19b48a8UL,
    0x391c0cb3c5c95a63UL, 0x4ed8aa4ae3418acbUL, 0x5b9cca4f7763e373UL, 0x682e6ff3d6b2b8a3UL,
    0x748f82ee5defb2fcUL, 0x78a5636f43172f60UL, 0x84c87814a1f0ab72UL, 0x8cc702081a6439ecUL,
    0x90befffa23631e28UL, 0xa4506cebde82bde9UL, 0xbef9a3f7b2c67915UL, 0xc67178f2e372532bUL,
    0xca273eceea26619cUL, 0xd186b8c721c0c207UL, 0xeada7dd6cde0eb1eUL, 0xf57d4f7fee6ed178UL,
    0x06f067aa72176fbaUL, 0x0a637dc5a2c898a6UL, 0x113f9804bef90daeUL, 0x1b710b35131c471bUL,
    0x28db77f523047d84UL, 0x32caab7b40c72493UL, 0x3c9ebe0a15c9bebcUL, 0x431d67c49c100d4cUL,
    0x4cc5d4becb3e42b6UL, 0x597f299cfc657e2aUL, 0x5fcb6fab3ad6faecUL, 0x6c44198c4a475817UL,
};

#define SHA512_BLOCK 128
#define SHA512_OUT   64

typedef struct { u64 h[8]; } Sha512;

INLINE Sha512 sha512_init() {
    Sha512 s;
    s.h[0] = 0x6a09e667f3bcc908UL; s.h[1] = 0xbb67ae8584caa73bUL;
    s.h[2] = 0x3c6ef372fe94f82bUL; s.h[3] = 0xa54ff53a5f1d36f1UL;
    s.h[4] = 0x510e527fade682d1UL; s.h[5] = 0x9b05688c2b3e6c1fUL;
    s.h[6] = 0x1f83d9abfb41bd6bUL; s.h[7] = 0x5be0cd19137e2179UL;
    return s;
}

// One compression, with a **rolling sixteen-word message schedule**.
//
// The textbook form materialises all eighty words. At eight bytes each that is 640 bytes
// of thread-private state, which does not fit the register file and spills to memory --
// and this kernel is the largest share of the sweep, so it spills on the hot path. The
// recurrence only ever reaches back sixteen words, so a ring buffer of sixteen is
// sufficient and is a quarter of the footprint: `i-16` is `i mod 16`, `i-15` is
// `(i+1) mod 16`, `i-7` is `(i+9) mod 16`, `i-2` is `(i+14) mod 16`.
//
// Identical output -- `sha512_matches_the_cpu` covers it -- for a quarter of the state.
// One round of the compression function, given this round's message word.
#define SHA512_ROUND(k, wi)                                             \
    do {                                                                \
        u64 S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);         \
        u64 ch = (e & f) ^ (~e & g);                                    \
        u64 t1 = h + S1 + ch + (k) + (wi);                              \
        u64 S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);         \
        u64 mj = (a & b) ^ (a & c) ^ (b & c);                           \
        u64 t2 = S0 + mj;                                               \
        h = g; g = f; f = e; e = d + t1;                                \
        d = c; c = b; b = a; a = t1 + t2;                               \
    } while (0)

INLINE Sha512 sha512_compress_w(Sha512 s, THREAD u64* w) {
    u64 a = s.h[0], b = s.h[1], c = s.h[2], d = s.h[3];
    u64 e = s.h[4], f = s.h[5], g = s.h[6], h = s.h[7];

    // **The round loop is unrolled in groups of sixteen, and that is the whole point of
    // this shape.** The ring buffer indexes itself as `w[i & 15]`, so with a single loop
    // over eighty rounds every subscript is a function of the loop variable -- and a local
    // array that is dynamically indexed cannot be held in registers. It goes to
    // thread-local memory, which on an Apple GPU is device-backed, and the schedule then
    // pays four loads and a store *per round*: ~400 memory operations per compression, on
    // the kernel that is the largest single share of the sweep.
    //
    // Grouping by sixteen is what fixes it without unrolling all eighty rounds. Rounds are
    // taken `base + j` for `j` in 0..16, and since `base` is a multiple of sixteen,
    // `(base + j) & 15` is just `j` -- so after the inner loop unrolls, every subscript is
    // a compile-time constant and `w` becomes sixteen registers. Only `SHA512_K[base + j]`
    // stays dynamic, and that is a constant-memory read, which is what constant memory is
    // for.
    //
    // Sixteen rather than eighty deliberately: see the note in ec.h about NVRTC honouring
    // an unroll pragma literally and never finishing. Five copies of a round is small.
    //
    // **Peeling the `base == 16` copy was tried and is slower.** The argument for it is
    // real: `hmac512_mac_words` seeds `w[8..15]` with `0x80…`, six zeroes and a fixed
    // 1536-bit length, and the moment `w` becomes loop-carried across the `base` loop the
    // compiler must treat all sixteen as live values, so every one of those constants is
    // lost. Peeled, the folding is worth about a hundred 32-bit operations per compression
    // -- eight `+ w[j]` addends in rounds 0-15, six zero `w[(j+9) & 15]` addends, seven
    // constant `s0` terms and two constant `s1` terms in the 16-31 schedule.
    //
    // It measured 3.42s -> 3.55s in `k_pbkdf2` and 2,646 -> 2,599 seeds/s over the whole
    // sweep (M1 Pro, default scope, 20,000 seeds at a 2,048 batch, medians of three).
    // Reproducibly worse, and the reason is that peeling makes a *third* inline copy of the
    // sixteen-round body: the arithmetic saved is real but it is spent, and more, on
    // instruction footprint inside a loop that runs 2,048 times per thread. The constants
    // are not the scarce resource here; the instruction cache is.
    UNROLL for (u32 j = 0; j < 16; j++) {
        SHA512_ROUND(SHA512_K[j], w[j]);
    }
    for (u32 base = 16; base < 80; base += 16) {
        UNROLL for (u32 j = 0; j < 16; j++) {
            u64 x = w[(j + 1) & 15], y = w[(j + 14) & 15];
            u64 s0 = rotr64(x, 1) ^ rotr64(x, 8) ^ shr64(x, 7);
            u64 s1 = rotr64(y, 19) ^ rotr64(y, 61) ^ shr64(y, 6);
            u64 wi = w[j] + s0 + w[(j + 9) & 15] + s1;
            w[j] = wi;
            SHA512_ROUND(SHA512_K[base + j], wi);
        }
    }

    s.h[0] += a; s.h[1] += b; s.h[2] += c; s.h[3] += d;
    s.h[4] += e; s.h[5] += f; s.h[6] += g; s.h[7] += h;
    return s;
}

// The same, from a byte block. Used everywhere except the PBKDF2 inner loop, which has its
// message in words already and must not pay to convert it twice per iteration.
INLINE Sha512 sha512_compress(Sha512 s, THREAD const u8* block) {
    u64 w[16];
    for (u32 i = 0; i < 16; i++) {
        u64 v = 0;
        for (u32 b = 0; b < 8; b++) v = (v << 8) | (u64)block[i * 8 + b];
        w[i] = v;
    }
    return sha512_compress_w(s, w);
}

INLINE void sha512_digest(Sha512 s, THREAD u8* out) {
    for (u32 i = 0; i < 8; i++)
        for (u32 b = 0; b < 8; b++) out[i * 8 + b] = (u8)(s.h[i] >> (56 - 8 * b));
}

// Absorb a final partial message and apply the padding.
//
// `total_len` is the length of the *whole* hashed message, including any blocks already
// folded into `s`; `len` must leave room for the terminator and the 16-byte length, i.e.
// at most 111 bytes. Mirrors `pbkdf2::finish`.
INLINE Sha512 sha512_finish(Sha512 s, THREAD const u8* tail, u32 len, u64 total_len) {
    u8 block[SHA512_BLOCK];
    for (u32 i = 0; i < SHA512_BLOCK; i++) block[i] = 0;
    for (u32 i = 0; i < len; i++) block[i] = tail[i];
    block[len] = 0x80;
    u64 bits = total_len * 8;
    for (u32 i = 0; i < 8; i++) block[SHA512_BLOCK - 1 - i] = (u8)(bits >> (8 * i));
    return sha512_compress(s, block);
}

// SHA-512 of an arbitrary short message, used only to fold an over-long HMAC key.
INLINE void sha512_hash(THREAD const u8* msg, u32 len, THREAD u8* out) {
    Sha512 s = sha512_init();
    u32 at = 0;
    while (len - at >= SHA512_BLOCK) {
        s = sha512_compress(s, msg + at);
        at += SHA512_BLOCK;
    }
    u8 tail[SHA512_BLOCK];
    u32 rest = len - at;
    for (u32 i = 0; i < rest; i++) tail[i] = msg[at + i];
    // A tail with no room for the length field needs one more all-padding block.
    if (rest + 1 + 16 > SHA512_BLOCK) {
        u8 full[SHA512_BLOCK];
        for (u32 i = 0; i < SHA512_BLOCK; i++) full[i] = (i < rest) ? tail[i] : 0;
        full[rest] = 0x80;
        s = sha512_compress(s, full);
        u8 empty[SHA512_BLOCK];
        for (u32 i = 0; i < SHA512_BLOCK; i++) empty[i] = 0;
        u64 bits = (u64)len * 8;
        for (u32 i = 0; i < 8; i++) empty[SHA512_BLOCK - 1 - i] = (u8)(bits >> (8 * i));
        s = sha512_compress(s, empty);
    } else {
        s = sha512_finish(s, tail, rest, (u64)len);
    }
    sha512_digest(s, out);
}

// HMAC-SHA512 with the key's pad midstates precomputed.
//
// The same specialisation `pbkdf2::HmacSha512` makes, and for the same reason: the key is
// constant across all 2048 iterations, so absorbing the two 128-byte pad blocks once
// instead of per iteration halves the compressions.
typedef struct { Sha512 ipad; Sha512 opad; } Hmac512;

INLINE Hmac512 hmac512_new(THREAD const u8* key, u32 key_len) {
    u8 k0[SHA512_BLOCK];
    for (u32 i = 0; i < SHA512_BLOCK; i++) k0[i] = 0;
    if (key_len > SHA512_BLOCK) {
        // Live for a 24-word mnemonic, which exceeds 128 bytes.
        u8 folded[SHA512_OUT];
        sha512_hash(key, key_len, folded);
        for (u32 i = 0; i < SHA512_OUT; i++) k0[i] = folded[i];
    } else {
        for (u32 i = 0; i < key_len; i++) k0[i] = key[i];
    }

    u8 pad[SHA512_BLOCK];
    Hmac512 h;
    for (u32 i = 0; i < SHA512_BLOCK; i++) pad[i] = k0[i] ^ 0x36;
    h.ipad = sha512_compress(sha512_init(), pad);
    for (u32 i = 0; i < SHA512_BLOCK; i++) pad[i] = k0[i] ^ 0x5c;
    h.opad = sha512_compress(sha512_init(), pad);
    return h;
}

// HMAC over a message short enough to finish in one compression each side. Both PBKDF2
// messages qualify: `salt || INT(1)`, and a 64-byte previous block.
INLINE void hmac512_mac(Hmac512 h, THREAD const u8* msg, u32 len, THREAD u8* out) {
    u8 inner[SHA512_OUT];
    sha512_digest(sha512_finish(h.ipad, msg, len, (u64)(SHA512_BLOCK + len)), inner);
    sha512_digest(
        sha512_finish(h.opad, inner, SHA512_OUT, (u64)(SHA512_BLOCK + SHA512_OUT)), out);
}

// PBKDF2-HMAC-SHA512 producing exactly 64 bytes, from a key whose midstates are already
// built.
//
// dkLen is one SHA-512 block, so there is a single T and no outer block loop. Cost is
// 2 + 2*iterations compressions, which is the whole reason for the midstates.
//
// **Taking the `Hmac512` rather than the phrase is what makes this fast**, and it is not
// a stylistic choice. The phrase is up to 216 bytes and is needed only to build the two
// midstates; passing it in leaves it live across all 2,048 iterations, and the register
// file does not have room for it alongside the compression state. Hoisting it out of the
// loop's scope took this kernel from 55.7 to 62.4 M compressions/s on an M1 Pro
// (2026-08-13) -- 12%, which is worth having but is nowhere near the 178 M/s the same
// compression reaches in isolation. The remaining gap is the rest of the live state: two
// 64-byte midstates, the chaining value and the accumulator, against a bare compression's
// message schedule alone.
//
// The chaining value is also updated in place instead of through a second buffer.
// `hmac512_mac` consumes its message into the inner compression before it writes a byte
// of its output, so passing the same array as both is safe and saves another 64 bytes.
// One HMAC over a message that is exactly 64 bytes, held as words.
//
// Both compressions of a PBKDF2 iteration have this shape, and the shape is what makes the
// specialisation worth writing: the padding is a compile-time constant (0x80, zeroes, and
// a fixed 1536-bit length), so no block is assembled and no byte ever appears. The general
// path builds and zeroes a 128-byte array twice per iteration and converts the message to
// words and the digest back to bytes -- 2,048 times.
INLINE void hmac512_mac_words(Hmac512 h, THREAD const u64* msg, THREAD u64* out) {
    u64 w[16];
    UNROLL for (u32 i = 0; i < 8; i++) w[i] = msg[i];
    w[8] = 0x8000000000000000UL;
    UNROLL for (u32 i = 9; i < 15; i++) w[i] = 0;
    w[15] = (u64)(SHA512_BLOCK + SHA512_OUT) * 8UL;
    Sha512 inner = sha512_compress_w(h.ipad, w);

    UNROLL for (u32 i = 0; i < 8; i++) w[i] = inner.h[i];
    w[8] = 0x8000000000000000UL;
    UNROLL for (u32 i = 9; i < 15; i++) w[i] = 0;
    w[15] = (u64)(SHA512_BLOCK + SHA512_OUT) * 8UL;
    Sha512 outer = sha512_compress_w(h.opad, w);
    UNROLL for (u32 i = 0; i < 8; i++) out[i] = outer.h[i];
}

INLINE void pbkdf2_from_hmac(Hmac512 h, u32 iterations, THREAD u8* out) {
    // salt = "mnemonic", then the big-endian block counter 1.
    u8 first[12];
    first[0] = 'm'; first[1] = 'n'; first[2] = 'e'; first[3] = 'm';
    first[4] = 'o'; first[5] = 'n'; first[6] = 'i'; first[7] = 'c';
    first[8] = 0; first[9] = 0; first[10] = 0; first[11] = 1;

    // Only the first iteration has an odd-length message, so it takes the general path.
    u8 first_out[SHA512_OUT];
    hmac512_mac(h, first, 12, first_out);

    u64 u[8], acc[8];
    UNROLL for (u32 i = 0; i < 8; i++) {
        u64 v = 0;
        for (u32 b = 0; b < 8; b++) v = (v << 8) | (u64)first_out[i * 8 + b];
        u[i] = v;
        acc[i] = v;
    }

    // Every later iteration is 64 bytes in and 64 out, so it stays in words throughout.
    for (u32 it = 1; it < iterations; it++) {
        hmac512_mac_words(h, u, u);
        UNROLL for (u32 i = 0; i < 8; i++) acc[i] ^= u[i];
    }

    UNROLL for (u32 i = 0; i < 8; i++)
        for (u32 b = 0; b < 8; b++) out[i * 8 + b] = (u8)(acc[i] >> (56 - 8 * b));
}

// The whole BIP39 stretch, for callers that still have the phrase in hand. The parity
// tests use this; the pipeline builds the midstates itself so the phrase can go out of
// scope first.
INLINE void bip39_seed(THREAD const u8* phrase, u32 phrase_len, u32 iterations,
                       THREAD u8* out) {
    pbkdf2_from_hmac(hmac512_new(phrase, phrase_len), iterations, out);
}
