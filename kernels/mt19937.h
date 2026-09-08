// The weak generators this program sweeps, on the GPU.
//
// A mirror of src/mt19937.rs, and the three details that decide whether a port reproduces
// a real byte stream or plausible-looking garbage are all here:
//
//   1. Seeding is `init_genrand` -- what `std::mt19937 twister(n)` does -- not
//      `init_by_array`, which is what Python's `random.seed(n)` does and which produces a
//      completely different stream from the same integer.
//   2. One of the three streams is not a C++ program at all: MT_PHP is PHP's mt_rand in
//      MT_RAND_PHP mode, whose *engine* is wrong (twist_php uses loBit(u) where the
//      reference uses loBit(v)) and whose byte is the naive high one. See `Dist::Php`.
//   3. Bytes come from `std::uniform_int_distribution`, which the standard specifies the
//      range of and nothing else, so the two mainstream C++ libraries compute it
//      differently and the same seed makes two unrelated wallets. libstdc++ is
//      `word / 16777215` with the short tail rejected -- not `word >> 24`; the divisor is
//      (2^32 - 1)/256, not 2^32/256, and an off-by-one there still yields mnemonics that
//      match nothing. libc++ masks the low byte instead. See `Dist` in src/mt19937.rs.
//
// **On the 624-word state.** That is 2,496 bytes of thread-private memory per lane, which
// is large enough to spill out of the register file and hurt this kernel's occupancy. It
// is kept anyway, because the alternative is a windowed recurrence that only works for
// small offsets, and because the arithmetic here is ~0.02% of the work per seed: even
// fifty times worse than modelled it does not reach 1%. Revisit only with a measurement.

#define MT_N 624
#define MT_M 397
#define MT_MATRIX_A  0x9908b0dfu
#define MT_UPPER     0x80000000u
#define MT_LOWER     0x7fffffffu

// (2^32 - 1) / 256, and the first word value that must be rejected.
#define MT_SCALING 16777215u
#define MT_PAST    4294967040u

// Must match the `Dist` discriminants the host sends; see `Dist::code` in src/mt19937.rs.
#define MT_LIBSTDCXX 0u
#define MT_LIBCXX    1u
#define MT_PHP       2u

typedef struct {
    u32 state[MT_N];
    u32 index;
    // Twist with PHP's `loBit(u)` instead of the reference `loBit(v)`. Set from the
    // dist at seed time, because unlike the byte mapping this is not a choice that can
    // be made per draw -- it changes the state, so it has to be right from word zero.
    u32 php;
} Mt19937;

INLINE void mt_seed(THREAD Mt19937* m, u32 seed, u32 dist) {
    m->state[0] = seed;
    for (u32 i = 1; i < MT_N; i++) {
        u32 prev = m->state[i - 1];
        m->state[i] = 1812433253u * (prev ^ (prev >> 30)) + i;
    }
    m->index = MT_N;
    m->php = (dist == MT_PHP) ? 1u : 0u;
}

// CPython's `random.seed(n)`: `init_by_array` over a one-word key.
//
// **This is not `mt_seed` with the same number.** CPython converts the integer to an
// array of 32-bit words and runs `init_by_array`, which produces a completely unrelated
// stream from `init_genrand`. Scanning one while meaning the other is the worst failure
// available here: the sweep finishes, reports clean, and has checked nothing. Mirrors
// `Mt19937::from_key` in src/vuln/mt19937.rs, which is pinned against values a real
// interpreter printed.
//
// The reference takes a key of any length. A 32-bit seed is one word, so the key index
// `j` is zero on every iteration -- both the word read and the `+ j` addend -- and the
// loops below are the general algorithm with that folded in. A wider key would need `j`
// back; nothing here scans one.
INLINE void mt_seed_by_array(THREAD Mt19937* m, u32 key) {
    mt_seed(m, 19650218u, MT_LIBSTDCXX);
    u32 i = 1;
    for (u32 k = 0; k < MT_N; k++) {
        u32 prev = m->state[i - 1];
        m->state[i] = (m->state[i] ^ (1664525u * (prev ^ (prev >> 30)))) + key;
        i++;
        if (i >= MT_N) { m->state[0] = m->state[MT_N - 1]; i = 1; }
    }
    for (u32 k = 0; k < MT_N - 1; k++) {
        u32 prev = m->state[i - 1];
        m->state[i] = (m->state[i] ^ (1566083941u * (prev ^ (prev >> 30)))) - i;
        i++;
        if (i >= MT_N) { m->state[0] = m->state[MT_N - 1]; i = 1; }
    }
    // The reference sets the high bit of state[0] so the state is never all-zero.
    m->state[0] = 0x80000000u;
    m->index = MT_N;
}

INLINE void mt_twist(THREAD Mt19937* m) {
    for (u32 i = 0; i < MT_N; i++) {
        u32 u = m->state[i];
        u32 y = (u & MT_UPPER) | (m->state[(i + 1) % MT_N] & MT_LOWER);
        u32 next = m->state[(i + MT_M) % MT_N] ^ (y >> 1);
        // `y & 1` is loBit(v) -- the low bit of state[i+1], since MT_UPPER cleared it
        // from u. php-src's twist_php takes loBit(u) instead. The branch is uniform
        // across the grid: a launch walks one dist.
        u32 low = m->php ? (u & 1u) : (y & 1u);
        if (low) next ^= MT_MATRIX_A;
        m->state[i] = next;
    }
    m->index = 0;
}

INLINE u32 mt_next_u32(THREAD Mt19937* m) {
    if (m->index >= MT_N) mt_twist(m);
    u32 y = m->state[m->index];
    m->index++;
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    return y ^ (y >> 18);
}

// One entropy byte, as the C++ library named by `dist` produced it.
//
// The whole launch walks one `dist`, so the branch is uniform across every lane and costs
// nothing beyond the compare -- and this is ~0.02% of the work per seed either way.
INLINE u8 mt_next_byte(THREAD Mt19937* m, u32 dist) {
    // No rejection under libc++: 2^32 divides by 256 exactly, so every word is a byte.
    if (dist == MT_LIBCXX) return (u8)(mt_next_u32(m) & 0xffu);
    // PHP's RAND_RANGE_BADSCALING over a 31-bit word truncates to exactly the high byte.
    if (dist == MT_PHP) return (u8)(mt_next_u32(m) >> 24);
    for (;;) {
        u32 word = mt_next_u32(m);
        // Fires for 256 of ~4.3e9 words. Kept so the stream matches bit for bit.
        if (word < MT_PAST) return (u8)(word / MT_SCALING);
    }
}

// The 32-byte entropy block for one seed, `offset` bytes into its stream.
//
// The discarded prefix goes through `mt_next_byte` rather than being counted in words, so
// the rejection behaviour is part of the stream position -- exactly as it was for a real
// second `bx seed` call against an un-reseeded generator.
INLINE void mt_entropy(u32 seed, u32 offset, u32 dist, THREAD u8* out) {
    Mt19937 m;
    mt_seed(&m, seed, dist);
    for (u32 i = 0; i < offset; i++) mt_next_byte(&m, dist);
    for (u32 i = 0; i < 32; i++) out[i] = mt_next_byte(&m, dist);
}
