// glibc's `random()` / `rand()`: a TYPE_3 additive-feedback generator.
//
// Not a Mersenne twister and not an LCG. A 31-word state is seeded by a Lehmer LCG, the
// generator then runs `r[i] = r[i-3] + r[i-31]` mod 2^32 and returns the top 31 bits, and
// the first 310 outputs are thrown away.
//
// Three ways to get this nearly right and entirely wrong, all mirrored from src/vuln/glibc.rs:
//
//   1. **The warm-up is not optional.** Without those 310 discards the early outputs still
//      carry the seeding LCG's very regular structure, and every derived wallet is wrong.
//   2. **Seeding is signed arithmetic.** glibc computes `16807 * r % 2147483647` by
//      Schrage's trick over `int32_t`, and the state word is read back as a *signed* int.
//      A seed with its top bit set therefore starts a negative chain, and C's division
//      truncating toward zero is part of the answer. Doing this in unsigned or in wider
//      arithmetic agrees for small seeds and diverges for the top half of the space --
//      which is the kind of bug that shows up as a clean sweep over half the range.
//   3. **Seed 0 is seed 1.** glibc remaps it, because an all-zero state stays all-zero
//      forever. The host's space starts at 1 so no scan walks it, but a `--start 0` would
//      reach here and must not produce a dead generator.
//
// The byte kept is the low one, which is what `random() & 0xff` and `random() % 256` both
// compile to. That is an assumption about the affected program, not a property of the
// generator, and the guide says so.

#define GLIBC_DEG    31u
#define GLIBC_SEP    3u
#define GLIBC_WARMUP 310u

INLINE u32 glibc_next_u31(THREAD u32* state, THREAD u32* f, THREAD u32* r) {
    state[*f] = state[*f] + state[*r];
    u32 result = state[*f] >> 1;
    *f = (*f + 1u) % GLIBC_DEG;
    *r = (*r + 1u) % GLIBC_DEG;
    return result;
}

INLINE void vuln_expand(u32 point_lo, u32 point_hi, u32 stream, THREAD u8* out) {
    (void)point_hi;
    const u32 offsets[ARRAY_N(N_STREAMS)] = VULN_OFFSETS;
    const u32 offset = offsets[stream];
    u32 seed = (point_lo == 0u) ? 1u : point_lo;

    u32 state[GLIBC_DEG];
    state[0] = seed;
    for (u32 i = 1; i < GLIBC_DEG; i++) {
        // Schrage, verbatim from glibc: the intermediate product never leaves 32 bits,
        // and the correction is applied to a signed value. See note 2 above.
        i64 prev = (i64)(i32)state[i - 1];
        i64 hi = prev / 127773;
        i64 lo = prev % 127773;
        i64 word = 16807 * lo - 2836 * hi;
        if (word < 0) word += 2147483647;
        state[i] = (u32)word;
    }

    u32 f = GLIBC_SEP, r = 0u;
    for (u32 i = 0; i < GLIBC_WARMUP; i++) glibc_next_u31(state, &f, &r);
    // An earlier wallet's bytes, drawn and thrown away: `random()` is a recurrence over
    // its own state, so there is no position to jump to.
    for (u32 i = 0; i < offset; i++) glibc_next_u31(state, &f, &r);
    for (u32 i = 0; i < 32; i++) out[i] = (u8)glibc_next_u31(state, &f, &r);
}
