// `java.util.Random`, a 48-bit linear congruential generator.
//
// The two details that decide whether this reproduces a real Java stream:
//
//   1. The constructor **scrambles** its argument -- `(seed ^ 0x5DEECE66D) & mask` --
//      and passing the raw value into the state is the most common way to get this
//      class wrong. It yields a plausible stream that matches no wallet.
//   2. `nextBytes` fills four bytes per `nextInt()` **low byte first**, and a short tail
//      takes only what it needs. Reversed, the stream looks just as random and is
//      entirely wrong. A 32-byte buffer is eight whole words, so the tail case cannot
//      arise here; the host covers it and this mirrors the whole-word path.
//
// The seed space is 2^48, so unlike the 32-bit generators this genuinely uses both
// halves of the point -- `java-random` is scanned over a narrowed millisecond window,
// and a window late in 2015 already exceeds 2^32.

#define JAVA_MULTIPLIER U64C(0x5DEECE66D)
#define JAVA_ADDEND     U64C(0xB)
#define JAVA_MASK       ((U64C(1) << 48) - U64C(1))

INLINE u32 java_next(THREAD u64* state, u32 bits) {
    *state = (*state * JAVA_MULTIPLIER + JAVA_ADDEND) & JAVA_MASK;
    return (u32)(*state >> (48u - bits));
}

INLINE void vuln_expand(u32 point_lo, u32 point_hi, u32 stream, THREAD u8* out) {
    (void)stream;
    u64 seed = ((u64)point_hi << 32) | (u64)point_lo;
    u64 state = (seed ^ JAVA_MULTIPLIER) & JAVA_MASK;
    for (u32 i = 0; i < 32; i += 4) {
        u32 rnd = java_next(&state, 32u);
        for (u32 k = 0; k < 4; k++) {
            out[i + k] = (u8)rnd;
            rnd >>= 8;
        }
    }
}
