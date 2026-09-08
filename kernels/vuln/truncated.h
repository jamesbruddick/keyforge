// Entropy buffers that were only partly filled.
//
// Not a generator at all: the randomness may have been perfect, and only the first
// `VULN_RANDOM_PREFIX` bytes of it ever reached the buffer. The rest is what the
// allocator left, which is zero.
//
// The prefix width is a host constant rather than a literal here, because a user who
// suspects a narrower short read narrows `--end` and the two have to agree about where
// the bytes land. It arrives as a `#define` from `TruncatedEntropy::kernel`.
//
// Big-endian in the leading bytes, mirroring the host: a buffer fills front to back, so
// what arrived is at the start and the zeros are the tail. The point's low 32 bits are
// laid out big-endian and the prefix takes the *last* `VULN_RANDOM_PREFIX` of those four
// bytes, which is what `material[..PREFIX] = bytes[4 - PREFIX..]` does on the host.

INLINE void vuln_expand(u32 point_lo, u32 point_hi, u32 stream, THREAD u8* out) {
    (void)point_hi;
    (void)stream;
    for (u32 i = 0; i < 32; i++) out[i] = 0;
    for (u32 i = 0; i < VULN_RANDOM_PREFIX; i++) {
        u32 shift = 8u * (VULN_RANDOM_PREFIX - 1u - i);
        out[i] = (u8)(point_lo >> shift);
    }
}
