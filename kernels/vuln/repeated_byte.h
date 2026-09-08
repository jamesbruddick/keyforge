// Private keys made of one byte repeated 32 times.
//
// What a `memset` on the wrong buffer leaves behind. The space is 255 points wide, so
// this kernel exists for completeness rather than for speed -- a device spends longer
// opening than the CPU spends finishing the whole sweep. It is here because a plugin
// that can be ported cheaply and is not is a plugin that quietly refuses `--gpu` for no
// reason a user can see.

INLINE void vuln_expand(u32 point_lo, u32 point_hi, u32 stream, THREAD u8* out) {
    (void)point_hi;
    (void)stream;
    for (u32 i = 0; i < 32; i++) out[i] = (u8)point_lo;
}
