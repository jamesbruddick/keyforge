// Private keys that are just small numbers: 1, 2, 3, and upwards.
//
// The smallest `vuln_expand` there is, and deliberately so: the vulnerability is a
// counter sitting where a key should be, so there is no generator to run and no byte
// mapping to guess. One stream, one key per point.
//
// The placement is the only thing here that can be wrong, and it is worth stating. The
// host writes the point big-endian into the *low* 16 bytes of the key, which is where a
// 128-bit counter lands in a 32-byte buffer. The device is handed only the low 64 bits of
// the point, so it fills the last eight bytes and zeroes the rest -- identical to the host
// for every point below 2^64, and the space this sweeps is 2^32 wide.

INLINE void vuln_expand(u32 point_lo, u32 point_hi, u32 stream, THREAD u8* out) {
    (void)stream;
    for (u32 i = 0; i < 24; i++) out[i] = 0;
    for (u32 i = 0; i < 4; i++) {
        out[24 + i] = (u8)(point_hi >> (24 - 8 * i));
        out[28 + i] = (u8)(point_lo >> (24 - 8 * i));
    }
}
