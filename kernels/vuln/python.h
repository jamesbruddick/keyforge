// Wallets built on CPython's `random` module.
//
// The generator is MT19937, but seeded through `init_by_array` rather than
// `init_genrand` -- see `mt_seed_by_array` in kernels/mt19937.h for why that distinction
// is the whole plugin. `random.seed(500)` and `std::mt19937 twister(500)` share a name
// and nothing else.
//
// The byte mapping is successive `getrandbits(32)` words in **little-endian** order,
// which is what `random.getrandbits(256).to_bytes(32, 'little')` and the common
// `bytes([random.getrandbits(8) ...])` idioms both reduce to. Note that this is a raw
// word, not a `uniform_int_distribution` byte: `mt_next_byte`'s rejection sampling has
// no part here, and using it would silently scan a different stream.

INLINE void vuln_expand(u32 point_lo, u32 point_hi, u32 stream, THREAD u8* out) {
    (void)point_hi;
    (void)stream;
    Mt19937 m;
    mt_seed_by_array(&m, point_lo);
    for (u32 i = 0; i < 32; i += 4) {
        u32 word = mt_next_u32(&m);
        for (u32 k = 0; k < 4; k++) {
            out[i + k] = (u8)word;
            word >>= 8;
        }
    }
}
