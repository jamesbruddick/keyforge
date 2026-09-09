// The MT19937 family: Milk Sad (`bx seed`), Trust Wallet Core, and PHP's mt_rand.
//
// Three products, two byte streams -- see `crate::vuln` for why they share a generator
// but not a scope. Which streams a scan walks is baked in as `VULN_STREAMS`, so `stream`
// here indexes that list rather than naming a distribution directly.
//
// The point is a 32-bit seed, so the high half is ignored: none of these generators can
// be seeded with more than `std::mt19937` takes.

INLINE void vuln_expand(u32 point_lo, u32 point_hi, u32 stream, THREAD u8* out) {
    (void)point_hi;
    const u32 streams[ARRAY_N(N_STREAMS)] = VULN_STREAMS;
    const u32 offsets[ARRAY_N(N_STREAMS)] = VULN_OFFSETS;
    // The offset prefix is *drawn*, not skipped: under libstdc++ the byte draw rejects,
    // so how far the generator has advanced depends on what came out, and there is no
    // position to jump to. This is what makes an offset mean the same thing under both
    // distributions -- see `entropy_for_seed_at` on the host.
    mt_entropy(point_lo, offsets[stream], streams[stream], out);
}
