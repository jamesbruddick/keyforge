// Probing a `keyscan bf-gen` bloom filter from the device.
//
// A mirror of `bloom::BloomFilter::contains`, in both of the layouts a filter can be laid
// out in: the scattered schedule inherited from ecloop's `blf_has`, and the blocked one
// `keyscan` writes its primary filter in now. Any drift turns every lookup into a silent
// miss, so both schedules are held to the CPU's by a differential test over random hashes.
//
// Which one a lookup runs is a kernel argument rather than a compile-time constant, and
// deliberately so. `keyscan` bakes its filter's layout into the kernel because its scan is
// filter-bound at billions of probes a second; this one is nowhere near that -- see the
// measurement below -- and a uniform branch per hash costs nothing beside the SHA-256 and
// RIPEMD-160 that produced the hash. What it buys is that the kernels are compiled before
// the filter is read, which is the order that puts a compilation failure in the first
// second of a run rather than after a multi-gigabyte read.
//
// `contains_batch`'s two-pass structure is deliberately *not* ported. It exists to overlap
// DRAM misses inside one core's out-of-order window; a GPU has thousands of memory
// requests in flight by construction, so one thread does the plain early-exiting loop and
// the hardware does the overlapping. Measured at 266 M random probes/s over a 7.6 GB
// no-copy buffer on an M1 Pro -- about 115,000 seeds/s of filter capacity, which is why
// this is not the bottleneck.

#define BLOOM_FOLDED 5
#define BLOOM_SHIFTS 4
#define BLOOM_PROBES 20

// Bits in one block of a blocked filter, and the nine-bit positions that fit in a mixed
// 64-bit word. Both mirror `BLOCK_BITS` and `POSITIONS_PER_WORD` in src/bloom.rs; the host
// and the device have to agree on every bit either of them names.
#define BLOOM_BLOCK_BITS 512u
#define BLOOM_POSITIONS_PER_WORD 7

CONSTANT u32 BLOOM_SHIFT[BLOOM_SHIFTS] = { 24u, 28u, 36u, 40u };

// The five big-endian 32-bit words of a hash160, which both schedules are written over.
INLINE void bloom_words(THREAD const u8* h, THREAD u64* w) {
    for (u32 i = 0; i < BLOOM_FOLDED; i++) {
        w[i] = ((u64)h[i * 4] << 24) | ((u64)h[i * 4 + 1] << 16)
             | ((u64)h[i * 4 + 2] << 8) | (u64)h[i * 4 + 3];
    }
}

// The five overlapping 64-bit words ecloop folds a hash160 into: the words above, paired
// cyclically.
INLINE void bloom_fold(THREAD const u8* h, THREAD u64* a) {
    u64 w[BLOOM_FOLDED];
    bloom_words(h, w);
    a[0] = (w[0] << 32) | w[1];
    a[1] = (w[2] << 32) | w[3];
    a[2] = (w[4] << 32) | w[0];
    a[3] = (w[1] << 32) | w[2];
    a[4] = (w[3] << 32) | w[4];
}

INLINE u64 bloom_index(THREAD const u64* a, u32 shift, u32 i) {
    return (a[i] << shift) | (a[(i + 1) % BLOOM_FOLDED] >> shift);
}

INLINE bool bloom_probe(DEVICE const u64* words, u64 bits, u64 index) {
    u64 at = index % bits;
    return (words[at / 64] & (U64C(1) << (at % 64))) != U64C(0);
}

// A bit the blocked schedule has already placed inside the array: the block exists and the
// position is inside it, so there is nothing left to reduce.
INLINE bool bloom_probe_at(DEVICE const u64* words, u64 index) {
    return (words[index / 64] & (U64C(1) << (index % 64))) != U64C(0);
}

// The SplitMix64 finalizer, mirroring `mix64` in src/bloom.rs.
INLINE u64 bloom_mix64(u64 z) {
    z = (z ^ (z >> 30)) * U64C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * U64C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

// Where a hash probes in a blocked filter: the first bit of its block, and the
// `BLOOM_PROBES` positions inside it.
//
// Four independent mixes of the tag, not an arithmetic progression over one: see
// `block_probes` in src/bloom.rs for what the progression form measured. Every constant
// here is part of the file `keyscan bf-gen` wrote and none of them is this side's to
// choose.
INLINE u64 bloom_block_probes(THREAD const u8* h, u64 blocks, THREAD u32* positions) {
    u64 w[BLOOM_FOLDED];
    bloom_words(h, w);

    // All five words into one value first, so every bit of the tag reaches every mix.
    u64 fold = ((w[0] << 32) | w[1])
             ^ (((w[2] << 32) | w[3]) * U64C(0x9e3779b97f4a7c15))
             ^ (w[4] * U64C(0xff51afd7ed558ccd));

    u64 m[3];
    m[0] = bloom_mix64(fold ^ U64C(0xa0761d6478bd642f));
    m[1] = bloom_mix64(fold ^ U64C(0xe7037ed1a0b428db));
    m[2] = bloom_mix64(fold ^ U64C(0x8ebc6af09c88c6e3));

    // A multiply-and-shift, not a remainder: a GPU has no 64-bit divide instruction at all.
    u64 block = ((bloom_mix64(fold) >> 32) * blocks) >> 32;

    for (u32 i = 0; i < BLOOM_PROBES; i++) {
        positions[i] = ((u32)(m[i / BLOOM_POSITIONS_PER_WORD]
                              >> (9u * (i % BLOOM_POSITIONS_PER_WORD))))
                     & (BLOOM_BLOCK_BITS - 1u);
    }
    return block * (u64)BLOOM_BLOCK_BITS;
}

// Is this hash160 possibly in the set, in a scattered filter?
//
// Early exit on the first clear bit, which is what keeps the average near 1.6 probes of
// the 20.
INLINE bool bloom_contains_scattered(DEVICE const u64* words, u64 bits, THREAD const u8* h) {
    u64 a[BLOOM_FOLDED];
    bloom_fold(h, a);
    for (u32 s = 0; s < BLOOM_SHIFTS; s++) {
        for (u32 i = 0; i < BLOOM_FOLDED; i++) {
            if (!bloom_probe(words, bits, bloom_index(a, BLOOM_SHIFT[s], i))) return false;
        }
    }
    return true;
}

// The same, in a blocked filter: every probe inside one 512-bit block, so a lookup reads
// one cache line however many probes it makes.
INLINE bool bloom_contains_blocked(DEVICE const u64* words, u64 bits, THREAD const u8* h) {
    u32 positions[BLOOM_PROBES];
    u64 origin = bloom_block_probes(h, bits / (u64)BLOOM_BLOCK_BITS, positions);
    for (u32 i = 0; i < BLOOM_PROBES; i++) {
        if (!bloom_probe_at(words, origin + (u64)positions[i])) return false;
    }
    return true;
}

// Is this hash160 possibly in the set?
//
// False positives at the rate the filter was built for; false negatives never. `blocked`
// is the layout the host read out of the filter's header -- one value for the whole
// launch, so the branch is uniform across every thread in it.
INLINE bool bloom_contains(DEVICE const u64* words, u64 bits, u32 blocked, THREAD const u8* h) {
    // A filter that has not been bound yet has no bits and matches nothing, which is what
    // the smoke dispatch wants. Checked rather than assumed: without it a scattered probe
    // divides by zero and a blocked one reads a block past the end of the placeholder
    // buffer, neither of which a device reports.
    if (bits == U64C(0)) return false;
    return blocked ? bloom_contains_blocked(words, bits, h)
                   : bloom_contains_scattered(words, bits, h);
}
