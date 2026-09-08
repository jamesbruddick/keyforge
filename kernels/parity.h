// Echo kernels for the differential tests in src/gpu/parity.rs.
//
// Each one does the least possible around a single primitive: read the case's inputs,
// apply the operation, write the result. That is what makes a failure point at one
// function rather than at "the GPU". They are appended to the translation unit only by
// the tests, so none of this compiles into a scanning binary.
//
// Every kernel bounds-checks `gid` against the case count. A dispatch rounds up to whole
// threadgroups, so the tail threads of any launch whose size is not a multiple of the
// group width would otherwise read and write past the end.

// Five results per case: a+b, a-b, a*b, a^2, 1/a. Inputs are pairs of big-endian
// field elements, already reduced below p by the host's rejection sampler.
KERNEL parity_field(
    BUF(const u8, in, 0),
    BUF(u8, out, 1),
    CBUF(u32, n, 2)
    GID_PARAM)
{
    GID_INIT
    if (gid >= n) return;

    u8 ab[32], bb[32];
    for (u32 i = 0; i < 32; i++) {
        ab[i] = in[gid * 64 + i];
        bb[i] = in[gid * 64 + 32 + i];
    }
    Fe a = fe_from_be(ab);
    Fe b = fe_from_be(bb);

    Fe r[5];
    r[0] = fe_add(a, b);
    r[1] = fe_sub(a, b);
    r[2] = fe_mul(a, b);
    r[3] = fe_sqr(a);
    r[4] = fe_inv(a);

    for (u32 k = 0; k < 5; k++) {
        u8 tmp[32];
        fe_to_be(r[k], tmp);
        for (u32 i = 0; i < 32; i++) out[(gid * 5 + k) * 32 + i] = tmp[i];
    }
}

// Messages as 120-byte records: one length byte, then up to 119 bytes -- which covers the
// 65-byte uncompressed public key, the longest thing this program hashes.
KERNEL parity_sha256(BUF(const u8, in, 0), BUF(u8, out, 1), CBUF(u32, n, 2) GID_PARAM) {
    GID_INIT
    if (gid >= n) return;
    u8 msg[119];
    u32 len = (u32)in[gid * 120];
    for (u32 i = 0; i < 119; i++) msg[i] = in[gid * 120 + 1 + i];
    u8 d[32];
    sha256_hash(msg, len, d);
    for (u32 i = 0; i < 32; i++) out[gid * 32 + i] = d[i];
}

// Same input shape; out: the 20-byte hash160. Covers RIPEMD-160 as well as the pairing.
KERNEL parity_hash160(BUF(const u8, in, 0), BUF(u8, out, 1), CBUF(u32, n, 2) GID_PARAM) {
    GID_INIT
    if (gid >= n) return;
    u8 msg[119];
    u32 len = (u32)in[gid * 120];
    for (u32 i = 0; i < 119; i++) msg[i] = in[gid * 120 + 1 + i];
    u8 d[20];
    hash160(msg, len, d);
    for (u32 i = 0; i < 20; i++) out[gid * 20 + i] = d[i];
}

// Messages as 256-byte records: two length bytes little-endian, then up to 254 bytes.
KERNEL parity_sha512(BUF(const u8, in, 0), BUF(u8, out, 1), CBUF(u32, n, 2) GID_PARAM) {
    GID_INIT
    if (gid >= n) return;
    u8 msg[254];
    u32 len = (u32)in[gid * 256] | ((u32)in[gid * 256 + 1] << 8);
    for (u32 i = 0; i < 254; i++) msg[i] = in[gid * 256 + 2 + i];
    u8 d[64];
    sha512_hash(msg, len, d);
    for (u32 i = 0; i < 64; i++) out[gid * 64 + i] = d[i];
}

// In: u32 seed, u32 offset, u32 dist, little-endian. Out: the 32-byte entropy block.
// Twelve bytes per case rather than eight: `dist` is half of what names a wallet's
// entropy stream, so both distributions are exercised by the same test.
KERNEL parity_mt(BUF(const u8, in, 0), BUF(u8, out, 1), CBUF(u32, n, 2) GID_PARAM) {
    GID_INIT
    if (gid >= n) return;
    u32 seed = 0, offset = 0, dist = 0;
    for (u32 i = 0; i < 4; i++) {
        seed   |= (u32)in[gid * 12 + i] << (8 * i);
        offset |= (u32)in[gid * 12 + 4 + i] << (8 * i);
        dist   |= (u32)in[gid * 12 + 8 + i] << (8 * i);
    }
    u8 e[32];
    mt_entropy(seed, offset, dist, e);
    for (u32 i = 0; i < 32; i++) out[gid * 32 + i] = e[i];
}

// In: one length byte then 32 entropy bytes, padded to 33. Out: two length bytes then the
// phrase, padded to 224.
KERNEL parity_bip39(BUF(const u8, in, 0), BUF(u8, out, 1), CBUF(u32, n, 2),
                    BUF(const u8, wordlist, 3) GID_PARAM) {
    GID_INIT
    if (gid >= n) return;
    u8 entropy[32];
    u32 len = (u32)in[gid * 33];
    for (u32 i = 0; i < 32; i++) entropy[i] = in[gid * 33 + 1 + i];

    u8 phrase[BIP39_MAX_PHRASE];
    u32 plen = bip39_phrase(entropy, len, wordlist, phrase);

    out[gid * 224] = (u8)plen;
    out[gid * 224 + 1] = (u8)(plen >> 8);
    for (u32 i = 0; i < 222; i++) out[gid * 224 + 2 + i] = (i < plen) ? phrase[i] : 0;
}

// In: two length bytes then the phrase, padded to 224. Out: the 64-byte BIP39 seed.
// Deliberately the full 2048 iterations -- a shortened count would not exercise the
// midstate reuse, which is exactly where this could be wrong.
KERNEL parity_pbkdf2(BUF(const u8, in, 0), BUF(u8, out, 1), CBUF(u32, n, 2) GID_PARAM) {
    GID_INIT
    if (gid >= n) return;
    u8 phrase[222];
    u32 len = (u32)in[gid * 224] | ((u32)in[gid * 224 + 1] << 8);
    for (u32 i = 0; i < 222; i++) phrase[i] = in[gid * 224 + 2 + i];
    u8 seed[64];
    bip39_seed(phrase, len, 2048u, seed);
    for (u32 i = 0; i < 64; i++) out[gid * 64 + i] = seed[i];
}

// In: a 32-byte scalar. Out: the 33-byte compressed key, then the 65-byte uncompressed
// one, padded to 104. Covers the digit recoding, the comb lookup, every branch of the
// Jacobian addition, the inversion, and both serialisations -- the whole EC layer end to
// end against `ec::public_key`.
//
// The inversion is per thread here, which is *not* how the pipeline does it (see the
// `invert_*` kernels). That is the point: this test isolates the multiply, and the batched
// inversion is checked separately against it.
KERNEL parity_ec(BUF(const u8, in, 0), BUF(u8, out, 1), CBUF(u32, n, 2),
                 BUF(const Ge, comb, 3) GID_PARAM) {
    GID_INIT
    if (gid >= n) return;
    u8 scalar[32];
    for (u32 i = 0; i < 32; i++) scalar[i] = in[gid * 32 + i];

    Gej j = ec_mul_gen(sc_from_be(scalar), comb);
    Ge p = gej_to_ge(j, fe_inv(j.z));

    u8 c[33], u[65];
    ge_serialize(p, c);
    ge_serialize_uncompressed(p, u);
    for (u32 i = 0; i < 33; i++) out[gid * 104 + i] = c[i];
    for (u32 i = 0; i < 65; i++) out[gid * 104 + 33 + i] = u[i];
}

// The batched inversion, checked against the per-thread one.
//
// In: `n` field elements. Out: `n` inverses. Three passes in one kernel launch is not how
// the pipeline runs it, but the arithmetic being verified is the same: a running product,
// one real inversion, then a walk back peeling factors off.
KERNEL parity_batch_invert(BUF(const u8, in, 0), BUF(u8, out, 1), CBUF(u32, n, 2)
                           GID_PARAM) {
    GID_INIT
    if (gid != 0) return; // one thread does the whole scan; this is a correctness check

    // Montgomery's trick. Bounded to keep the test's scratch small; the pipeline's
    // version is tiled across threadgroups instead.
    Fe running = fe_one();
    for (u32 i = 0; i < n; i++) {
        u8 b[32];
        for (u32 k = 0; k < 32; k++) b[k] = in[i * 32 + k];
        Fe v = fe_from_be(b);
        // Stash the prefix product where the result will go, then overwrite it below.
        u8 tmp[32];
        fe_to_be(running, tmp);
        for (u32 k = 0; k < 32; k++) out[i * 32 + k] = tmp[k];
        running = fe_mul(running, v);
    }

    Fe inv = fe_inv(running);
    for (int i = (int)n - 1; i >= 0; i--) {
        u8 b[32];
        for (u32 k = 0; k < 32; k++) b[k] = in[i * 32 + k];
        Fe v = fe_from_be(b);
        for (u32 k = 0; k < 32; k++) b[k] = out[i * 32 + k];
        Fe prefix = fe_from_be(b);

        Fe result = fe_mul(prefix, inv);
        inv = fe_mul(inv, v);
        u8 tmp[32];
        fe_to_be(result, tmp);
        for (u32 k = 0; k < 32; k++) out[i * 32 + k] = tmp[k];
    }
}

// In: a 20-byte hash160. Out: the twenty 64-bit scattered probe indices, in schedule
// order, then the twenty blocked ones for a filter of PARITY_BLOOM_BITS bits. Compared
// against the CPU's own arithmetic rather than against a filter, so a drift in either
// schedule is caught as a schedule difference rather than as a missed match.
//
// Both schedules in one kernel because a sweep runs both: the primary filter is blocked
// and the verification filter beside it is scattered, and a drift in either loses every
// find with nothing to show for it.
#define PARITY_BLOOM_BITS (U64C(1000003) * (u64)BLOOM_BLOCK_BITS)

KERNEL parity_bloom(BUF(const u8, in, 0), BUF(u8, out, 1), CBUF(u32, n, 2) GID_PARAM) {
    GID_INIT
    if (gid >= n) return;
    u8 h[20];
    for (u32 i = 0; i < 20; i++) h[i] = in[gid * 20 + i];

    u32 at = 0;
    u64 a[BLOOM_FOLDED];
    bloom_fold(h, a);
    for (u32 s = 0; s < BLOOM_SHIFTS; s++) {
        for (u32 i = 0; i < BLOOM_FOLDED; i++) {
            u64 v = bloom_index(a, BLOOM_SHIFT[s], i);
            for (u32 b = 0; b < 8; b++) out[(gid * 40 + at) * 8 + b] = (u8)(v >> (8 * b));
            at++;
        }
    }

    u32 positions[BLOOM_PROBES];
    u64 origin = bloom_block_probes(h, PARITY_BLOOM_BITS / (u64)BLOOM_BLOCK_BITS, positions);
    for (u32 i = 0; i < BLOOM_PROBES; i++) {
        u64 v = origin + (u64)positions[i];
        for (u32 b = 0; b < 8; b++) out[(gid * 40 + at) * 8 + b] = (u8)(v >> (8 * b));
        at++;
    }
}

// A tight chain of field multiplies, for measuring this device's ceiling.
//
// The round count arrives in the input buffer rather than as a scalar, so these match the
// `(in, out, n, [extra])` shape every other kernel here uses. They did not, once: the
// harness bound the *output buffer* to the `rounds` parameter and the loop ran an
// arbitrary number of times, which is how this reported 74,765 M field multiplies a second
// on a machine that cannot do a hundredth of that. A benchmark that cannot be wrong is
// worth more than a benchmark that is usually right.
//
// Not a correctness test: it exists to answer "is the pipeline near what fe_mul can do, or
// is it losing time somewhere else?". Each thread runs a dependent chain, which is the
// same shape the curve accumulator has, so the number is directly comparable.
KERNEL bench_fe_mul(BUF(const u8, in, 0), BUF(u8, out, 1), CBUF(u32, n, 2) GID_PARAM) {
    GID_INIT
    if (gid >= n) return;
    u32 rounds = (u32)in[0] | ((u32)in[1] << 8) | ((u32)in[2] << 16) | ((u32)in[3] << 24);
    Fe a = fe_one();
    a.n[0] = gid | 1u;
    a.n[3] = gid * 2654435761u;
    Fe b = a;
    b.n[1] = gid ^ 0x5bf03635u;
    for (u32 i = 0; i < rounds; i++) a = fe_mul(a, b);
    u8 tmp[32];
    fe_to_be(a, tmp);
    for (u32 i = 0; i < 32; i++) out[gid * 32 + i] = tmp[i];
}

// Point addition with a fixed addend: pure arithmetic and register pressure, no gather.
KERNEL bench_gej_add(BUF(const u8, in, 0), BUF(u8, out, 1), CBUF(u32, n, 2),
                     BUF(const Ge, comb, 3) GID_PARAM) {
    GID_INIT
    if (gid >= n) return;
    u32 rounds = (u32)in[0] | ((u32)in[1] << 8) | ((u32)in[2] << 16) | ((u32)in[3] << 24);
    Ge q = comb[gid & 4095u];
    Gej acc = gej_from_ge(comb[(gid + 7u) & 4095u]);
    for (u32 i = 0; i < rounds; i++) acc = gej_add_ge(acc, q);
    u8 tmp[32];
    fe_to_be(acc.x, tmp);
    for (u32 i = 0; i < 32; i++) out[gid * 32 + i] = tmp[i];
}
