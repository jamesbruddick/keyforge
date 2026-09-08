// The kernel entry points.
//
// The chain mirrors `derive::walk_levels` level for level, which is what lets the
// differential test compare the two bit for bit. Where the CPU builds each level as a
// `Vec` that compacts -- an invalid node is simply not pushed -- the GPU keeps every level
// dense and marks dead slots with `valid = 0`. A thread's index is its position, so a
// level has to have a fixed shape before the launch; dense-and-marked is the only form
// that gives one. See src/gpu/layout.rs.
//
// One launch is:
//
//     entropy -> [bip39_pbkdf2] -> master -> prefix
//       -> kmul/invert/ckd_normal   (chain level)
//       -> kmul/invert/ckd_normal   (index level)
//       -> kmul/invert/leaf         (hashes, probes, hits)
//
// The three `kmul -> invert` pairs are where all the curve work is. See ec.h for why the
// inversion is batched across a level rather than shared inside the multiply.

// Everything crossing a buffer boundary is stored as 32-bit words, in the limb order the
// arithmetic already uses.
//
// The alternative -- bytes, big-endian, converted on every load and store -- makes a `Gej`
// 100 individual byte accesses each way plus two endian conversions. This form is kept
// because it is plainly less work, not because it was worth much: measured at 1,425
// seeds/s before and 1,368 after, i.e. **no change outside the noise.** The buffers are
// simply not where the time goes. Recorded so the next person does not re-derive the same
// hypothesis and re-run the same experiment.
//
// Big-endian bytes still appear at the edges, where a format demands them -- a serialised
// public key, an HMAC input, a hash160 -- and nowhere else.

// A BIP32 extended private key: the scalar, the chain code, and whether this slot holds a
// key at all. 20 words rather than 17 so the stride is a multiple of four words.
#define NODE_WORDS 20

typedef struct { Scalar key; u8 chain[32]; bool valid; } Node;

// The chain code is bytes to its consumer (it keys an HMAC) but storage is storage: any
// consistent packing does, and little-endian words are what the hardware wants.
INLINE void bytes_from_words(THREAD const u32* w, THREAD u8* b) {
    for (u32 i = 0; i < 8; i++) {
        b[i * 4]     = (u8)(w[i]);
        b[i * 4 + 1] = (u8)(w[i] >> 8);
        b[i * 4 + 2] = (u8)(w[i] >> 16);
        b[i * 4 + 3] = (u8)(w[i] >> 24);
    }
}

INLINE void words_from_bytes(THREAD const u8* b, THREAD u32* w) {
    for (u32 i = 0; i < 8; i++) {
        w[i] = (u32)b[i * 4] | ((u32)b[i * 4 + 1] << 8)
             | ((u32)b[i * 4 + 2] << 16) | ((u32)b[i * 4 + 3] << 24);
    }
}

INLINE Node node_load(DEVICE const u32* buf, u32 i) {
    DEVICE const u32* at = buf + i * NODE_WORDS;
    Node n;
    u32 tmp[8];
    for (u32 k = 0; k < 8; k++) n.key.n[k] = at[k];
    for (u32 k = 0; k < 8; k++) tmp[k] = at[8 + k];
    bytes_from_words(tmp, n.chain);
    n.valid = at[16] != 0;
    return n;
}

INLINE void node_store(DEVICE u32* buf, u32 i, Node n) {
    DEVICE u32* at = buf + i * NODE_WORDS;
    u32 tmp[8];
    for (u32 k = 0; k < 8; k++) at[k] = n.key.n[k];
    words_from_bytes(n.chain, tmp);
    for (u32 k = 0; k < 8; k++) at[8 + k] = tmp[k];
    at[16] = n.valid ? 1u : 0u;
}

INLINE Node node_dead() {
    Node n;
    for (u32 i = 0; i < 8; i++) n.key.n[i] = 0;
    for (u32 i = 0; i < 32; i++) n.chain[i] = 0;
    n.valid = false;
    return n;
}

// Split an HMAC-SHA512 output into a node: IL is the key, IR the chain code. Invalid when
// IL is not a valid scalar, which BIP32 says to skip.
INLINE Node node_split(THREAD const u8* i64) {
    Node n;
    n.key = sc_from_be(i64);
    for (u32 k = 0; k < 32; k++) n.chain[k] = i64[32 + k];
    n.valid = sc_is_valid(n.key);
    return n;
}

// The BIP32 master node for some seed material.
INLINE Node bip32_master(THREAD const u8* material, u32 len) {
    u8 key[12];
    key[0] = 'B'; key[1] = 'i'; key[2] = 't'; key[3] = 'c'; key[4] = 'o'; key[5] = 'i';
    key[6] = 'n'; key[7] = ' '; key[8] = 's'; key[9] = 'e'; key[10] = 'e'; key[11] = 'd';
    u8 i64[64];
    hmac512_mac(hmac512_new(key, 12), material, len, i64);
    return node_split(i64);
}

// CKDpriv, given the 37-byte data block the two variants build differently and the
// parent's chain-code HMAC already keyed.
//
// **The key is a parameter rather than built here**, and that is what makes
// `k_ckd_normal` worth its shape. `hmac512_new` absorbs two 128-byte pad blocks, so it
// costs two SHA-512 compressions against the two `hmac512_mac` then spends -- and the key
// is the *parent's* chain code, identical for every child of that parent. Built per child,
// half the compressions of the whole level are the same two, recomputed ten times over.
INLINE Node bip32_ckd_keyed(Scalar parent_key, Hmac512 key, THREAD const u8* data) {
    u8 i64[64];
    hmac512_mac(key, data, 37, i64);
    Node child = node_split(i64);
    if (!child.valid) return node_dead();
    // child = parse256(IL) + k_par (mod n). A zero result is invalid per BIP32; both that
    // and "IL >= n" above are ~2^-127 events, and skipping is the right answer for a
    // scanner.
    child.key = sc_add(child.key, parent_key);
    if (sc_is_zero(child.key)) return node_dead();
    child.valid = true;
    return child;
}

// The same, for a caller with one child to derive and no reason to hoist anything.
INLINE Node bip32_ckd(Node parent, THREAD const u8* data) {
    return bip32_ckd_keyed(parent.key, hmac512_new(parent.chain, 32), data);
}

// A hardened child. Needs no public key, so it costs no EC work.
INLINE Node bip32_hardened(Node parent, u32 index) {
    if (!parent.valid) return node_dead();
    u8 data[37];
    data[0] = 0;
    sc_to_be(parent.key, data + 1);
    u32 h = index | 0x80000000u;
    for (u32 i = 0; i < 4; i++) data[33 + i] = (u8)(h >> (24 - 8 * i));
    return bip32_ckd(parent, data);
}

// ------------------------------------------------------------------ K0: entropy

KERNEL k_entropy(BUF(u8, entropy, 0), CBUF(u32, base_seed, 1), CBUF(u32, offset, 2),
                 CBUF(u32, dist, 3), CBUF(u32, seeds, 4) GID_PARAM) {
    GID_INIT
    if (gid >= seeds) return;
    u8 e[32];
    mt_entropy(base_seed + gid, offset, dist, e);
    for (u32 i = 0; i < 32; i++) entropy[gid * 32 + i] = e[i];
}

// ------------------------------------------------------------------ K1: PBKDF2
//
// One thread per (seed, entropy size). The largest single share of the sweep: 46% of it on
// an M1 Pro at the default scope, against `k_kmul`'s 40%.

#if HAS_BIP39
KERNEL k_pbkdf2(BUF(const u8, entropy, 0), BUF(u8, seeds_out, 1),
                BUF(const u8, wordlist, 2), CBUF(u32, count, 3) GID_PARAM) {
    GID_INIT
    if (gid >= count) return;
    const u32 sizes[ARRAY_N(N_ENTROPY_SIZES)] = ENTROPY_SIZES;
    u32 seed = gid / N_ENTROPY_SIZES;
    u32 which = gid % N_ENTROPY_SIZES;

    // The phrase is scoped so it is dead before the stretch begins. It is 216 bytes and
    // is only needed to build the two midstates; leaving it live across 2,048 iterations
    // is what was costing this kernel more than half its throughput. See sha512.h.
    Hmac512 key;
    {
        u8 e[32];
        for (u32 i = 0; i < 32; i++) e[i] = entropy[seed * 32 + i];
        u8 phrase[BIP39_MAX_PHRASE];
        u32 plen = bip39_phrase(e, sizes[which], wordlist, phrase);
        key = hmac512_new(phrase, plen);
    }

    u8 out[64];
    pbkdf2_from_hmac(key, 2048u, out);
    for (u32 i = 0; i < 64; i++) seeds_out[gid * 64 + i] = out[i];
}
#endif

// ------------------------------------------------------------------ K2: master nodes
//
// One thread per (seed, entropy size, HD variant), in exactly that nesting -- the order
// `walk_seed_batch` pushes them and therefore the order `Layout::decode` unwinds.

KERNEL k_master(BUF(const u8, entropy, 0), BUF(const u8, bip39_seeds, 1),
                BUF(u32, nodes, 2), CBUF(u32, count, 3) GID_PARAM) {
    GID_INIT
    if (gid >= count) return;
    const u32 sizes[ARRAY_N(N_ENTROPY_SIZES)] = ENTROPY_SIZES;
    const u32 variants[ARRAY_N(N_HD_VARIANTS)] = HD_VARIANTS;

    u32 variant = gid % N_HD_VARIANTS;
    u32 rest = gid / N_HD_VARIANTS;
    u32 which = rest % N_ENTROPY_SIZES;
    u32 seed = rest / N_ENTROPY_SIZES;

    Node n;
    if (variants[variant] == VARIANT_BIP39) {
#if HAS_BIP39
        u8 material[64];
        u32 at = seed * N_ENTROPY_SIZES + which;
        for (u32 i = 0; i < 64; i++) material[i] = bip39_seeds[at * 64 + i];
        n = bip32_master(material, 64);
#else
        n = node_dead();
#endif
    } else {
        u8 material[32];
        for (u32 i = 0; i < 32; i++) material[i] = entropy[seed * 32 + i];
        n = bip32_master(material, sizes[which]);
    }
    node_store(nodes, gid, n);
}

// ------------------------------------------------------------------ K3: the hardened prefix
//
// m/purpose'/0'/account' -- three hardened steps and no EC work. `Bare` takes none of
// them: it is the master node itself, which the chain and index levels then expand into
// m/chain/index.

KERNEL k_prefix(BUF(const u32, masters, 0), BUF(u32, out, 1), CBUF(u32, count, 2) GID_PARAM) {
    GID_INIT
    if (gid >= count) return;
    const u32 purposes[ARRAY_N(N_PURPOSES)] = PURPOSES;

    u32 purpose = gid % N_PURPOSES;
    u32 parent = gid / N_PURPOSES;

    Node n = node_load(masters, parent);
    if (purposes[purpose] != PURPOSE_BARE) {
        n = bip32_hardened(n, purposes[purpose]);
        n = bip32_hardened(n, 0);
        n = bip32_hardened(n, ACCOUNT);
    }
    node_store(out, gid, n);
}

// --------------------------------------------------------------- K3b: raw private keys
//
// `bx seed | bx ec-new`: the raw 32 bytes are the private key, with no derivation at all.
// These join the leaf level rather than the master level, so their public key rides the
// same batched inversion as every HD leaf -- which is why they are written into the tail
// of the leaf node array rather than into a level of their own.

#if N_RAW_SIZES > 0
KERNEL k_raw(BUF(const u8, entropy, 0), BUF(u32, leaf_nodes, 1), CBUF(u32, count, 2),
             CBUF(u32, base, 3) GID_PARAM) {
    GID_INIT
    if (gid >= count) return;
    u32 seed = gid / N_RAW_SIZES;

    u8 e[32];
    for (u32 i = 0; i < 32; i++) e[i] = entropy[seed * 32 + i];

    Node n;
    n.key = sc_from_be(e);
    for (u32 i = 0; i < 32; i++) n.chain[i] = 0;
    n.valid = sc_is_valid(n.key);
    if (!n.valid) n = node_dead();
    node_store(leaf_nodes, base + gid, n);
}
#endif

// ------------------------------------------------------------------ K4: k*G per node

// 28 words rather than 25 so the stride is a multiple of four.
#define GEJ_WORDS 28

INLINE void gej_store(DEVICE u32* buf, u32 i, Gej p) {
    DEVICE u32* at = buf + i * GEJ_WORDS;
    for (u32 k = 0; k < 8; k++) at[k] = p.x.n[k];
    for (u32 k = 0; k < 8; k++) at[8 + k] = p.y.n[k];
    for (u32 k = 0; k < 8; k++) at[16 + k] = p.z.n[k];
    at[24] = p.infinity ? 1u : 0u;
}

INLINE Gej gej_load(DEVICE const u32* buf, u32 i) {
    DEVICE const u32* at = buf + i * GEJ_WORDS;
    Gej p;
    for (u32 k = 0; k < 8; k++) p.x.n[k] = at[k];
    for (u32 k = 0; k < 8; k++) p.y.n[k] = at[8 + k];
    for (u32 k = 0; k < 8; k++) p.z.n[k] = at[16 + k];
    p.infinity = at[24] != 0;
    return p;
}

// Field elements in the inversion's scratch buffers: eight words, no conversion.
INLINE Fe fe_load(DEVICE const u32* buf, u32 i) {
    Fe r;
    for (u32 k = 0; k < 8; k++) r.n[k] = buf[i * 8 + k];
    return r;
}

INLINE void fe_store(DEVICE u32* buf, u32 i, Fe v) {
    for (u32 k = 0; k < 8; k++) buf[i * 8 + k] = v.n[k];
}

KERNEL k_kmul(BUF(const u32, nodes, 0), BUF(const Ge, comb, 1), BUF(u32, gej, 2),
              CBUF(u32, count, 3) GID_PARAM) {
    GID_INIT
    if (gid >= count) return;
    Node n = node_load(nodes, gid);
    Gej p;
    if (n.valid) {
        p = ec_mul_gen(n.key, comb);
    } else {
        // A dead slot still needs a z the inversion can multiply through, so it gets
        // z = 1 -- the same placeholder `ec::to_affine` uses, for the same reason.
        p = gej_infinity();
        p.z = fe_one();
    }
    gej_store(gej, gid, p);
}

// ------------------------------------------------- K5: one inversion for a whole level
//
// Montgomery's trick, three passes. The runs are what make it parallel: each thread walks
// R consecutive z values on its own, and only the per-run totals need a sequential scan.
// That scan is one thread over count/R elements -- ~15k field multiplies against the
// level's ~200M, i.e. under a tenth of a percent. Priced, not assumed.

#define INVERT_RUN 64

// Pass A: per-run prefix products, and the run's total.
KERNEL k_invert_a(BUF(const u32, gej, 0), BUF(u32, prefix, 1), BUF(u32, totals, 2),
                  CBUF(u32, count, 3) GID_PARAM) {
    GID_INIT
    u32 lo = gid * INVERT_RUN;
    if (lo >= count) return;
    u32 hi = lo + INVERT_RUN;
    if (hi > count) hi = count;

    Fe running = fe_one();
    for (u32 i = lo; i < hi; i++) {
        fe_store(prefix, i, running);
        running = fe_mul(running, gej_load(gej, i).z);
    }
    fe_store(totals, gid, running);
}

// The same run decomposition applied to a plain array of field elements, so that pass B's
// input can itself be reduced before a single thread has to walk it.
//
// Without this second level, pass B walks count/64 elements on **one lane**. At a
// 2,048-seed launch that is 15,392 sequential field multiplies against a launch of about a
// second, and a single GPU lane runs a dependent chain at roughly 99k multiplies/s -- so
// the scan alone was ~23% of every launch. One more level divides it by another 64.
KERNEL k_invert_fe_a(BUF(const u32, src, 0), BUF(u32, prefix, 1), BUF(u32, totals, 2),
                     CBUF(u32, count, 3) GID_PARAM) {
    GID_INIT
    u32 lo = gid * INVERT_RUN;
    if (lo >= count) return;
    u32 hi = lo + INVERT_RUN;
    if (hi > count) hi = count;

    Fe running = fe_one();
    for (u32 i = lo; i < hi; i++) {
        fe_store(prefix, i, running);
        running = fe_mul(running, fe_load(src, i));
    }
    fe_store(totals, gid, running);
}

KERNEL k_invert_fe_c(BUF(const u32, src, 0), BUF(const u32, prefix, 1),
                     BUF(const u32, inv_totals, 2), BUF(u32, out, 3), CBUF(u32, count, 4)
                     GID_PARAM) {
    GID_INIT
    u32 lo = gid * INVERT_RUN;
    if (lo >= count) return;
    u32 hi = lo + INVERT_RUN;
    if (hi > count) hi = count;

    Fe inv = fe_load(inv_totals, gid);
    for (int i = (int)hi - 1; i >= (int)lo; i--) {
        Fe v = fe_load(src, (u32)i);
        fe_store(out, (u32)i, fe_mul(fe_load(prefix, (u32)i), inv));
        inv = fe_mul(inv, v);
    }
}

// Pass B: invert every run total, sequentially. One thread -- but by the time it runs, the
// level has been reduced twice, so it walks count/4096 elements rather than count/64.
KERNEL k_invert_b(BUF(const u32, totals, 0), BUF(u32, inv_totals, 1), CBUF(u32, runs, 2)
                  GID_PARAM) {
    GID_INIT
    if (gid != 0) return;

    Fe running = fe_one();
    for (u32 i = 0; i < runs; i++) {
        fe_store(inv_totals, i, running);
        running = fe_mul(running, fe_load(totals, i));
    }

    Fe inv = fe_inv(running);
    for (int i = (int)runs - 1; i >= 0; i--) {
        Fe pre = fe_load(inv_totals, (u32)i);
        Fe t = fe_load(totals, (u32)i);
        fe_store(inv_totals, (u32)i, fe_mul(pre, inv));
        inv = fe_mul(inv, t);
    }
}

// Pass C: each run walks back with its own 1/T, which is all it needs -- the runs before
// and after cancel exactly.
KERNEL k_invert_c(BUF(const u32, gej, 0), BUF(const u32, prefix, 1),
                  BUF(const u32, inv_totals, 2), BUF(u32, zinv, 3), CBUF(u32, count, 4)
                  GID_PARAM) {
    GID_INIT
    u32 lo = gid * INVERT_RUN;
    if (lo >= count) return;
    u32 hi = lo + INVERT_RUN;
    if (hi > count) hi = count;

    Fe inv = fe_load(inv_totals, gid);
    for (int i = (int)hi - 1; i >= (int)lo; i--) {
        Fe pre = fe_load(prefix, (u32)i);
        Fe z = gej_load(gej, (u32)i).z;
        fe_store(zinv, (u32)i, fe_mul(pre, inv));
        inv = fe_mul(inv, z);
    }
}

INLINE Ge affine_at(DEVICE const u32* gej, DEVICE const u32* zinv, u32 i) {
    return gej_to_ge(gej_load(gej, i), fe_load(zinv, i));
}

// ------------------------------------------------------------------ K7: normal children
//
// **One thread per parent, looping over its children** -- not one thread per child, which
// is the obvious shape and is wasteful. Everything a child needs from its parent is the
// same for all of them: the chain-code HMAC's two pad midstates (two SHA-512 compressions),
// the affine public key (a `gej_to_ge`, so a square and two multiplies), and the first 33
// bytes of the 37-byte data block. Only the trailing index differs. Per child that is 4
// compressions against 2, plus the curve and serialisation work, so the level's SHA-512
// cost falls by nearly half at ten indices per parent.
//
// The parallelism it gives up is affordable and worth checking rather than assuming: the
// index level goes from `seeds * 480` threads to `seeds * 48`, which at even a 2,048-seed
// launch is 98,000 -- far past what either device needs to saturate. `count` is therefore
// the *parent* count here, unlike every other kernel in this file.
//
// Measured on an M1 Pro, default scope, 20,000 seeds at a 2,048 batch, medians of three:
// this kernel 0.60s -> 0.29s (**-51%**, which is the predicted halving), and the whole
// sweep 2,540 -> 2,646 seeds/s (**+4.2%**).

KERNEL k_ckd_normal(BUF(const u32, parents, 0), BUF(const u32, gej, 1), BUF(const u32, zinv, 2),
                    BUF(u32, out, 3), CBUF(u32, count, 4), CBUF(u32, children, 5),
                    BUF(const u32, child_values, 6) GID_PARAM) {
    GID_INIT
    if (gid >= count) return;
    u32 base = gid * children;

    Node p = node_load(parents, gid);
    if (!p.valid) {
        // A dead parent's slots still have to be written: the level is dense, and whatever
        // the buffer held from the previous launch is not `valid = 0`.
        for (u32 c = 0; c < children; c++) node_store(out, base + c, node_dead());
        return;
    }

    Hmac512 key = hmac512_new(p.chain, 32);
    u8 data[37];
    ge_serialize(affine_at(gej, zinv, gid), data);

    for (u32 c = 0; c < children; c++) {
        u32 index = child_values[c];
        for (u32 i = 0; i < 4; i++) data[33 + i] = (u8)(index >> (24 - 8 * i));
        node_store(out, base + c, bip32_ckd_keyed(p.key, key, data));
    }
}

// ------------------------------------------------------------------ K8: leaves
//
// hash160 in every requested form, probed against the filter, and a record emitted for
// each survivor. The record is deliberately thin -- the host re-derives the seed through
// `derive::Deriver` to produce the phrase and the location, so nothing a user reads comes
// from here. See src/gpu/mod.rs.

#define HIT_STRIDE 32

KERNEL k_leaf(BUF(const u32, gej, 0), BUF(const u32, zinv, 1), BUF(const u64, filter, 2),
              CBUF(u64, filter_bits, 3), BUF(ATOMIC_U32, hit_count, 4), BUF(u8, hits, 5),
              CBUF(u32, count, 6), CBUF(u32, base_seed, 7), CBUF(u32, hit_capacity, 8),
              CBUF(u32, seeds, 9), CBUF(u32, filter_blocked, 10) GID_PARAM) {
    GID_INIT
    if (gid >= count) return;

    Gej j = gej_load(gej, gid);
    if (j.infinity) return;
    Ge p = affine_at(gej, zinv, gid);

    const u32 forms[ARRAY_N(N_FORMS)] = FORMS;

    // Two of the three forms are built from the compressed hash, so it is computed once
    // and shared -- but only if the scope actually asks for one of them. A sweep
    // restricted to `--hash-forms uncompressed` would otherwise pay a SHA-256 and a
    // RIPEMD-160 per key for a value it never looks at.
    u8 compressed[20];
    bool have_compressed = false;

    for (u32 f = 0; f < N_FORMS; f++) {
        u8 hash[20];
        if (forms[f] == FORM_UNCOMPRESSED) {
            u8 ser[65];
            ge_serialize_uncompressed(p, ser);
            hash160(ser, 65, hash);
        } else {
            if (!have_compressed) {
                u8 ser[33];
                ge_serialize(p, ser);
                hash160(ser, 33, compressed);
                have_compressed = true;
            }
            if (forms[f] == FORM_COMPRESSED) {
                for (u32 i = 0; i < 20; i++) hash[i] = compressed[i];
            } else {
                u8 script[22];
                script[0] = 0x00; // OP_0
                script[1] = 0x14; // push 20 bytes
                for (u32 i = 0; i < 20; i++) script[2 + i] = compressed[i];
                hash160(script, 22, hash);
            }
        }

        if (!bloom_contains(filter, filter_bits, filter_blocked, hash)) continue;

        u32 slot = atomic_add_u32(hit_count, 1u);
        if (slot >= hit_capacity) continue; // overflow is reported by the host, not lost here

        // Which seed produced this. The raw-privkey leaves sit after *every* HD leaf of
        // the whole launch -- that is where `walk_levels` puts them, so that they share
        // the leaf level's inversion -- which means the two regions index seeds
        // differently. `Layout::raw_leaf_base` is the same boundary on the host.
        u32 hd_total = HD_LEAVES_PER_SEED * seeds;
        u32 seed = base_seed
                 + ((gid < hd_total) ? (gid / HD_LEAVES_PER_SEED)
                                     : ((gid - hd_total) / MAX(N_RAW_SIZES, 1u)));

        DEVICE u8* rec = hits + slot * HIT_STRIDE;
        for (u32 i = 0; i < 4; i++) rec[i] = (u8)(seed >> (8 * i));
        for (u32 i = 0; i < 4; i++) rec[4 + i] = (u8)(gid >> (8 * i));
        rec[8] = (u8)f;
        rec[9] = 0; rec[10] = 0; rec[11] = 0;
        for (u32 i = 0; i < 20; i++) rec[12 + i] = hash[i];
    }
}

// ---------------------------------------------------------------- smoke
//
// Proves the whole path works before any real arithmetic depends on it: that the source
// assembles, that the runtime compiler accepts it, that a dispatch reaches the device, and
// that the scope defines arrived intact.
//
// It stays after the pipeline is finished. A one-dispatch check that the toolchain still
// works is worth its twenty lines when the alternative failure looks like wrong hashes.

KERNEL smoke(BUF(u32, out, 0) GID_PARAM) {
    GID_INIT
    const u32 entropy_sizes[ARRAY_N(N_ENTROPY_SIZES)] = ENTROPY_SIZES;
    const u32 chains[ARRAY_N(N_CHAINS)] = CHAINS;

    u32 acc = 0;
    for (u32 i = 0; i < N_ENTROPY_SIZES; i++) acc += entropy_sizes[i];
    for (u32 i = 0; i < N_CHAINS; i++) acc += chains[i];
    acc += N_INDICES * 1000u;
    acc += HASHES_PER_SEED * 1000000u;

    out[gid] = acc + gid;
}
