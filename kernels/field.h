// Arithmetic in the secp256k1 base field, GF(2^256 - 2^32 - 977), for the GPU.
//
// A mirror of src/field.rs, with one deliberate change: **eight 32-bit limbs instead of
// four 64-bit ones.** Neither Apple nor NVIDIA has a 64-bit integer ALU, so a `u64`
// multiply is emulated and a 4x64 schoolbook would pay that emulation 16 times per
// multiply. Eight 32-bit limbs with `u64` *accumulators* is the shape both compilers want:
// every product is a single 32x32->64 instruction, and the carry chain is a plain add.
//
// The same invariant as the CPU: elements are **always fully reduced**, below p. It costs
// a conditional subtract per operation and buys an invariant simple enough to test
// directly against `num-bigint`, which is what src/gpu/parity.rs does over random inputs.
//
// Everything here is by value on thread-private structs. That is what keeps `DEVICE` off
// the arithmetic and lets one source compile as both MSL and CUDA -- see compat.h.
//
// **Not constant time, by design.** Every scalar in this program derives from a published
// 32-bit timestamp, so there is no secret to leak. Do not reuse this for anything holding
// one.

// p = 2^256 - C.
#define FE_C 0x1000003D1UL

typedef struct { u32 n[8]; } Fe;

INLINE Fe fe_zero() { Fe r; for (u32 i = 0; i < 8; i++) r.n[i] = 0; return r; }
INLINE Fe fe_one()  { Fe r; r.n[0] = 1; for (u32 i = 1; i < 8; i++) r.n[i] = 0; return r; }

INLINE bool fe_is_zero(Fe a) {
    u32 acc = 0;
    UNROLL for (u32 i = 0; i < 8; i++) acc |= a.n[i];
    return acc == 0;
}

// The limbs of p, least significant first.
INLINE u32 fe_p_limb(u32 i) {
    // p = 2^256 - 0x1000003D1, so only the bottom two limbs differ from 0xffffffff.
    if (i == 0) return 0xFFFFFC2Fu;
    if (i == 1) return 0xFFFFFFFEu;
    return 0xFFFFFFFFu;
}

// Is `a` at or above p? Compared from the top down, which settles in one limb for almost
// every input.
INLINE bool fe_ge_p(Fe a) {
    UNROLL for (int i = 7; i >= 0; i--) {
        u32 pi = fe_p_limb((u32)i);
        if (a.n[i] != pi) return a.n[i] > pi;
    }
    return true; // exactly p
}

// Subtract p unconditionally. Only called where the result is known not to borrow.
INLINE Fe fe_sub_p(Fe a) {
    Fe r;
    u64 borrow = 0;
    UNROLL for (u32 i = 0; i < 8; i++) {
        u64 d = (u64)a.n[i] - (u64)fe_p_limb(i) - borrow;
        r.n[i] = (u32)d;
        borrow = (d >> 32) & 1;
    }
    return r;
}

// Restore the "below p" invariant for a value known to be under 2^256.
//
// The early-return compare is a data-dependent branch on the output of every field
// operation, which is the shape a GPU is supposed to hate. A branchless form -- subtract
// unconditionally, select on the borrow -- was written and measured at 4,371 seeds/s
// against this version's 4,474 on a curve-only scope (M1 Pro, 2026-08-13). No better, and
// further from `src/field.rs`, so the version that matches the CPU stays.
//
// **That measurement has a mechanism, and it says the same thing will happen on CUDA, so
// do not spend an afternoon re-running it there.** The branch looks divergent and is not:
//
//   * p is 2^256 - 0x1000003D1, so its top limb is 0xFFFFFFFF and `fe_ge_p` settles on
//     limb 7 unless the input's top limb is exactly 0xFFFFFFFF -- probability 2^-32. One
//     comparison, then a return, for essentially every value that ever reaches here.
//   * Every lane settles on the same limb for the same reason, so the branch is
//     warp-uniform rather than divergent. A warp does not pay for a branch its lanes agree
//     on, and 32 lanes disagreeing needs one of them to hold that 2^-32 top limb.
//   * The subtract itself is rarer still: a value under 2^256 is at or above p with
//     probability (2^32 + 977)/2^256, about 2^-224. `fe_canon` is a no-op in practice.
//
// So this costs roughly three instructions on the common path, and the branchless form
// costs twenty-four unconditionally -- eight subtract-with-borrow, then a masked select per
// limb. It is slower for an arithmetic reason that has nothing to do with which GPU is
// running it. The Metal number above is the confirmation, not the argument.
INLINE Fe fe_canon(Fe a) { return fe_ge_p(a) ? fe_sub_p(a) : a; }

// r += k, where k is small enough that the carry dies out. Used only to fold a 2^256
// overflow back in as k*C.
INLINE Fe fe_add_u64(Fe a, u64 k) {
    Fe r;
    u64 c = k;
    UNROLL for (u32 i = 0; i < 8; i++) {
        c += (u64)a.n[i];
        r.n[i] = (u32)c;
        c >>= 32;
    }
    return r;
}

// Addition, with the same carry-chain argument as `fe_mul` and a bigger share of the point
// addition than it looks.
//
// `gej_add_ge` is 11 multiplicative operations and **14 additive ones** -- 8 `fe_sub`, 1
// `fe_add`, and `fe_mul2`/`fe_mul4`, which are `fe_add(a, a)` in `ec.h`. Left in C the
// additive half is around a third of the kernel's instructions, for the same reason the
// multiply was: `u64 c` is a carry the machine has a flag for, and ptxas cannot turn it
// back into one.
//
// Two blocks rather than one, because the fold between them is a 64-bit multiply that
// belongs in C. Nothing carries across the boundary -- the first block leaves its carry in
// a named register, not in CC -- so splitting is safe where splitting a row of `fe_mul`
// would not be.
//
// The second block cannot carry out of limb 7: a and b are below p, so a + b < 2p < 2^257,
// the first carry is 0 or 1, and folding it as +C is exactly -p, which lands below p.
// Checked over 450,064 pairs against Python's mod-p arithmetic -- every combination of 0,
// 1, p-1, p-2, (p-1)/2, 2^255, 2^256-1-C and C with each other, 400,000 random pairs and
// 50,000 with p-1 held on one side. No mismatch, and that second carry was never 1.
//
// `MILKSAD_PORTABLE_FE_ADD` restores the portable body as the A/B control.
INLINE Fe fe_add(Fe a, Fe b) {
    Fe r;
#if defined(MILKSAD_CUDA) && !defined(MILKSAD_PORTABLE_FE_ADD)
    u32 c = 0;
    asm("add.cc.u32  %0, %9,  %17;\n\t"
        "addc.cc.u32 %1, %10, %18;\n\t"
        "addc.cc.u32 %2, %11, %19;\n\t"
        "addc.cc.u32 %3, %12, %20;\n\t"
        "addc.cc.u32 %4, %13, %21;\n\t"
        "addc.cc.u32 %5, %14, %22;\n\t"
        "addc.cc.u32 %6, %15, %23;\n\t"
        "addc.cc.u32 %7, %16, %24;\n\t"
        "addc.u32    %8, %8, 0;"
        : "=r"(r.n[0]), "=r"(r.n[1]), "=r"(r.n[2]), "=r"(r.n[3]), "=r"(r.n[4]),
          "=r"(r.n[5]), "=r"(r.n[6]), "=r"(r.n[7]), "+r"(c)
        : "r"(a.n[0]), "r"(a.n[1]), "r"(a.n[2]), "r"(a.n[3]), "r"(a.n[4]),
          "r"(a.n[5]), "r"(a.n[6]), "r"(a.n[7]),
          "r"(b.n[0]), "r"(b.n[1]), "r"(b.n[2]), "r"(b.n[3]), "r"(b.n[4]),
          "r"(b.n[5]), "r"(b.n[6]), "r"(b.n[7]));

    u64 k = (u64)c * FE_C;
    u32 klo = (u32)k, khi = (u32)(k >> 32);
    asm("add.cc.u32  %0, %0, %8;\n\t"
        "addc.cc.u32 %1, %1, %9;\n\t"
        "addc.cc.u32 %2, %2, 0;\n\t"
        "addc.cc.u32 %3, %3, 0;\n\t"
        "addc.cc.u32 %4, %4, 0;\n\t"
        "addc.cc.u32 %5, %5, 0;\n\t"
        "addc.cc.u32 %6, %6, 0;\n\t"
        "addc.u32    %7, %7, 0;"
        : "+r"(r.n[0]), "+r"(r.n[1]), "+r"(r.n[2]), "+r"(r.n[3]),
          "+r"(r.n[4]), "+r"(r.n[5]), "+r"(r.n[6]), "+r"(r.n[7])
        : "r"(klo), "r"(khi));
#else
    u64 c = 0;
    UNROLL for (u32 i = 0; i < 8; i++) {
        c += (u64)a.n[i] + (u64)b.n[i];
        r.n[i] = (u32)c;
        c >>= 32;
    }
    // a, b < p, so a + b < 2p < 2^257 and the carry is 0 or 1. Folding it as
    // `+ C` is exactly `- p`, and the result is then below p, so this cannot
    // overflow a second time.
    r = fe_add_u64(r, c * FE_C);
#endif
    return fe_canon(r);
}

// Subtract k*C from a value that has already wrapped past 2^256.
INLINE Fe fe_sub_kC(Fe a, u64 k) {
    Fe r;
    u64 sub = k * FE_C;
    u64 borrow = 0;
    UNROLL for (u32 i = 0; i < 8; i++) {
        u64 d = (u64)a.n[i] - (sub & 0xFFFFFFFFu) - borrow;
        r.n[i] = (u32)d;
        borrow = (d >> 32) & 1;
        sub >>= 32;
    }
    return r;
}

// p - a, which is -a for every a but zero.
INLINE Fe fe_neg(Fe a) {
    if (fe_is_zero(a)) return a;
#if defined(MILKSAD_CUDA) && !defined(MILKSAD_PORTABLE_FE_ADD)
    // p's limbs as immediates: only the bottom two differ from 0xffffffff. Nothing reads
    // the carry out, because a is below p and this cannot borrow -- see the note on
    // `fe_sub` for why that is the whole trick.
    Fe r;
    asm("sub.cc.u32  %0, 0xFFFFFC2F, %8;\n\t"
        "subc.cc.u32 %1, 0xFFFFFFFE, %9;\n\t"
        "subc.cc.u32 %2, 0xFFFFFFFF, %10;\n\t"
        "subc.cc.u32 %3, 0xFFFFFFFF, %11;\n\t"
        "subc.cc.u32 %4, 0xFFFFFFFF, %12;\n\t"
        "subc.cc.u32 %5, 0xFFFFFFFF, %13;\n\t"
        "subc.cc.u32 %6, 0xFFFFFFFF, %14;\n\t"
        "subc.u32    %7, 0xFFFFFFFF, %15;"
        : "=r"(r.n[0]), "=r"(r.n[1]), "=r"(r.n[2]), "=r"(r.n[3]),
          "=r"(r.n[4]), "=r"(r.n[5]), "=r"(r.n[6]), "=r"(r.n[7])
        : "r"(a.n[0]), "r"(a.n[1]), "r"(a.n[2]), "r"(a.n[3]),
          "r"(a.n[4]), "r"(a.n[5]), "r"(a.n[6]), "r"(a.n[7]));
    return r;
#else
    Fe r;
    u64 borrow = 0;
    UNROLL for (u32 i = 0; i < 8; i++) {
        u64 d = (u64)fe_p_limb(i) - (u64)a.n[i] - borrow;
        r.n[i] = (u32)d;
        borrow = (d >> 32) & 1;
    }
    return r;
#endif
}

// Subtraction, as an addition of the negation -- which is what settles the borrow question
// this file used to defer, by not asking it.
//
// The direct form needs the final borrow, to fold `borrow * C` back in. Extracting it with
// `subc.u32 t, 0, 0` yields 0 under one reading of the PTX carry convention and 0xFFFFFFFF
// under the other; getting that wrong is not a slowdown, it is silently wrong arithmetic on
// every subtraction, and it cannot be settled without a device.
//
// It does not have to be settled. **`p - b` cannot borrow** -- b is below p, by the
// invariant this file maintains everywhere -- so `fe_neg` reads no flag at all, and only
// the *composition* of `sub.cc` with `subc.cc` is relied on, which is what makes the
// multi-limb idiom work whichever way CC.CF points. Then `a + (p - b)` is `a - b (mod p)`
// through the `fe_add` above.
//
// The b == 0 case looks like a bug and is not: `fe_neg` returns 0 rather than p there, so
// this becomes `a + 0`, which is a. Written the other way -- returning p -- it would still
// be right, because a + p carries and folding that carry as +C is exactly -p. Both forms
// were checked against Python's mod-p arithmetic over 440,100 cases, 20,010 of them with
// `p - b` landing on exactly p. No mismatch.
//
// The cost is a `fe_neg` a direct chain would not need: ~35 instructions against a direct
// form's ~25 and the portable body's ~63. Taking the certain 1.8x over an uncertain 2.5x is
// the whole point. `fe_neg` is not only paid here either -- `ge_negate` calls it for every
// negative comb digit, which is about half of every row of `ec_mul_gen`.
INLINE Fe fe_sub(Fe a, Fe b) {
#if defined(MILKSAD_CUDA) && !defined(MILKSAD_PORTABLE_FE_ADD)
    return fe_add(a, fe_neg(b));
#else
    Fe r;
    u64 borrow = 0;
    UNROLL for (u32 i = 0; i < 8; i++) {
        u64 d = (u64)a.n[i] - (u64)b.n[i] - borrow;
        r.n[i] = (u32)d;
        borrow = (d >> 32) & 1;
    }
    // A borrow means the true difference was negative and `r` holds it plus 2^256.
    // Adding p is the fix, and on a value that has already wrapped, `+ p` is `- C`.
    return fe_canon(fe_sub_kC(r, borrow));
#endif
}


// Add k*C into eight limbs in place, returning the carry out.
//
// C = 2^32 + 977, so the multiply splits into a small product at limb 0 and the
// multiplier itself one limb up. That split is what keeps every intermediate inside a
// u64: `k * C` for k near 2^34 would not fit, but `k * 977` does.
INLINE u64 fe_add_kC(THREAD u32* r, u64 k) {
    u64 c = k * 977ul;
    UNROLL for (u32 i = 0; i < 8; i++) {
        c += (u64)r[i];
        if (i == 1) c += k; // the 2^32 term
        r[i] = (u32)c;
        c >>= 32;
    }
    return c;
}

// Multiply, with the 512-bit reduction folded in.
//
// **The reduction is inline rather than a call taking `THREAD const u64*`.** Taking a
// pointer to a local array forces it out of the register file and into thread-local
// memory, which on an Apple GPU is device-backed; every multiply then round-trips its
// sixteen intermediate limbs through memory. That single indirection was most of the
// distance between 1.9% and 22% of this machine's integer peak. Nothing here may take the
// address of a local -- see the note at the top of compat.h about staying pointer-free.
//
// The wide product is also `u32[16]` rather than `u64[16]`: each limb only ever holds a
// 32-bit value, so the wider array was doubling register pressure to store zeroes.
//
// 2^256 = C (mod p), so the top half folds into the bottom. Three stages, each over a
// strictly smaller value, which makes termination obvious rather than argued:
//
//   1. `lo + hi*C` -- 512 bits down to 290, in ten limbs. C's two terms are applied in
//      one sweep, the 977 at limb i and the 2^32 at limb i+1.
//   2. the two limbs above 2^256 fold back in. They hold under 2^34, so `k*C` is under
//      2^67 and cannot reach the top.
//   3. at most one bit of carry is left, and folding it cannot carry again.
// The 8x8 schoolbook as eight PTX carry chains, on CUDA only.
//
// **This is the one place the kernels stop being one source for two dialects, and it is
// worth saying why here rather than in a commit message.** The portable body below is the
// right code for Metal and the wrong code for NVIDIA, and the difference is not a spelling.
//
// A limb product has to add into a 512-bit accumulator, so every one of the 64 needs its
// carry propagated. Written in C the carry has to live in a value -- the `u64 carry` in the
// `#else` branch -- because C has no way to name the hardware carry flag. ptxas cannot
// recover it: it sees a 64-bit add whose high half feeds the next iteration, and emits an
// `IMAD.WIDE` plus two 64-bit adds where the machine has a single-instruction form. The
// carry flag exists, `mad.lo.cc.u32` / `madc.hi.cc.u32` are how PTX reaches it, and inline
// asm is the only way to write them.
//
// **The chain must be one asm block.** CC.CF is invisible to the compiler, so it will
// happily schedule something between two separate asm statements and lose the carry. Rows
// are safe to split because a row absorbs its own carry and starts the next chain fresh;
// within a row, nothing may be inserted, which is what makes this a 17-instruction block
// rather than 17 helper calls.
//
// The shape, for one row i, is two chained passes over `w[i..i+8]`:
//
//   * the low halves of a[i]*b[0..7], starting at limb i, then `addc` the carry into
//     limb i+8 -- which is still zero, because the accumulation so far is under
//     2^(32(i+8));
//   * the high halves, starting one limb up at i+1, ending at limb i+8.
//
// The carry out of that last instruction is dropped rather than absorbed, and that is the
// one place this deviates from a mechanical transcription. It is provably zero: after row
// i the accumulator is (sum_{k<=i} a[k] 2^32k) * B < 2^(32(i+1)) * 2^256 = 2^(32(i+9)),
// so limb i+9 is zero and nothing can carry into it. It has to be dropped for row 7, where
// limb 16 does not exist. Checked as well as argued -- 5.8 million products against the
// portable body below, including every pairing of zero, one, p, p-1, 2^256-1 and
// alternating-bit patterns with each other and with random inputs, plus 60,004 against
// Python's arbitrary precision with the dropped carry instrumented. No mismatch, and the
// dropped carry was never anything but zero.
//
// **The reduction stays in C.** It is ten limbs against the multiply's sixty-four, so it is
// most of the risk for a seventh of the benefit.
//
// **The known cost, and the thing to look at first if the A/B disappoints: `fe_sqr` loses
// its free triangle.** `fe_sqr(a)` is `fe_mul(a, a)`, and the note on it below records that
// the compiler collapses `a.n[i] * a.n[j]` against `a.n[j] * a.n[i]` by common-subexpression
// elimination without being told the identity -- which is why the hand-written triangle
// measured 12% *worse*. The compiler cannot see inside an asm block, so that collapse is
// gone here and a squaring now costs a full multiply. `gej_add_ge` is 7 multiplies and 4
// squarings, so this has to win more on the 7 than it loses on the 4. If it does not, a PTX
// triangle squaring is the next thing to write, and it would now be earning its keep rather
// than duplicating the optimiser.
//
// `MILKSAD_PORTABLE_FE_MUL` restores the portable body, so the A/B is an environment
// variable rather than an edit:
//
//     MILKSAD_KERNEL_DEFINES=-DMILKSAD_PORTABLE_FE_MUL \
//         ./target/release/milksad-scan bench -f addresses.blf --gpu only --seeds 3000000
#if defined(MILKSAD_CUDA) && !defined(MILKSAD_PORTABLE_FE_MUL)
#define FE_MUL_PTX 1

#define FE_MUL_ROW_(W0, W1, W2, W3, W4, W5, W6, W7, W8, AI,                    \
                    B0, B1, B2, B3, B4, B5, B6, B7)                            \
    asm("mad.lo.cc.u32   %0, %9, %10, %0;\n\t"                                 \
        "madc.lo.cc.u32  %1, %9, %11, %1;\n\t"                                 \
        "madc.lo.cc.u32  %2, %9, %12, %2;\n\t"                                 \
        "madc.lo.cc.u32  %3, %9, %13, %3;\n\t"                                 \
        "madc.lo.cc.u32  %4, %9, %14, %4;\n\t"                                 \
        "madc.lo.cc.u32  %5, %9, %15, %5;\n\t"                                 \
        "madc.lo.cc.u32  %6, %9, %16, %6;\n\t"                                 \
        "madc.lo.cc.u32  %7, %9, %17, %7;\n\t"                                 \
        "addc.u32        %8, %8, 0;\n\t"                                       \
        "mad.hi.cc.u32   %1, %9, %10, %1;\n\t"                                 \
        "madc.hi.cc.u32  %2, %9, %11, %2;\n\t"                                 \
        "madc.hi.cc.u32  %3, %9, %12, %3;\n\t"                                 \
        "madc.hi.cc.u32  %4, %9, %13, %4;\n\t"                                 \
        "madc.hi.cc.u32  %5, %9, %14, %5;\n\t"                                 \
        "madc.hi.cc.u32  %6, %9, %15, %6;\n\t"                                 \
        "madc.hi.cc.u32  %7, %9, %16, %7;\n\t"                                 \
        "madc.hi.u32     %8, %9, %17, %8;"                                     \
        : "+r"(W0), "+r"(W1), "+r"(W2), "+r"(W3), "+r"(W4),                    \
          "+r"(W5), "+r"(W6), "+r"(W7), "+r"(W8)                               \
        : "r"(AI), "r"(B0), "r"(B1), "r"(B2), "r"(B3),                         \
          "r"(B4), "r"(B5), "r"(B6), "r"(B7))

// Spelled with literal row indices at the call sites, so every subscript is a compile-time
// constant. An `"+r"` operand has to be a register, and a dynamically indexed local array
// cannot be one -- see the note at the top of compat.h.
#define FE_MUL_ROW(w, i, a, b)                                                 \
    FE_MUL_ROW_(w[(i) + 0], w[(i) + 1], w[(i) + 2], w[(i) + 3], w[(i) + 4],    \
                w[(i) + 5], w[(i) + 6], w[(i) + 7], w[(i) + 8], (a).n[i],      \
                (b).n[0], (b).n[1], (b).n[2], (b).n[3],                        \
                (b).n[4], (b).n[5], (b).n[6], (b).n[7])
#endif

INLINE Fe fe_mul(Fe a, Fe b) {
    u32 w[16];
    UNROLL for (u32 i = 0; i < 16; i++) w[i] = 0;
#if defined(FE_MUL_PTX)
    // Eight rows, each one PTX carry chain. See the note above `FE_MUL_ROW`.
    FE_MUL_ROW(w, 0, a, b);
    FE_MUL_ROW(w, 1, a, b);
    FE_MUL_ROW(w, 2, a, b);
    FE_MUL_ROW(w, 3, a, b);
    FE_MUL_ROW(w, 4, a, b);
    FE_MUL_ROW(w, 5, a, b);
    FE_MUL_ROW(w, 6, a, b);
    FE_MUL_ROW(w, 7, a, b);
#else
    UNROLL for (u32 i = 0; i < 8; i++) {
        // `(u64)x * (u64)y` on 32-bit operands, rather than an explicit `mul_hi32` pair.
        // The explicit form looks like it should win on a machine with no 64-bit ALU and
        // measured *slower*: 4,184 seeds/s against 4,474 on a curve-only scope, M1 Pro,
        // 2026-08-13. The compiler already recognises the narrow multiply, and spelling it
        // out only got in its way. Left as the plain form, with the experiment recorded so
        // it is not repeated.
        u64 carry = 0;
        UNROLL for (u32 j = 0; j < 8; j++) {
            u64 p = (u64)a.n[i] * (u64)b.n[j] + (u64)w[i + j] + carry;
            w[i + j] = (u32)p;
            carry = p >> 32;
        }
        // Never a read-modify-write: the inner loop only ever writes w[i..i+7], so this
        // limb is still zero when iteration i reaches it.
        w[i + 8] = (u32)carry;
    }
#endif

    u32 t[10];
    u64 c = 0;
    UNROLL for (u32 i = 0; i < 10; i++) {
        u64 acc = c;
        if (i < 8) acc += (u64)w[i];                  // lo
        if (i < 8) acc += (u64)w[i + 8] * 977ul;      // hi * 977, at limb i
        if (i >= 1 && i <= 8) acc += (u64)w[i + 7];   // hi * 2^32, at limb i
        t[i] = (u32)acc;
        c = acc >> 32;
    }

    Fe r;
    UNROLL for (u32 i = 0; i < 8; i++) r.n[i] = t[i];
    u64 k = (u64)t[8] | ((u64)t[9] << 32);

    // `fe_add_kC` inlined for the same reason the reduction is.
    u64 e = k * 977ul;
    UNROLL for (u32 i = 0; i < 8; i++) {
        e += (u64)r.n[i];
        if (i == 1) e += k;
        r.n[i] = (u32)e;
        e >>= 32;
    }
    r = fe_add_u64(r, e * FE_C);
    return fe_canon(r);
}

// Squaring is a multiply with equal arguments, and **deliberately nothing more**.
//
// The textbook optimisation is to compute only the strict upper triangle and double it:
// `a[i]*a[j]` and `a[j]*a[i]` are the same product, so 36 multiplies do the work of 64.
// That was written -- triangle, one fused doubling-and-diagonal pass, sharing this
// function's reduction -- verified against `num-bigint` by the parity tests, and measured
// on an M1 Pro over the default scope, interleaved A/B, best of eight:
//
//   k_kmul       1.39s -> 1.54s   (+12%)
//   whole sweep  3.18s -> 3.26s
//
// Slower, and reproducibly so. The reason is that the compiler is already doing it: with
// both operands the same SSA value, `a.n[i] * a.n[j]` and `a.n[j] * a.n[i]` are literally
// the same expression, and common-subexpression elimination collapses the pair without
// being told the algebraic identity. What the hand-written version added on top was the
// doubling and diagonal passes, which are pure cost -- and in `ec_mul_gen`'s row loop,
// where this is inlined four times per point addition, that cost lands on occupancy.
//
// Fusing the two extra passes into one made no difference (1.54s -> 1.55s), which is what
// confirms the multiplies were never the thing being saved.
//
// Recorded rather than deleted so the next person does not spend the afternoon on it.
//
// **On CUDA that argument no longer holds, and this is where the PTX multiply above has to
// pay for itself.** The collapse described here is the compiler's, and the compiler cannot
// see inside an asm block, so under `FE_MUL_PTX` a squaring really does compute all 64
// products. `gej_add_ge` is 7 multiplies and 4 squarings; if the A/B on `FE_MUL_PTX` comes
// out flat or worse, the triangle is the first thing to try again -- written in PTX this
// time, where it would be saving products nothing else is saving rather than duplicating an
// optimiser pass that had already run.
INLINE Fe fe_sqr(Fe a) { return fe_mul(a, a); }

// 1/a, by Fermat: a^(p-2). The addition chain is the same one src/field.rs uses -- 255
// squarings and 15 multiplies -- and its shape is what makes a single inversion per level
// (rather than per point) worth the batching in ec.h.
// Deliberately **not** inlined. This is an addition chain of 255 squarings and 15
// multiplies -- by far the largest function here -- and it is called once per level per
// thread, against which a call costs nothing. Inlining it at four call sites was enough to
// stop NVRTC finishing the parity translation unit at all.
NOINLINE Fe fe_inv(Fe a) {
    Fe x2  = fe_mul(fe_sqr(a), a);
    Fe x3  = fe_mul(fe_sqr(x2), a);

    Fe x6 = x3;
    for (u32 i = 0; i < 3; i++) x6 = fe_sqr(x6);
    x6 = fe_mul(x6, x3);

    Fe x9 = x6;
    for (u32 i = 0; i < 3; i++) x9 = fe_sqr(x9);
    x9 = fe_mul(x9, x3);

    Fe x11 = x9;
    for (u32 i = 0; i < 2; i++) x11 = fe_sqr(x11);
    x11 = fe_mul(x11, x2);

    Fe x22 = x11;
    for (u32 i = 0; i < 11; i++) x22 = fe_sqr(x22);
    x22 = fe_mul(x22, x11);

    Fe x44 = x22;
    for (u32 i = 0; i < 22; i++) x44 = fe_sqr(x44);
    x44 = fe_mul(x44, x22);

    Fe x88 = x44;
    for (u32 i = 0; i < 44; i++) x88 = fe_sqr(x88);
    x88 = fe_mul(x88, x44);

    Fe x176 = x88;
    for (u32 i = 0; i < 88; i++) x176 = fe_sqr(x176);
    x176 = fe_mul(x176, x88);

    Fe x220 = x176;
    for (u32 i = 0; i < 44; i++) x220 = fe_sqr(x220);
    x220 = fe_mul(x220, x44);

    Fe x223 = x220;
    for (u32 i = 0; i < 3; i++) x223 = fe_sqr(x223);
    x223 = fe_mul(x223, x3);

    Fe r = x223;
    for (u32 i = 0; i < 23; i++) r = fe_sqr(r);
    r = fe_mul(r, x22);
    for (u32 i = 0; i < 5; i++) r = fe_sqr(r);
    r = fe_mul(r, a);
    for (u32 i = 0; i < 3; i++) r = fe_sqr(r);
    r = fe_mul(r, x2);
    for (u32 i = 0; i < 2; i++) r = fe_sqr(r);
    return fe_mul(r, a);
}

// Big-endian 32 bytes, which is how every serialised key and hash input spells a field
// element, in and out.
INLINE Fe fe_from_be(THREAD const u8* b) {
    Fe r;
    UNROLL for (u32 i = 0; i < 8; i++) {
        u32 j = (7 - i) * 4;
        r.n[i] = ((u32)b[j] << 24) | ((u32)b[j + 1] << 16) | ((u32)b[j + 2] << 8) | (u32)b[j + 3];
    }
    return r;
}

INLINE void fe_to_be(Fe a, THREAD u8* b) {
    UNROLL for (u32 i = 0; i < 8; i++) {
        u32 j = (7 - i) * 4;
        b[j]     = (u8)(a.n[i] >> 24);
        b[j + 1] = (u8)(a.n[i] >> 16);
        b[j + 2] = (u8)(a.n[i] >> 8);
        b[j + 3] = (u8)(a.n[i]);
    }
}
