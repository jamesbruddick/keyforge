// Metal and CUDA, behind one set of names.
//
// The kernels are written once and compiled as both Metal Shading Language and CUDA.
// Both are C++ dialects and agree on far more than they disagree on; this header is the
// list of what they do not agree on, and it is short on purpose.
//
// The rule that keeps it short: **the arithmetic is pointer-free.** Field elements,
// curve points, hash states and MT state are passed and returned by value on
// thread-private structs, never through pointers. MSL demands an address-space qualifier
// on every pointer and CUDA forbids one, so every pointer in the source is a place where
// `DEVICE` has to appear and be right. Keeping them to the kernel signatures and a
// handful of loaders is what stops this file from becoming a maintenance liability.
//
// Anything added here should be a genuine dialect difference. Anything that is merely
// convenient belongs in the header that uses it.
//
// **A `#if`-split function has two call graphs, and only one compiler will check each.**
// `fe_sub` in kernels/field.h is `fe_add(a, fe_neg(b))` on CUDA and a limb loop on Metal,
// so it calls `fe_neg` under one dialect and not the other. Definition order therefore has
// to satisfy the *union* of both arms' call graphs, and a Mac will happily compile a file
// that NVRTC rejects for an undeclared identifier -- which is exactly how one shipped.
// Building the Metal side is not evidence about the CUDA side, and vice versa.

#if defined(KEYFORGE_METAL)

#include <metal_stdlib>
using namespace metal;

// Address spaces. MSL requires these; CUDA has no notion of them.
#define DEVICE   device
#define CONSTANT constant
#define THREAD   thread
#define LOCAL    threadgroup

// A kernel entry point, and how a thread learns where it is.
#define KERNEL   kernel void
#define GID      uint gid [[thread_position_in_grid]]
#define INLINE   inline
#define NOINLINE __attribute__((noinline))

// Barriers and atomics.
#define BARRIER() threadgroup_barrier(mem_flags::mem_threadgroup)
typedef atomic_uint ATOMIC_U32;
INLINE uint atomic_add_u32(DEVICE ATOMIC_U32* p, uint v) {
    return atomic_fetch_add_explicit(p, v, memory_order_relaxed);
}

// The one arithmetic primitive the two spell differently. Everything else -- including
// 64-bit add, shift and rotate -- is written plainly and left to the compiler, which
// measured faster than hand-splitting into hi/lo pairs on both backends: 178.6 vs 169.7 M
// compressions/s on an M1 Pro (2026-08-13), and 107,227 vs 107,085 seeds/s on an RTX 5070
// Ti. Both compilers already emit the funnel shift for a constant-distance rotate. See
// kernels/sha512.h, which records what that cost to find out twice.
INLINE uint mul_hi32(uint a, uint b) { return mulhi(a, b); }

#elif defined(KEYFORGE_CUDA)

#define DEVICE
#define CONSTANT __constant__
#define THREAD
#define LOCAL    __shared__

#define KERNEL   extern "C" __global__ void
#define GID      /* see gid() below */
// `inline`, not `__forceinline__`. The two dialects mean different things by the same
// word: MSL's `inline` is a hint the compiler may decline, while `__forceinline__` is a
// directive nvcc obeys. Obeying it here is ruinous -- `fe_inv` alone is ~270 sequential
// field operations, each an unrolled 8x8 multiply, so every call site expands to tens of
// thousands of instructions and NVRTC stops finishing. Let the compiler decide; it inlines
// the small things anyway.
#define INLINE   __device__ inline
#define NOINLINE __noinline__ __device__

#define BARRIER() __syncthreads()
typedef unsigned int ATOMIC_U32;
INLINE unsigned int atomic_add_u32(ATOMIC_U32* p, unsigned int v) { return atomicAdd(p, v); }

INLINE unsigned int mul_hi32(unsigned int a, unsigned int b) { return __umulhi(a, b); }

// CUDA has no attribute for this, so it is a call rather than a parameter. The kernels
// use `GID_INIT` and then `gid`, which reads the same under both dialects.
#define gid_value() (blockIdx.x * blockDim.x + threadIdx.x)

#else
#error "define KEYFORGE_METAL or KEYFORGE_CUDA"
#endif

// How a kernel body gets its thread index, spelled once.
#if defined(KEYFORGE_METAL)
#define GID_PARAM , GID
#define GID_INIT
#else
#define GID_PARAM
#define GID_INIT const u32 gid = gid_value();
#endif

// Kernel parameters.
//
// MSL binds each one to an explicit slot and takes small scalars by reference out of the
// constant address space; CUDA takes a plain pointer or value and numbers arguments by
// position. Both are spelled here so a kernel signature reads the same under either.
// `n` must match the argument's index in the `Arg` slice the host dispatches with.
//
// **The CUDA side marks every buffer `__restrict__`, and that is a promise the host has to
// keep**: no two buffer arguments of a single dispatch may be the same buffer. Every launch
// in `Gpu::run` and `Gpu::public_keys` satisfies it today -- the two `k_ckd_normal` steps
// read `prefixes` into `chain_nodes` and then `chain_nodes` into `leaf_nodes`, never a
// buffer into itself -- and anything added there must too.
//
// It is worth the obligation because of what it unlocks. Without it nvcc must assume a
// store through any pointer can alias a load through any other, so it cannot keep a value
// in a register across a store and cannot route reads through the read-only data cache. The
// comb is the case that matters: `ec_comb_entry` does one random 64-byte gather per digit
// out of a 35 MB table, seventeen per scalar, on the kernel that is 40% of the sweep --
// exactly the access pattern `LDG` exists for. MSL infers the same non-aliasing from its
// address spaces, so the Metal branch needs nothing.
#if defined(KEYFORGE_METAL)
#define BUF(type, name, n)  DEVICE type* name [[buffer(n)]]
#define CBUF(type, name, n) CONSTANT type& name [[buffer(n)]]
#else
#define BUF(type, name, n)  type* __restrict__ name
#define CBUF(type, name, n) type name
#endif

// Fixed-width names, so the source never says `unsigned long` and means two things.
typedef unsigned int  u32;
typedef unsigned char u8;
// Signed, for the one generator that needs it: glibc's seeding does its Schrage
// reduction in signed arithmetic and relies on C's truncation toward zero, so porting it
// to unsigned would diverge for exactly the seeds whose top bit is set. See
// kernels/vuln/glibc.h.
typedef int i32;
#if defined(KEYFORGE_METAL)
typedef long  i64;
typedef ulong u64;
// A 64-bit literal. Metal has no `long long`, so `ull` is not available here; CUDA has no
// 64-bit `unsigned long` on Windows, so `ul` is not portable there. The suffix is spelled
// per dialect and everything else says `U64C`.
#define U64C(x) x##ul
#else
typedef long long          i64;
typedef unsigned long long u64;
#define U64C(x) x##ull
#endif

// The kernel-side encodings of `derive::Variant` and `derive::HashForm`. These must match
// `variant_code` and `form_code` in src/gpu/source.rs; they are duplicated rather than
// generated because a `#define` per enum variant reads worse than the table does.
#define VARIANT_BIP39       0
#define VARIANT_RAW_MASTER  1
#define VARIANT_RAW_PRIVKEY 2

#define FORM_COMPRESSED   0
#define FORM_UNCOMPRESSED 1
#define FORM_P2SH_P2WPKH  2

// Full unrolling, which for this code is a correctness-of-codegen matter rather than a
// micro-optimisation. A loop like `w[i + j] = ...` over local arrays keeps its index
// dynamic unless the loop is unrolled, and a dynamically indexed local array cannot live
// in registers -- it goes to thread-local memory, which on an Apple GPU is device-backed.
// Both dialects spell the hint the same way.
#define UNROLL _Pragma("unroll")

// Spelled out because MSL and CUDA disagree about which `max` overloads exist for mixed
// integer types, and the kernels only ever need this one.
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// An array length that is never zero.
//
// A scope can empty one of the scope lists -- `--variants raw-privkey` leaves no HD
// variants -- and a zero-length array is a compile error in C++, which both kernel
// compilers enforce. The one-element array that results is never read: every loop over one
// of these is bounded by the real `N_*` count, which is genuinely zero.
#define ARRAY_N(n) (MAX((n), 1u))

// The marker `gpu::source` emits for `Purpose::Bare`, which takes no hardened steps at
// all. Not a value a real BIP44 purpose can hold: it would alias a hardened index, and
// `main::to_scope` rejects those on the way in.
#define PURPOSE_BARE 0xffffffffu
