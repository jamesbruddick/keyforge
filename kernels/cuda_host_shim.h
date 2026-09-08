// Enough CUDA to let an ordinary C++ compiler parse the CUDA dialect of these kernels.
//
// Test-only, like kernels/parity.h: it is prepended to the assembled translation unit by
// `gpu::source::tests::the_cuda_dialect_compiles`, and nothing here ships in a scanning
// binary or reaches a real device.
//
// **Why this exists.** The kernels are one source compiled as two dialects, and each
// compiler only ever sees its own half of every `#if`. A Mac compiles the Metal branch on
// every `cargo test --features metal` and learns nothing whatsoever about the CUDA branch,
// which is how a file that NVRTC rejects for an undeclared identifier once shipped -- see
// the note at the head of kernels/compat.h. Without a card in the machine there is no
// NVRTC to ask, so this asks clang instead.
//
// **What a pass does and does not prove.** It proves the CUDA branch parses, that every
// identifier it names is declared, that the declaration order satisfies its call graph,
// and that the types agree. It does not prove NVRTC accepts it -- NVRTC has its own
// builtins and its own limits -- and it says nothing about what the generated code does on
// a device. The definitions below are the simplest things with the right *signature*;
// several are deliberately not the right *semantics*, because nothing here is executed.
// The device tests remain the only evidence that the CUDA path computes anything correct.

#define __global__
#define __device__
#define __constant__
#define __shared__
#define __forceinline__ inline
#define __noinline__

// The launch-geometry builtins, as plain objects. A host compiler only has to see fields
// of the right type on them; `gid_value()` is never evaluated here.
struct KfDim3 {
    unsigned int x, y, z;
};
static KfDim3 blockIdx, threadIdx, blockDim;

// Intrinsics, by signature. The bodies are the obvious host equivalents so that anything
// reading this is not misled about intent, but they are never run.
static inline unsigned int __umulhi(unsigned int a, unsigned int b) {
    return (unsigned int)(((unsigned long long)a * (unsigned long long)b) >> 32);
}
static inline unsigned int __funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift) {
    unsigned long long wide = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    return (unsigned int)(wide >> (shift & 31u));
}
static inline unsigned int atomicAdd(unsigned int* p, unsigned int v) {
    unsigned int old = *p;
    *p += v;
    return old;
}
static inline void __syncthreads() {}
