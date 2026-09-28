// include/strata/gpu/hip_compat.hpp - AMD ROCm/HIP build: maps the CUDA names Strata uses onto HIP.
//
// Strata's GPU code is written against the CUDA runtime.  HIP mirrors that API almost name for name, so instead
// of rewriting ~60 files, the HIP build puts `hip_shim/` first on the include path: its <cuda_runtime.h>,
// <cuda_fp16.h> and <cublas_v2.h> all land here.  This header then includes the real HIP headers and #defines
// every cuda* / cublas* name the code uses onto its HIP equivalent.
//
// Where HIP's behaviour differs from CUDA's in a way that matters, there is a small wrapper instead of a plain
// #define, with a comment saying why.  Target: RDNA3 (gfx1100, RX 7900 XTX), wave32.  ROCm 6.x and 7.x.
#pragma once

#if !defined(STRATA_HIP)
#define STRATA_HIP 1
#endif

// HIP 6.2+ ships its own __shfl_*_sync, and from 7.0 they static_assert a 64-bit mask.  Strata passes the
// 32-bit 0xffffffff everywhere (correct for a wave32 GPU), so switch the builtins off and use the macros below,
// exactly as llama.cpp's HIP backend does.
#define HIP_DISABLE_WARP_SYNC_BUILTINS 1

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstddef>
#include <cstdint>

#if !defined(__HIP_PLATFORM_AMD__)
#error "Strata's HIP build supports only AMD GPUs"
#endif

// ---- types, status codes ---------------------------------------------------------------------------------
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorStreamCaptureUnsupported hipErrorStreamCaptureUnsupported
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaGraph_t hipGraph_t
#define cudaGraphExec_t hipGraphExec_t
#define cudaDeviceProp hipDeviceProp_t

// ---- device ---------------------------------------------------------------------------------------------
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaMemGetInfo hipMemGetInfo
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaPeekAtLastError hipPeekAtLastError
// HIP's "SharedMemPerBlockOptin" attribute is CUDA-only and reads 0 on AMD, which would make the QSA kernel
// refuse to start.  AMD has no opt-in tier: the ordinary per-block limit (64 KB on RDNA3) is the real one.
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

// hipFuncSetAttribute takes `const void*`; CUDA's is a template over the kernel's type.
template <typename F>
static inline hipError_t strata_hip_func_set_attribute(F* kernel, hipFuncAttribute attr, int value) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(kernel), attr, value);
}
#define cudaFuncSetAttribute strata_hip_func_set_attribute

// ---- memory ---------------------------------------------------------------------------------------------
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemcpyToSymbol(symbol, ...) hipMemcpyToSymbol(HIP_SYMBOL(symbol), __VA_ARGS__)

// Pinned host memory.  The flag values are the same bits in both APIs (Default 0, Portable 1, Mapped 2).
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaHostAllocPortable hipHostMallocPortable
#define cudaHostAllocMapped hipHostMallocMapped
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostRegisterMapped hipHostRegisterMapped
#define cudaHostRegister hipHostRegister
#define cudaHostUnregister hipHostUnregister
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaFreeHost hipHostFree

// THE ONE THAT MATTERS: Strata's "doorbells" are GPU kernels that spin on a flag the CPU writes into mapped
// host memory, and the CPU likewise waits on values the GPU writes there.  On NVIDIA mapped host memory is
// always coherent.  On AMD it is coherent only if asked for (or if HIP_HOST_COHERENT=1 is set), and without it
// the GPU may keep reading a stale cached flag and the engine hangs.  So every Mapped allocation asks for
// coherent memory explicitly.  Plain staging buffers (no Mapped flag) stay non-coherent, which is faster.
template <typename T>
static inline hipError_t strata_hip_host_alloc(T** ptr, size_t bytes, unsigned int flags) {
    if (flags & hipHostMallocMapped) flags |= hipHostMallocCoherent;
    return hipHostMalloc(reinterpret_cast<void**>(ptr), bytes, flags);
}
#define cudaHostAlloc strata_hip_host_alloc

// ---- streams, events, graphs -----------------------------------------------------------------------------
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamQuery hipStreamQuery
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventQuery hipEventQuery
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphUpload hipGraphUpload
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphGetNodes hipGraphGetNodes

// Strata uses CUDA 11's 5-argument cudaGraphInstantiate(exec, graph, nullptr, nullptr, 0).  HIP has that form
// too; the 3-argument (flags) form is routed to hipGraphInstantiateWithFlags in case a newer call site uses it.
static inline hipError_t strata_hip_graph_instantiate(hipGraphExec_t* exec, hipGraph_t graph, hipGraphNode_t* err_node,
                                                      char* log, size_t log_size) {
    return hipGraphInstantiate(exec, graph, err_node, log, log_size);
}
static inline hipError_t strata_hip_graph_instantiate(hipGraphExec_t* exec, hipGraph_t graph,
                                                      unsigned long long flags = 0) {
    return hipGraphInstantiateWithFlags(exec, graph, flags);
}
#define cudaGraphInstantiate strata_hip_graph_instantiate

// ---- warp-level operations (RDNA3 runs 32-wide waves, like an NVIDIA warp) ---------------------------------
// The mask argument is dropped: every Strata call site uses the full 0xffffffff mask.
#define __shfl_sync(mask, ...) __shfl(__VA_ARGS__)
#define __shfl_up_sync(mask, ...) __shfl_up(__VA_ARGS__)
#define __shfl_down_sync(mask, ...) __shfl_down(__VA_ARGS__)
#define __shfl_xor_sync(mask, ...) __shfl_xor(__VA_ARGS__)
#define __all_sync(mask, pred) __all(pred)
#define __any_sync(mask, pred) __any(pred)
// HIP's __ballot returns 64 bits; on a wave32 GPU the top 32 are zero.
#define __ballot_sync(mask, pred) ((unsigned) __ballot(pred))

// Device helpers below only exist for the HIP compiler; plain .cpp files (compiled by g++) see just the names.
#if defined(__HIPCC__)
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN_WAVEFRONT_SIZE)
#if __AMDGCN_WAVEFRONT_SIZE != 32
#error "Strata's kernels assume 32-lane waves: build for RDNA (gfx10/11/12), not wave64 GCN/CDNA"
#endif
#endif

// __syncwarp: a wave executes in lock-step, but the compiler must still not move shared-memory accesses across
// it, so it is a wave-scope fence + barrier (what HIP 6.2+'s own __syncwarp compiles to).
static __device__ __forceinline__ void strata_hip_syncwarp() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
#define __syncwarp(...) strata_hip_syncwarp()

// __nanosleep: only used inside spin-waits.  s_sleep 1 is ~64 clocks, the same order as CUDA's 100 ns hint.
static __device__ __forceinline__ void strata_hip_nanosleep(unsigned) { __builtin_amdgcn_s_sleep(1); }
#define __nanosleep strata_hip_nanosleep

// ---- byte-wise integer helpers -----------------------------------------------------------------------------
// __dp4a: signed 4 x int8 dot product accumulated into an int.  RDNA3 has it as one instruction (v_dot4_i32_iu8).
static __device__ __forceinline__ int strata_hip_dp4a(const int a, const int b, const int c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    const int8_t* a8 = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b8 = reinterpret_cast<const int8_t*>(&b);
    return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
#endif
}
static __device__ __forceinline__ int strata_hip_dp4a(const unsigned a, const unsigned b, const unsigned c) {
    const uint8_t* a8 = reinterpret_cast<const uint8_t*>(&a);
    const uint8_t* b8 = reinterpret_cast<const uint8_t*>(&b);
    return (int) (c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3]);
}
#define __dp4a strata_hip_dp4a

// __vsub4: per-byte subtraction with wrap-around, bit-exact to CUDA's (llama.cpp maps it to the SATURATING
// form, which agrees only when nothing overflows; this is the exact one).
static __device__ __forceinline__ unsigned strata_hip_vsub4(const unsigned a, const unsigned b) {
    return ((a | 0x80808080u) - (b & 0x7f7f7f7fu)) ^ ((a ^ ~b) & 0x80808080u);
}
#define __vsub4 strata_hip_vsub4

// __vcmpne4: 0xff in each byte where a and b differ, 0x00 where equal (CUDA semantics).
static __device__ __forceinline__ unsigned strata_hip_vcmpne4(const unsigned a, const unsigned b) {
    const unsigned x = a ^ b;                                            // non-zero bytes differ
    const unsigned t = ((x & 0x7f7f7f7fu) + 0x7f7f7f7fu) | x;            // top bit of each byte = byte != 0
    return ((t >> 7) & 0x01010101u) * 0xffu;
}
#define __vcmpne4 strata_hip_vcmpne4

// __vsubss4: per-byte SIGNED subtraction, saturated to [-128, 127] (CUDA semantics).
static __device__ __forceinline__ int strata_hip_vsubss4(const int a, const int b) {
    int r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int v = (int) (int8_t) (a >> (8 * i)) - (int) (int8_t) (b >> (8 * i));
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        r |= (v & 0xff) << (8 * i);
    }
    return r;
}
#define __vsubss4 strata_hip_vsubss4

#endif  // __HIPCC__
