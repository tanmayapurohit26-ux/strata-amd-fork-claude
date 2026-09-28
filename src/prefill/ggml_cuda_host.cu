// src/prefill/ggml_cuda_host.cu - prompt-speed plan step 2b: the host-side symbols of llama.cpp's ggml-cuda that its MMQ
// and quantize code reference, for the MMQ kernels compiled into strata_mmq without the rest of ggml-cuda.cu.
#include "common.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    std::fprintf(stderr, "ggml-cuda (strata mmq): %s: %s\n  in %s at %s:%d\n", msg, stmt, func, file, line);
    std::abort();
}

int ggml_cuda_get_device() {
    int id = 0;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = [] {
        ggml_cuda_device_info in = {};
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
        n = n > GGML_CUDA_MAX_DEVICES ? GGML_CUDA_MAX_DEVICES : n;
        in.device_count = n;
        in.physical_device_count = n;
        for (int id = 0; id < n; ++id) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
            auto & d = in.devices[id];
#if defined(GGML_USE_HIP)
            // AMD: llama.cpp encodes the gfx id ("gfx1100" -> 0x1100) above GGML_CUDA_CC_OFFSET_AMD; MMQ picks its
            // RDNA3 code path from this number (ggml-cuda.cu, ggml_cuda_parse_id).
            {
                const char * arch = prop.gcnArchName;
                if (std::strncmp(arch, "gfx", 3) == 0) arch += 3;
                char digits[16] = {0};
                size_t len = std::strcspn(arch, ":-");
                if (len >= sizeof(digits)) len = sizeof(digits) - 1;
                std::memcpy(digits, arch, len);
                int major = 0, minor = 0;
                if (len >= 3) {
                    minor = (int) std::strtoul(digits + len - 2, nullptr, 16);
                    digits[len - 2] = '\0';
                    major = (int) std::strtoul(digits, nullptr, 16);
                }
                d.cc = GGML_CUDA_CC_OFFSET_AMD + major * 0x100 + minor;
                if ((d.cc & 0xff00) == 0 && prop.major > 0)
                    d.cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100 + prop.minor * 0x10;
            }
            d.nsm = prop.multiProcessorCount;
            d.smpb = prop.sharedMemPerBlock;
            d.smpbo = prop.sharedMemPerBlock;   // AMD has no opt-in shared-memory tier
#else
            d.cc = 100 * prop.major + 10 * prop.minor;
            d.nsm = prop.multiProcessorCount;
            d.smpb = prop.sharedMemPerBlock;
            d.smpbo = prop.sharedMemPerBlockOptin;
#endif
            d.integrated = prop.integrated != 0;
            d.vmm = false;
            d.total_vram = prop.totalGlobalMem;
            d.warp_size = prop.warpSize;
            d.supports_cooperative_launch = prop.cooperativeLaunch != 0;
            d.physical_device = id;
            d.physical_share_count = 1;
            d.virtual_index = 0;
        }
        return in;
    }();
    return info;
}

namespace {
// Buffers are kept and reused: MMQ asks for the same few sizes every launch (its stream-k fixup tiles).
struct CachingPool : ggml_cuda_pool {
    struct Buf { void * p; size_t size; bool used; };
    std::vector<Buf> bufs;
    std::mutex mu;
    void * alloc(size_t size, size_t * actual_size) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & b : bufs)
            if (!b.used && b.size >= size) { b.used = true; *actual_size = b.size; return b.p; }
        void * p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, size));
        bufs.push_back({p, size, true});
        *actual_size = size;
        return p;
    }
    void free(void * ptr, size_t) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & b : bufs)
            if (b.p == ptr) { b.used = false; return; }
    }
    ~CachingPool() override {
        for (auto & b : bufs) cudaFree(b.p);
    }
};
}  // namespace

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int, int) {
    return std::unique_ptr<ggml_cuda_pool>(new CachingPool());
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {}
