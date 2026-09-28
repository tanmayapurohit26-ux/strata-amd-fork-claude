// src/kernels/native_qsa_score_test.cpp - AMD port check for the QSA block scorer (native_qsa_score).
//
// The NVIDIA kernel uses TF32 tensor-core instructions RDNA3 does not have; the AMD build replaces it with plain
// F32 multiply-adds.  This test runs the kernel on the GPU against a double-precision CPU reference of the same
// rule (per block: sum over 4 heads of ReLU(pooled . query_h), + optional bias, + 1e9 on an incomplete tail
// block; every cell of a block gets its block's score) and reports the largest relative error.
//
// Runs on either build.  Pass: every cell within 1e-4 relative (TF32 on NVIDIA is ~1e-3; F32 should be ~1e-6).
#include "strata/kernels/native_qsa_score.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace k = strata::kernels;

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "native_qsa_score_test: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    const int D = 128, H = 4, R = 4;
    const int64_t max_cells = 16384, max_blocks = max_cells / R + 1;
    k::QsaShapes s = k::qsa_real_shapes();

    std::mt19937 g(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> pooled((size_t) max_blocks * D), query((size_t) H * D), bias((size_t) max_blocks);
    for (auto& v : pooled) v = nd(g);
    for (auto& v : query) v = nd(g) * 0.25f;
    for (auto& v : bias) v = nd(g) * 0.1f;

    float *d_pooled, *d_query, *d_bias, *d_cells;
    int32_t* d_step;
    ck(cudaMalloc((void**) &d_pooled, pooled.size() * 4), "malloc");
    ck(cudaMalloc((void**) &d_query, query.size() * 4), "malloc");
    ck(cudaMalloc((void**) &d_bias, bias.size() * 4), "malloc");
    ck(cudaMalloc((void**) &d_cells, (size_t) max_cells * 4), "malloc");
    ck(cudaMalloc((void**) &d_step, k::kStepCount * 4), "malloc");
    ck(cudaMemcpy(d_pooled, pooled.data(), pooled.size() * 4, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(d_query, query.data(), query.size() * 4, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(d_bias, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice), "copy");
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");

    double worst = 0.0;
    int failures = 0;
    for (int n : {1, 3, 4, 5, 127, 128, 129, 2050, 2051, 2052, 4097, 9999, 16383}) {
        for (bool use_bias : {false, true}) {
            const int full = n / R;
            int32_t step[k::kStepCount] = {};
            step[k::kStepPos] = n - 1;
            step[k::kStepNKv] = n;
            step[k::kStepNBid] = full;
            step[k::kStepWidth] = n < 2051 ? n : 2051;
            ck(cudaMemcpy(d_step, step, sizeof(step), cudaMemcpyHostToDevice), "copy step");
            ck(cudaMemset(d_cells, 0xff, (size_t) max_cells * 4), "memset");
            k::native_qsa_score(d_pooled, d_query, use_bias ? d_bias : nullptr, s, d_step, max_blocks, max_cells,
                                d_cells, st);
            ck(cudaStreamSynchronize(st), "sync");
            std::vector<float> cells((size_t) max_cells);
            ck(cudaMemcpy(cells.data(), d_cells, cells.size() * 4, cudaMemcpyDeviceToHost), "copy back");
            double case_worst = 0.0;
            for (int row = 0; row <= full; ++row) {
                double sum = 0.0, mag = 0.0;
                for (int h = 0; h < H; ++h) {
                    double dot = 0.0, dmag = 0.0;
                    for (int c = 0; c < D; ++c) {
                        dot += (double) pooled[(size_t) row * D + c] * query[(size_t) h * D + c];
                        dmag += std::fabs((double) pooled[(size_t) row * D + c] * query[(size_t) h * D + c]);
                    }
                    sum += dot > 0.0 ? dot : 0.0;
                    mag += dmag;
                }
                if (use_bias) { sum += bias[row]; mag += std::fabs(bias[row]); }
                const bool tail = row == full && n % R;
                for (int i = row * R; i < n && i < (row + 1) * R; ++i) {
                    const double got = cells[i];
                    double err;
                    if (tail) err = std::fabs(got - (sum + 1e9)) / 1e9;
                    else err = std::fabs(got - sum) / std::max(mag, 1e-30);
                    if (!(err <= 1e-4)) {       // also catches NaN / unwritten (0xffffffff) cells
                        if (failures < 10)
                            std::fprintf(stderr, "  n=%d bias=%d cell %d: got %.9g want %.9g\n", n, (int) use_bias, i,
                                         got, tail ? sum + 1e9 : sum);
                        ++failures;
                    }
                    case_worst = std::max(case_worst, err);
                }
            }
            worst = std::max(worst, case_worst);
            std::printf("n=%5d bias=%d  worst relative error %.3g\n", n, (int) use_bias, case_worst);
        }
    }
    std::printf("native_qsa_score_test: %s (worst %.3g, %d bad cells)\n", failures ? "FAIL" : "PASS", worst, failures);
    return failures ? 1 : 0;
}
