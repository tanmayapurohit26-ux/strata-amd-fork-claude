// HIP build only: stands in for NVIDIA's cuBLAS header, mapping the few names src/prefill/gemm.cu uses onto
// hipBLAS.  The constants follow llama.cpp's vendors/hip.h at the pinned commit, so they work on ROCm 6 and 7.
#pragma once
#include "strata/gpu/hip_compat.hpp"
#include <hipblas/hipblas.h>

#define cublasHandle_t hipblasHandle_t
#define cublasStatus_t hipblasStatus_t
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cublasGemmEx hipblasGemmEx
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUDA_R_16F  HIPBLAS_R_16F
#define CUDA_R_16BF HIPBLAS_R_16B
#define CUDA_R_32F  HIPBLAS_R_32F
#if HIP_VERSION >= 60500000
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#else
#define CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#endif
// rocBLAS manages its own workspace and has no math-mode switch (there is no TF32 to turn off), so these are
// no-ops that report success.
#define CUBLAS_DEFAULT_MATH 0
#define cublasSetMathMode(handle, mode) HIPBLAS_STATUS_SUCCESS
#define cublasSetWorkspace(handle, ptr, bytes) HIPBLAS_STATUS_SUCCESS
