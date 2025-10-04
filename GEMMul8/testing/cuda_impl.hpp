#pragma once

#if defined(__NVCC__)
    #include <cuComplex.h>
    #include <cublas_v2.h>
    #include <cuda_runtime.h>
    #include <curand_kernel.h>
    #include <nvml.h>

    #define GET_DEV_POWER(gpu_id, power_val)                                     \
        do {                                                                     \
            nvmlDevice_t device;                                                 \
            nvmlReturn_t result = nvmlDeviceGetHandleByIndex((gpu_id), &device); \
            if (result != NVML_SUCCESS) {                                        \
                std::cerr << "Failed to get GPU handle: "                        \
                          << nvmlErrorString(result) << std::endl;               \
                (power_val) = 0.0;                                               \
                break;                                                           \
            }                                                                    \
            unsigned int power_raw;                                              \
            result = nvmlDeviceGetPowerUsage(device, &power_raw);                \
            if (result != NVML_SUCCESS) {                                        \
                std::cerr << "Failed to get power usage: "                       \
                          << nvmlErrorString(result) << std::endl;               \
                (power_val) = 0.0;                                               \
                break;                                                           \
            }                                                                    \
            (power_val) = static_cast<double>(power_raw) / 1000.0;               \
        } while (0)

    #define NVML_INIT                                              \
        do {                                                       \
            nvmlReturn_t result = nvmlInit();                      \
            if (result != NVML_SUCCESS) {                          \
                std::cerr << "Failed to initialize NVML: "         \
                          << nvmlErrorString(result) << std::endl; \
            }                                                      \
        } while (0)

#endif

#if defined(__HIPCC__)
    #include <hip/hip_complex.h>
    #include <hip/hip_runtime.h>
    #include <hipblas/hipblas.h>
    #include <hiprand/hiprand_kernel.h>
    #include <rocm_smi/rocm_smi.h>

    #define cublasCreate                 hipblasCreate
    #define cublasDestroy                hipblasDestroy
    #define cublasHandle_t               hipblasHandle_t
    #define cublasOperation_t            hipblasOperation_t
    #define cublasGemmEx                 hipblasGemmEx_v2
    #define CUBLAS_OP_N                  HIPBLAS_OP_N
    #define CUBLAS_OP_T                  HIPBLAS_OP_T
    #define CUBLAS_OP_C                  HIPBLAS_OP_C
    #define CUDA_R_8I                    HIP_R_8I
    #define CUDA_R_32I                   HIP_R_32I
    #define CUDA_R_32F                   HIP_R_32F
    #define CUDA_R_64F                   HIP_R_64F
    #define CUDA_C_32F                   HIP_C_32F
    #define CUDA_C_64F                   HIP_C_64F
    #define CUBLAS_COMPUTE_32I           HIPBLAS_COMPUTE_32I
    #define CUBLAS_GEMM_DEFAULT          HIPBLAS_GEMM_DEFAULT
    #define CUBLAS_COMPUTE_32F           HIPBLAS_COMPUTE_32F
    #define CUBLAS_COMPUTE_64F           HIPBLAS_COMPUTE_64F
    #define CUBLAS_COMPUTE_32F_FAST_TF32 HIPBLAS_COMPUTE_32F_FAST_TF32

    #define cudaDeviceSynchronize                  hipDeviceSynchronize
    #define cudaMemcpyToSymbol(symbol, src, count) hipMemcpyToSymbol(HIP_SYMBOL(symbol), src, count)
    #define cudaDeviceProp                         hipDeviceProp_t
    #define cudaGetDeviceProperties                hipGetDeviceProperties
    #define cudaMalloc                             hipMalloc
    #define cudaMemcpy                             hipMemcpy
    #define cudaMemcpyDeviceToHost                 hipMemcpyDeviceToHost
    #define cudaFree                               hipFree

    #define cuComplex              hipFloatComplex
    #define cuFloatComplex         hipFloatComplex
    #define cuDoubleComplex        hipDoubleComplex
    #define cuCreal                hipCreal
    #define cuCrealf               hipCrealf
    #define cuCimag                hipCimag
    #define cuCimagf               hipCimagf
    #define make_cuComplex         make_hipFloatComplex
    #define make_cuFloatComplex    make_hipFloatComplex
    #define make_cuDoubleComplex   make_hipDoubleComplex
    #define cuCmul                 hipCmul
    #define cuCmulf                hipCmulf
    #define cuCadd                 hipCadd
    #define cuCaddf                hipCaddf
    #define cuCfma                 hipCfma
    #define cuCfmaf                hipCfmaf
    #define cuCabs                 hipCabs
    #define cuCsub                 hipCsub
    #define cuComplexFloatToDouble hipComplexFloatToDouble
    #define cuComplexDoubleToFloat hipComplexDoubleToFloat

    #define curandState           hiprandState
    #define curand_init           hiprand_init
    #define curand_uniform_double hiprand_uniform_double
    #define curand_normal_double  hiprand_normal_double

    #define nvmlShutdown rsmi_shut_down
    #define GET_DEV_POWER(gpu_id, power_val)                                            \
        do {                                                                            \
            uint64_t power_raw;                                                         \
            RSMI_POWER_TYPE power_type;                                                 \
            rsmi_status_t result = rsmi_dev_power_get(gpu_id, &power_raw, &power_type); \
            if (result != RSMI_STATUS_SUCCESS) {                                        \
                const char *error_string;                                               \
                rsmi_status_string(result, &error_string);                              \
                std::cerr << "Failed to get power usage: "                              \
                          << error_string << std::endl;                                 \
                (power_val) = 0.0;                                                      \
                break;                                                                  \
            }                                                                           \
            (power_val) = static_cast<double>(power_raw) / 1.e6;                        \
        } while (0)

    #define NVML_INIT                                          \
        do {                                                   \
            rsmi_status_t result = rsmi_init(0);               \
            if (result != RSMI_STATUS_SUCCESS) {               \
                const char *error_string;                      \
                rsmi_status_string(result, &error_string);     \
                std::cerr << "Failed to initialize ROCm SMI: " \
                          << error_string << std::endl;        \
                break;                                         \
            }                                                  \
        } while (0)

#endif
