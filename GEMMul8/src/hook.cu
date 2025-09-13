#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <vector>

#include "../include/gemmul8.hpp" // gemmul8::gemm, gemmul8::workSize

// ==========================================
// get environment variable
// ==========================================
namespace {
static inline void get_env(unsigned &num_moduli, bool &fastmode) {
    const char *nm = getenv("NUM_MODULI");
    const char *fm = getenv("FASTMODE");
    num_moduli     = (nm ? std::stoi(nm) : 18);
    fastmode       = (fm ? std::string(fm) == std::string("1") : false);
}
} // namespace

// ==========================================
// thread_local cache
// ==========================================
namespace {
thread_local void *cached_work  = nullptr;
thread_local size_t cached_size = 0;
} // namespace

namespace {
static void *get_work(size_t req_size) {
    if (req_size == 0) return nullptr;
    if (cached_size < req_size) {
        if (cached_work) cudaFree(cached_work);
        cudaMalloc(&cached_work, req_size);
        cached_size = req_size;
    }
    return cached_work;
}

__attribute__((destructor)) static void cleanup_work() {
    if (cached_work) {
        cudaFree(cached_work);
        cached_work = nullptr;
        cached_size = 0;
    }
}
} // namespace

// =======================
// Hook: cublasSgemm_v2
// =======================
extern "C" cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m,
                                         int n,
                                         int k,
                                         const float *alpha,
                                         const float *A,
                                         int lda,
                                         const float *B,
                                         int ldb,
                                         const float *beta,
                                         float *C,
                                         int ldc) //
{
#ifdef __CUDA_ARCH__

    return CUBLAS_STATUS_NOT_SUPPORTED;

#else

    if (m == 0 || n == 0 || k == 0) return CUBLAS_STATUS_SUCCESS;

    unsigned num_moduli;
    bool fastmode;
    get_env(num_moduli, fastmode);

    size_t wsize = gemmul8::workSize(m, n, k, num_moduli);
    // void *work   = nullptr;
    // if (wsize > 0) cudaMalloc(&work, wsize);
    void *work = get_work(wsize);

    float one_f = 1.0f, zero_f = 0.0f;
    const float *alpha_f = alpha ? alpha : &one_f;
    const float *beta_f  = beta ? beta : &zero_f;
    gemmul8::gemm<float>(handle,
                         transa,
                         transb,
                         m,
                         n,
                         k,
                         alpha_f,
                         A,
                         lda,
                         B,
                         ldb,
                         beta_f,
                         C,
                         ldc,
                         num_moduli,
                         fastmode,
                         work);

    // if (work) cudaFree(work);
    return CUBLAS_STATUS_SUCCESS;

#endif
}

// =======================
// Hook: cublasDgemm_v2
// =======================
extern "C" cublasStatus_t cublasDgemm_v2(cublasHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m,
                                         int n,
                                         int k,
                                         const double *alpha,
                                         const double *A,
                                         int lda,
                                         const double *B,
                                         int ldb,
                                         const double *beta,
                                         double *C,
                                         int ldc) //
{
#ifdef __CUDA_ARCH__

    return CUBLAS_STATUS_NOT_SUPPORTED;

#else

    if (m == 0 || n == 0 || k == 0) return CUBLAS_STATUS_SUCCESS;

    unsigned num_moduli;
    bool fastmode;
    get_env(num_moduli, fastmode);

    size_t wsize = gemmul8::workSize(m, n, k, num_moduli);
    // void *work   = nullptr;
    // if (wsize > 0) cudaMalloc(&work, wsize);
    void *work = get_work(wsize);

    double one_d = 1.0, zero_d = 0.0;
    const double *alpha_d = alpha ? alpha : &one_d;
    const double *beta_d  = beta ? beta : &zero_d;
    gemmul8::gemm<double>(handle,
                          transa,
                          transb,
                          m,
                          n,
                          k,
                          alpha_d,
                          A,
                          lda,
                          B,
                          ldb,
                          beta_d,
                          C,
                          ldc,
                          num_moduli,
                          fastmode,
                          work);

    // if (work) cudaFree(work);
    return CUBLAS_STATUS_SUCCESS;

#endif
}

// =======================
// Hook: cublasGemmEx
// =======================
extern "C" cublasStatus_t cublasGemmEx(cublasHandle_t handle,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       int m,
                                       int n,
                                       int k,
                                       const void *alpha,
                                       const void *A,
                                       cudaDataType Atype,
                                       int lda,
                                       const void *B,
                                       cudaDataType Btype,
                                       int ldb,
                                       const void *beta,
                                       void *C,
                                       cudaDataType Ctype,
                                       int ldc,
                                       cublasComputeType_t computeType,
                                       cublasGemmAlgo_t algo) //
{
#ifdef __CUDA_ARCH__

    return CUBLAS_STATUS_NOT_SUPPORTED;

#else

    if (m == 0 || n == 0 || k == 0) return CUBLAS_STATUS_SUCCESS;

    // SGEMM
    if (computeType == CUBLAS_COMPUTE_32F &&
        Atype == CUDA_R_32F &&
        Btype == CUDA_R_32F &&
        Ctype == CUDA_R_32F) {
        unsigned num_moduli;
        bool fastmode;
        get_env(num_moduli, fastmode);

        size_t wsize = gemmul8::workSize(m, n, k, num_moduli);
        // void *work   = nullptr;
        // if (wsize > 0) cudaMalloc(&work, wsize);
        void *work = get_work(wsize);

        float one_f = 1.0f, zero_f = 0.0f;
        const float *alpha_f = alpha ? static_cast<const float *>(alpha) : &one_f;
        const float *beta_f  = beta ? static_cast<const float *>(beta) : &zero_f;
        gemmul8::gemm<float>(handle,
                             transa,
                             transb,
                             m,
                             n,
                             k,
                             alpha_f,
                             reinterpret_cast<const float *>(A),
                             lda,
                             reinterpret_cast<const float *>(B),
                             ldb,
                             beta_f,
                             reinterpret_cast<float *>(C),
                             ldc,
                             num_moduli,
                             fastmode,
                             work);

        // if (work) cudaFree(work);
        return CUBLAS_STATUS_SUCCESS;
    }

    // DGEMM
    if (computeType == CUBLAS_COMPUTE_64F &&
        Atype == CUDA_R_64F &&
        Btype == CUDA_R_64F &&
        Ctype == CUDA_R_64F) {
        unsigned num_moduli;
        bool fastmode;
        get_env(num_moduli, fastmode);

        size_t wsize = gemmul8::workSize(m, n, k, num_moduli);
        // void *work   = nullptr;
        // if (wsize > 0) cudaMalloc(&work, wsize);
        void *work = get_work(wsize);

        double one_d = 1.0, zero_d = 0.0;
        const double *alpha_d = alpha ? static_cast<const double *>(alpha) : &one_d;
        const double *beta_d  = beta ? static_cast<const double *>(beta) : &zero_d;

        gemmul8::gemm<double>(handle,
                              transa,
                              transb,
                              m,
                              n,
                              k,
                              alpha_d,
                              reinterpret_cast<const double *>(A),
                              lda,
                              reinterpret_cast<const double *>(B),
                              ldb,
                              beta_d,
                              reinterpret_cast<double *>(C),
                              ldc,
                              num_moduli,
                              fastmode,
                              work);

        // if (work) cudaFree(work);
        return CUBLAS_STATUS_SUCCESS;
    }

    // otherwise
    using gemmEx_t = cublasStatus_t (*)(cublasHandle_t,
                                        cublasOperation_t,
                                        cublasOperation_t,
                                        int,
                                        int,
                                        int,
                                        const void *,
                                        const void *,
                                        cudaDataType,
                                        int,
                                        const void *,
                                        cudaDataType,
                                        int,
                                        const void *,
                                        void *,
                                        cudaDataType,
                                        int,
                                        cublasComputeType_t,
                                        cublasGemmAlgo_t);

    static gemmEx_t real_gemmEx = (gemmEx_t)dlsym(RTLD_NEXT, "cublasGemmEx");
    return real_gemmEx(handle,
                       transa,
                       transb,
                       m,
                       n,
                       k,
                       alpha,
                       A,
                       Atype,
                       lda,
                       B,
                       Btype,
                       ldb,
                       beta,
                       C,
                       Ctype,
                       ldc,
                       computeType,
                       algo);

#endif
}
