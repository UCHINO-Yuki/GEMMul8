#include "../include/gemmul8.hpp" // gemmul8::gemm, gemmul8::workSize
#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <unordered_map>

// NOTE: skip_scalA/B rely on pointer identity; actual A/B contents are not verified.
//       Use GEMMUL8_SKIP_SCALE_* only when A/B data are unchanged.

namespace {

namespace initval {
inline constexpr size_t MAX_M         = 1u;
inline constexpr size_t MAX_N         = 1u;
inline constexpr size_t MAX_K         = 1u;
inline constexpr unsigned MAX_NUM_MOD = 2u;
inline constexpr unsigned NUM_MOD_D   = 2u;
inline constexpr unsigned NUM_MOD_S   = 2u;
} // namespace initval

struct Info_t {
    unsigned num_moduli    = 2;
    cublasOperation_t op_A = CUBLAS_OP_N;
    cublasOperation_t op_B = CUBLAS_OP_N;
    size_t m               = 0;
    size_t n               = 0;
    size_t k               = 0;
    size_t lda             = 0;
    size_t ldb             = 0;
    const void *A          = nullptr;
    const void *B          = nullptr;
    int8_t *workA          = nullptr;
    int8_t *workB          = nullptr;
    bool is_double         = true;
    bool fastmode          = false;
};
static std::unordered_map<cublasHandle_t, Info_t> last_info;
static std::unordered_map<cublasHandle_t, void *> work_cache;
static std::unordered_map<cublasHandle_t, size_t> work_size;
static size_t max_workSize    = 0; // in byte
static size_t max_workSizeA   = 0; // in byte
static size_t max_workSizeB   = 0; // in byte
static bool skip_scalA_switch = false;
static bool skip_scalB_switch = false;

static void init_max_workspace() {
    if (max_workSize != 0) return; // already initialized

    size_t max_m        = initval::MAX_M;
    size_t max_n        = initval::MAX_N;
    size_t max_k        = initval::MAX_K;
    unsigned max_moduli = initval::MAX_NUM_MOD;

    const char *sm   = getenv("GEMMUL8_MAX_M");
    const char *sn   = getenv("GEMMUL8_MAX_N");
    const char *sk   = getenv("GEMMUL8_MAX_K");
    const char *smod = getenv("GEMMUL8_MAX_NUM_MOD");

    if (sm) {
        try {
            max_m = std::stoull(sm);
        } catch (...) {}
    }
    if (sn) {
        try {
            max_n = std::stoull(sn);
        } catch (...) {}
    }
    if (sk) {
        try {
            max_k = std::stoull(sk);
        } catch (...) {}
    }
    if (smod) {
        try {
            max_moduli = std::stoul(smod);
        } catch (...) {}
    }

    max_workSize = gemmul8::workSize(max_m, max_n, max_k, max_moduli, true, true, &max_workSizeA, &max_workSizeB);
}

static inline void get_env_d(unsigned &num_moduli, bool &fastmode) {
    const char *skipA = getenv("GEMMUL8_SKIP_SCALE_A");
    const char *skipB = getenv("GEMMUL8_SKIP_SCALE_B");
    const char *nm    = getenv("GEMMUL8_NUM_MOD_D");
    const char *fm    = getenv("GEMMUL8_FASTMODE_D");
    num_moduli        = initval::NUM_MOD_D;
    if (nm) {
        try {
            num_moduli = std::stoul(nm);
        } catch (...) {}
    }
    fastmode          = (fm ? std::string(fm) == std::string("1") : false);
    skip_scalA_switch = (skipA ? std::string(skipA) == std::string("1") : false);
    skip_scalB_switch = (skipB ? std::string(skipB) == std::string("1") : false);
}

static inline void get_env_s(unsigned &num_moduli, bool &fastmode) {
    const char *skipA = getenv("GEMMUL8_SKIP_SCALE_A");
    const char *skipB = getenv("GEMMUL8_SKIP_SCALE_B");
    const char *nm    = getenv("GEMMUL8_NUM_MOD_S");
    const char *fm    = getenv("GEMMUL8_FASTMODE_S");
    num_moduli        = initval::NUM_MOD_S;
    if (nm) {
        try {
            num_moduli = std::stoul(nm);
        } catch (...) {}
    }
    fastmode          = (fm ? std::string(fm) == std::string("1") : false);
    skip_scalA_switch = (skipA ? std::string(skipA) == std::string("1") : false);
    skip_scalB_switch = (skipB ? std::string(skipB) == std::string("1") : false);
}

static cublasStatus_t get_work(cublasHandle_t handle, size_t req_size, void **ptr) {
    if (!ptr) return CUBLAS_STATUS_INVALID_VALUE;
    init_max_workspace();

    if (req_size == 0) {
        *ptr = nullptr;
        return CUBLAS_STATUS_SUCCESS;
    }

    void *&current_cache = work_cache[handle];
    size_t &current_size = work_size[handle];
    req_size             = max(req_size, max_workSize);

    if (current_cache && current_size >= req_size) {
        *ptr = current_cache;
        return CUBLAS_STATUS_SUCCESS;
    }

    if (current_cache) {
        cudaError_t free_err = cudaFree(current_cache);
        if (free_err != cudaSuccess) {
            std::cerr << "[GEMMUL8 HOOK] Warning: cudaFree failed ("
                      << cudaGetErrorString(free_err) << ")" << std::endl;
        }
        current_cache = nullptr;
        current_size  = 0;
    }

    cudaError_t err = cudaMalloc(&current_cache, req_size);
    if (err != cudaSuccess) {
        current_cache = nullptr;
        current_size  = 0;
        *ptr          = nullptr;
        std::cerr << "[GEMMUL8 HOOK] Malloc failed for size " << req_size
                  << " bytes. Error: " << cudaGetErrorString(err) << std::endl;
        return CUBLAS_STATUS_ALLOC_FAILED;
    }

    current_size = req_size;
    *ptr         = current_cache;
    return CUBLAS_STATUS_SUCCESS;
}

static void cleanup_work(cublasHandle_t handle) {
    auto it = work_cache.find(handle);
    if (it != work_cache.end()) {
        cudaError_t free_err = cudaFree(it->second);
        if (free_err != cudaSuccess) {
            std::cerr << "[GEMMUL8 HOOK] cublasDestroy: Warning: cudaFree failed ("
                      << cudaGetErrorString(free_err) << ")" << std::endl;
        }
        work_cache.erase(it);
        work_size.erase(handle);
        last_info.erase(handle);
    }
}
} // namespace

// =======================
// Hook: cublasDestroy
// =======================
extern "C" cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
#ifdef __CUDA_ARCH__
    return CUBLAS_STATUS_NOT_SUPPORTED;
#else
    cleanup_work(handle);
    using cublasDestroy_t                     = cublasStatus_t (*)(cublasHandle_t);
    static cublasDestroy_t real_cublasDestroy = (cublasDestroy_t)dlsym(RTLD_NEXT, "cublasDestroy_v2");
    if (!real_cublasDestroy) return CUBLAS_STATUS_NOT_INITIALIZED;
    return real_cublasDestroy(handle);
#endif
}

// =======================
// Hook: cublasSgemm_v2
// =======================
extern "C" cublasStatus_t cublasSgemm_v2(
    cublasHandle_t handle,
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
    int ldc //
) {
#ifdef __CUDA_ARCH__
    return CUBLAS_STATUS_NOT_SUPPORTED;

#else
    if (m == 0 || n == 0 || k == 0) return CUBLAS_STATUS_SUCCESS;
    if (!A || !B || !C) return CUBLAS_STATUS_INVALID_VALUE;

    unsigned num_moduli;
    bool fastmode;
    get_env_s(num_moduli, fastmode);

    size_t wsizeA, wsizeB;
    size_t wsize          = gemmul8::workSize(m, n, k, num_moduli, skip_scalA_switch, skip_scalB_switch, &wsizeA, &wsizeB);
    void *work            = nullptr;
    cublasStatus_t status = get_work(handle, wsize, &work);
    if (status != CUBLAS_STATUS_SUCCESS) return status;

    int8_t *workA        = reinterpret_cast<int8_t *>(work);
    const size_t offsetA = max(max_workSizeA, wsizeA);
    const size_t offsetB = max(max_workSizeB, wsizeB);
    int8_t *workB        = workA + offsetA;
    int8_t *workC        = workB + offsetB;

    bool skip_scalA  = false; // false (unskip scaling_A) or true (skip scaling_A)
    bool skip_scalB  = false; // false (unskip scaling_B) or true (skip scaling_B)
    Info_t &info_pre = last_info[handle];
    if (info_pre.num_moduli == num_moduli && info_pre.k == k && !info_pre.is_double && info_pre.fastmode == fastmode) {
        if (skip_scalA_switch && info_pre.workA == workA && info_pre.A == A && info_pre.m == m && info_pre.lda == lda && info_pre.op_A == transa) {
            skip_scalA = skip_scalA_switch && true;
        }
        if (skip_scalB_switch && info_pre.workB == workB && info_pre.B == B && info_pre.n == n && info_pre.ldb == ldb && info_pre.op_B == transb) {
            skip_scalB = skip_scalB_switch && true;
        }
    }

    gemmul8::gemm<float>(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A,
        lda,
        B,
        ldb,
        beta,
        C,
        ldc,
        num_moduli,
        fastmode,
        reinterpret_cast<void *>(workC),
        reinterpret_cast<void *>(workA),
        reinterpret_cast<void *>(workB),
        skip_scalA_switch,
        skip_scalB_switch,
        skip_scalA,
        skip_scalB);

    info_pre.num_moduli = num_moduli;
    info_pre.op_A       = transa;
    info_pre.op_B       = transb;
    info_pre.m          = m;
    info_pre.n          = n;
    info_pre.k          = k;
    info_pre.lda        = lda;
    info_pre.ldb        = ldb;
    info_pre.A          = A;
    info_pre.B          = B;
    info_pre.workA      = workA;
    info_pre.workB      = workB;
    info_pre.is_double  = false;
    info_pre.fastmode   = fastmode;

    return CUBLAS_STATUS_SUCCESS;

#endif
}

// =======================
// Hook: cublasDgemm_v2
// =======================
extern "C" cublasStatus_t cublasDgemm_v2(
    cublasHandle_t handle,
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
    int ldc //
) {
#ifdef __CUDA_ARCH__
    return CUBLAS_STATUS_NOT_SUPPORTED;

#else
    if (m == 0 || n == 0 || k == 0) return CUBLAS_STATUS_SUCCESS;
    if (!A || !B || !C) return CUBLAS_STATUS_INVALID_VALUE;

    unsigned num_moduli;
    bool fastmode;
    get_env_d(num_moduli, fastmode);

    size_t wsizeA, wsizeB;
    size_t wsize          = gemmul8::workSize(m, n, k, num_moduli, skip_scalA_switch, skip_scalB_switch, &wsizeA, &wsizeB);
    void *work            = nullptr;
    cublasStatus_t status = get_work(handle, wsize, &work);
    if (status != CUBLAS_STATUS_SUCCESS) return status;

    int8_t *workA        = reinterpret_cast<int8_t *>(work);
    const size_t offsetA = max(max_workSizeA, wsizeA);
    const size_t offsetB = max(max_workSizeB, wsizeB);
    int8_t *workB        = workA + offsetA;
    int8_t *workC        = workB + offsetB;

    bool skip_scalA  = false; // false (unskip scaling_A) or true (skip scaling_A)
    bool skip_scalB  = false; // false (unskip scaling_B) or true (skip scaling_B)
    Info_t &info_pre = last_info[handle];
    if (info_pre.num_moduli == num_moduli && info_pre.k == k && info_pre.is_double && info_pre.fastmode == fastmode) {
        if (skip_scalA_switch && info_pre.workA == workA && info_pre.A == A && info_pre.m == m && info_pre.lda == lda && info_pre.op_A == transa) {
            skip_scalA = true;
        }
        if (skip_scalB_switch && info_pre.workB == workB && info_pre.B == B && info_pre.n == n && info_pre.ldb == ldb && info_pre.op_B == transb) {
            skip_scalB = true;
        }
    }

    gemmul8::gemm<double>(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A,
        lda,
        B,
        ldb,
        beta,
        C,
        ldc,
        num_moduli,
        fastmode,
        reinterpret_cast<void *>(workC),
        reinterpret_cast<void *>(workA),
        reinterpret_cast<void *>(workB),
        skip_scalA_switch,
        skip_scalB_switch,
        skip_scalA,
        skip_scalB);

    info_pre.num_moduli = num_moduli;
    info_pre.op_A       = transa;
    info_pre.op_B       = transb;
    info_pre.m          = m;
    info_pre.n          = n;
    info_pre.k          = k;
    info_pre.lda        = lda;
    info_pre.ldb        = ldb;
    info_pre.A          = A;
    info_pre.B          = B;
    info_pre.workA      = workA;
    info_pre.workB      = workB;
    info_pre.is_double  = true;
    info_pre.fastmode   = fastmode;

    return CUBLAS_STATUS_SUCCESS;

#endif
}

// =======================
// Hook: cublasGemmEx
// =======================
extern "C" cublasStatus_t cublasGemmEx(
    cublasHandle_t handle,
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
    cublasGemmAlgo_t algo //
) {
#ifdef __CUDA_ARCH__
    return CUBLAS_STATUS_NOT_SUPPORTED;

#else
    if (m == 0 || n == 0 || k == 0) return CUBLAS_STATUS_SUCCESS;
    if (!A || !B || !C) return CUBLAS_STATUS_INVALID_VALUE;

    // SGEMM
    if (computeType == CUBLAS_COMPUTE_32F &&
        Atype == CUDA_R_32F &&
        Btype == CUDA_R_32F &&
        Ctype == CUDA_R_32F) {
        unsigned num_moduli;
        bool fastmode;
        get_env_s(num_moduli, fastmode);

        size_t wsizeA, wsizeB;
        size_t wsize          = gemmul8::workSize(m, n, k, num_moduli, skip_scalA_switch, skip_scalB_switch, &wsizeA, &wsizeB);
        void *work            = nullptr;
        cublasStatus_t status = get_work(handle, wsize, &work);
        if (status != CUBLAS_STATUS_SUCCESS) return status;

        int8_t *workA        = reinterpret_cast<int8_t *>(work);
        const size_t offsetA = max(max_workSizeA, wsizeA);
        const size_t offsetB = max(max_workSizeB, wsizeB);
        int8_t *workB        = workA + offsetA;
        int8_t *workC        = workB + offsetB;

        bool skip_scalA  = false; // false (unskip scaling_A) or true (skip scaling_A)
        bool skip_scalB  = false; // false (unskip scaling_B) or true (skip scaling_B)
        Info_t &info_pre = last_info[handle];
        if (info_pre.num_moduli == num_moduli && info_pre.k == k && !info_pre.is_double && info_pre.fastmode == fastmode) {
            if (skip_scalA_switch && info_pre.workA == workA && info_pre.A == A && info_pre.m == m && info_pre.lda == lda && info_pre.op_A == transa) {
                skip_scalA = true;
            }
            if (skip_scalB_switch && info_pre.workB == workB && info_pre.B == B && info_pre.n == n && info_pre.ldb == ldb && info_pre.op_B == transb) {
                skip_scalB = true;
            }
        }

        gemmul8::gemm<float>(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            reinterpret_cast<const float *>(alpha),
            reinterpret_cast<const float *>(A),
            lda,
            reinterpret_cast<const float *>(B),
            ldb,
            reinterpret_cast<const float *>(beta),
            reinterpret_cast<float *>(C),
            ldc,
            num_moduli,
            fastmode,
            reinterpret_cast<void *>(workC),
            reinterpret_cast<void *>(workA),
            reinterpret_cast<void *>(workB),
            skip_scalA_switch,
            skip_scalB_switch,
            skip_scalA,
            skip_scalB);

        info_pre.num_moduli = num_moduli;
        info_pre.op_A       = transa;
        info_pre.op_B       = transb;
        info_pre.m          = m;
        info_pre.n          = n;
        info_pre.k          = k;
        info_pre.lda        = lda;
        info_pre.ldb        = ldb;
        info_pre.A          = A;
        info_pre.B          = B;
        info_pre.workA      = workA;
        info_pre.workB      = workB;
        info_pre.is_double  = false;
        info_pre.fastmode   = fastmode;

        return CUBLAS_STATUS_SUCCESS;
    }

    // DGEMM
    if (computeType == CUBLAS_COMPUTE_64F &&
        Atype == CUDA_R_64F &&
        Btype == CUDA_R_64F &&
        Ctype == CUDA_R_64F) {
        unsigned num_moduli;
        bool fastmode;
        get_env_d(num_moduli, fastmode);

        size_t wsizeA, wsizeB;
        size_t wsize          = gemmul8::workSize(m, n, k, num_moduli, skip_scalA_switch, skip_scalB_switch, &wsizeA, &wsizeB);
        void *work            = nullptr;
        cublasStatus_t status = get_work(handle, wsize, &work);
        if (status != CUBLAS_STATUS_SUCCESS) return status;

        int8_t *workA        = reinterpret_cast<int8_t *>(work);
        const size_t offsetA = max(max_workSizeA, wsizeA);
        const size_t offsetB = max(max_workSizeB, wsizeB);
        int8_t *workB        = workA + offsetA;
        int8_t *workC        = workB + offsetB;

        bool skip_scalA  = false; // false (unskip scaling_A) or true (skip scaling_A)
        bool skip_scalB  = false; // false (unskip scaling_B) or true (skip scaling_B)
        Info_t &info_pre = last_info[handle];
        if (info_pre.num_moduli == num_moduli && info_pre.k == k && info_pre.is_double && info_pre.fastmode == fastmode) {
            if (skip_scalA_switch && info_pre.workA == workA && info_pre.A == A && info_pre.m == m && info_pre.lda == lda && info_pre.op_A == transa) {
                skip_scalA = true;
            }
            if (skip_scalB_switch && info_pre.workB == workB && info_pre.B == B && info_pre.n == n && info_pre.ldb == ldb && info_pre.op_B == transb) {
                skip_scalB = true;
            }
        }

        gemmul8::gemm<double>(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            reinterpret_cast<const double *>(alpha),
            reinterpret_cast<const double *>(A),
            lda,
            reinterpret_cast<const double *>(B),
            ldb,
            reinterpret_cast<const double *>(beta),
            reinterpret_cast<double *>(C),
            ldc,
            num_moduli,
            fastmode,
            reinterpret_cast<void *>(workC),
            reinterpret_cast<void *>(workA),
            reinterpret_cast<void *>(workB),
            skip_scalA_switch,
            skip_scalB_switch,
            skip_scalA,
            skip_scalB);

        info_pre.num_moduli = num_moduli;
        info_pre.op_A       = transa;
        info_pre.op_B       = transb;
        info_pre.m          = m;
        info_pre.n          = n;
        info_pre.k          = k;
        info_pre.lda        = lda;
        info_pre.ldb        = ldb;
        info_pre.A          = A;
        info_pre.B          = B;
        info_pre.workA      = workA;
        info_pre.workB      = workB;
        info_pre.is_double  = true;
        info_pre.fastmode   = fastmode;

        return CUBLAS_STATUS_SUCCESS;
    }

    // otherwise
    using gemmEx_t = cublasStatus_t (*)(
        cublasHandle_t,
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
    if (!real_gemmEx) return CUBLAS_STATUS_NOT_INITIALIZED;
    return real_gemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);

#endif
}
