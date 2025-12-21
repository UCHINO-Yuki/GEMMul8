#include "../include/gemmul8.hpp"
#include "gemmul8_complex.hpp"
#include "gemmul8_real.hpp"

#if !defined(GEMM_ARGS)
    #define GEMM_ARGS(T) cublasHandle_t handle,                                      \
                         const cublasOperation_t op_A, const cublasOperation_t op_B, \
                         const size_t m, const size_t n, const size_t k,             \
                         const T *alpha, const T *const A, const size_t lda,         \
                         const T *const B, const size_t ldb,                         \
                         const T *beta, T *const C, const size_t ldc,                \
                         const unsigned num_moduli, const bool fastmode,             \
                         void *const work, void *const workA, void *const workB,     \
                         const bool enable_skip_scalA, const bool enable_skip_scalB, \
                         const bool skip_scalA, const bool skip_scalB
#endif

#if !defined(GEMM_CALL_ARGS)
    #define GEMM_CALL_ARGS handle, op_A, op_B, m, n, k,              \
                           alpha, A, lda, B, ldb, beta, C, ldc,      \
                           num_moduli, fastmode, work, workA, workB, \
                           enable_skip_scalA, enable_skip_scalB,     \
                           skip_scalA, skip_scalB
#endif

namespace gemmul8 {

//------------------------------
// Calculate required work size
//------------------------------
template <> size_t workSize<true>(
    const size_t m, const size_t n, const size_t k,
    const unsigned num_moduli,
    const bool enable_skip_scalA, const bool enable_skip_scalB,
    size_t *workSizeA, size_t *workSizeB //
) {
    return oz2::complex::workSize(m, n, k, num_moduli, enable_skip_scalA, enable_skip_scalB, workSizeA, workSizeB);
}

template <> size_t workSize<false>(
    const size_t m, const size_t n, const size_t k,
    const unsigned num_moduli,
    const bool enable_skip_scalA, const bool enable_skip_scalB,
    size_t *workSizeA, size_t *workSizeB //
) {
    return oz2::real::workSize(m, n, k, num_moduli, enable_skip_scalA, enable_skip_scalB, workSizeA, workSizeB);
}

//------------------------------
// GEMM emulation using INT8 Tensor Cores
//------------------------------
template <> std::vector<double> gemm<double>(GEMM_ARGS(double)) { return oz2::real::gemm<double>(GEMM_CALL_ARGS); }
template <> std::vector<double> gemm<float>(GEMM_ARGS(float)) { return oz2::real::gemm<float>(GEMM_CALL_ARGS); }
template <> std::vector<double> gemm<cuFloatComplex>(GEMM_ARGS(cuFloatComplex)) { return oz2::complex::gemm<cuFloatComplex>(GEMM_CALL_ARGS); }
template <> std::vector<double> gemm<cuDoubleComplex>(GEMM_ARGS(cuDoubleComplex)) { return oz2::complex::gemm<cuDoubleComplex>(GEMM_CALL_ARGS); }

} // namespace gemmul8
