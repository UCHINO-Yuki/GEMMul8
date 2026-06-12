#pragma once
#include "../common/common.hpp"
#include "../core/oz2_core.hpp"
#include "worksize.hpp"

namespace gemmul8::oz2::her2k {

template <typename TA, typename TB, typename TC, Backend BACKEND, unsigned NUM_MODULI>
std::vector<double> her2k_core(
    common::Handle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans,
    size_t n, size_t k,
    const TC *alpha,
    const TA *const A, size_t lda,
    const TB *const B, size_t ldb,
    const common::underlying_t<TC> *beta,
    TC *const C, size_t ldc,
    bool fastmode,
    void *const work, void *const workA, void *const workB,
    bool enable_skip_scalA, bool enable_skip_scalB,
    bool skip_scalA, bool skip_scalB,
    cudaStream_t stream //
) {
    static_assert(common::isComplex<TA> == common::isComplex<TB> &&
                      common::isComplex<TB> == common::isComplex<TC>,
                  "TA, TB, and TC must be all real or all complex");
    static_assert(common::isComplex<TA>, "her2k requires complex input type.");

    if (uplo == CUBLAS_FILL_MODE_FULL || trans == CUBLAS_OP_T) {
        assert(false && "unsupported");
        return std::vector<double>(4, 0.0);
    }

    using TBeta                  = common::underlying_t<TC>;
    const size_t m               = n;
    const cublasOperation_t op_A = trans;
    const cublasOperation_t op_B = (trans == CUBLAS_OP_N) ? CUBLAS_OP_C : CUBLAS_OP_N;

    constexpr common::MatStruct STRUCT_A = common::MatStruct::Full;
    constexpr common::MatStruct STRUCT_B = common::MatStruct::Full;
    constexpr cublasFillMode_t UPLO_A    = CUBLAS_FILL_MODE_FULL;
    constexpr cublasFillMode_t UPLO_B    = CUBLAS_FILL_MODE_FULL;
    constexpr cublasDiagType_t DIAG_A    = CUBLAS_DIAG_NON_UNIT;
    constexpr cublasDiagType_t DIAG_B    = CUBLAS_DIAG_NON_UNIT;

    if (work == nullptr) {
        size_t workSize_A, workSize_B;
        size_t workSize_total = workSize<common::isComplex<TA>, BACKEND>(
            m, n, k, NUM_MODULI, enable_skip_scalA, enable_skip_scalB, &workSize_A, &workSize_B);
        std::vector<double> timer(4, 0.0);
        timer[0] = static_cast<double>(workSize_total);
        timer[1] = static_cast<double>(workSize_A);
        timer[2] = static_cast<double>(workSize_B);
        return timer;
    }

    if (uplo == CUBLAS_FILL_MODE_UPPER) {
        constexpr cublasFillMode_t UPLO_C = CUBLAS_FILL_MODE_UPPER;
        return core::oz2_core<Func::her2k, TA, TB, TC, BACKEND, NUM_MODULI, TC, TBeta,
                              STRUCT_A, STRUCT_B, UPLO_A, UPLO_B, DIAG_A, DIAG_B, UPLO_C>(
            handle, op_A, op_B, m, n, k,
            alpha, A, lda, B, ldb, beta, C, ldc,
            fastmode, work, workA, workB,
            enable_skip_scalA, enable_skip_scalB,
            skip_scalA, skip_scalB, stream);
    } else {
        constexpr cublasFillMode_t UPLO_C = CUBLAS_FILL_MODE_LOWER;
        return core::oz2_core<Func::her2k, TA, TB, TC, BACKEND, NUM_MODULI, TC, TBeta,
                              STRUCT_A, STRUCT_B, UPLO_A, UPLO_B, DIAG_A, DIAG_B, UPLO_C>(
            handle, op_A, op_B, m, n, k,
            alpha, A, lda, B, ldb, beta, C, ldc,
            fastmode, work, workA, workB,
            enable_skip_scalA, enable_skip_scalB,
            skip_scalA, skip_scalB, stream);
    }
}

} // namespace gemmul8::oz2::her2k
