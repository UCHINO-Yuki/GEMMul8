#pragma once
#include "../common/common.hpp"
#include "../core/oz2_core.hpp"
#include "worksize.hpp"

namespace gemmul8::oz2::hemm {

template <typename THerm, typename TFull, typename TC, Backend BACKEND, unsigned NUM_MODULI>
std::vector<double> hemm_core(
    common::Handle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    size_t rowsC, size_t colsC,
    const TC *alpha,
    const THerm *const Herm, size_t ldHerm,
    const TFull *const Full, size_t ldFull,
    const TC *beta,
    TC *const C, size_t ldc,
    bool fastmode,
    void *const work, void *const workHerm, void *const workFull,
    bool enable_skip_scalHerm, bool enable_skip_scalFull,
    bool skip_scalHerm, bool skip_scalFull,
    cudaStream_t stream //
) {
    static_assert(common::isComplex<THerm> == common::isComplex<TFull> &&
                      common::isComplex<TFull> == common::isComplex<TC>,
                  "THerm, TFull, and TC must be all real or all complex");
    static_assert(common::isComplex<THerm>, "hemm requires complex input type.");

    if (uplo == CUBLAS_FILL_MODE_FULL) {
        assert(false && "unsupported");
        return std::vector<double>(4, 0.0);
    }

    if (side == CUBLAS_SIDE_LEFT) {
        // A: rowsC x rowsC, B: rowsC x colsC

        const size_t m = rowsC;
        const size_t n = colsC;
        const size_t k = rowsC;

        using TA                             = THerm;
        const TA *const A                    = Herm;
        const size_t lda                     = ldHerm;
        void *const workA                    = workHerm;
        const bool enable_skip_scalA         = enable_skip_scalHerm;
        const bool skip_scalA                = skip_scalHerm;
        constexpr cublasOperation_t op_A     = CUBLAS_OP_N;
        constexpr cublasDiagType_t DIAG_A    = CUBLAS_DIAG_NON_UNIT;
        constexpr common::MatStruct STRUCT_A = common::MatStruct::Hermitian;

        using TB                             = TFull;
        const TB *const B                    = Full;
        const size_t ldb                     = ldFull;
        void *const workB                    = workFull;
        const bool enable_skip_scalB         = enable_skip_scalFull;
        const bool skip_scalB                = skip_scalFull;
        constexpr cublasOperation_t op_B     = CUBLAS_OP_N;
        constexpr cublasFillMode_t UPLO_B    = CUBLAS_FILL_MODE_FULL;
        constexpr cublasDiagType_t DIAG_B    = CUBLAS_DIAG_NON_UNIT;
        constexpr common::MatStruct STRUCT_B = common::MatStruct::Full;

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
            constexpr cublasFillMode_t UPLO_A = CUBLAS_FILL_MODE_UPPER;
            return core::oz2_core<Func::hemm, TA, TB, TC, BACKEND, NUM_MODULI, TC, TC,
                                  STRUCT_A, STRUCT_B, UPLO_A, UPLO_B, DIAG_A, DIAG_B>(
                handle, op_A, op_B, m, n, k,
                alpha, A, lda, B, ldb, beta, C, ldc,
                fastmode, work, workA, workB,
                enable_skip_scalA, enable_skip_scalB,
                skip_scalA, skip_scalB, stream);
        } else {
            constexpr cublasFillMode_t UPLO_A = CUBLAS_FILL_MODE_LOWER;
            return core::oz2_core<Func::hemm, TA, TB, TC, BACKEND, NUM_MODULI, TC, TC,
                                  STRUCT_A, STRUCT_B, UPLO_A, UPLO_B, DIAG_A, DIAG_B>(
                handle, op_A, op_B, m, n, k,
                alpha, A, lda, B, ldb, beta, C, ldc,
                fastmode, work, workA, workB,
                enable_skip_scalA, enable_skip_scalB,
                skip_scalA, skip_scalB, stream);
        }

    } else {
        // A: rowsC x colsC, B: colsC x colsC

        const size_t m = rowsC;
        const size_t n = colsC;
        const size_t k = colsC;

        using TA                             = TFull;
        const TA *const A                    = Full;
        const size_t lda                     = ldFull;
        void *const workA                    = workFull;
        const bool enable_skip_scalA         = enable_skip_scalFull;
        const bool skip_scalA                = skip_scalFull;
        constexpr cublasOperation_t op_A     = CUBLAS_OP_N;
        constexpr cublasFillMode_t UPLO_A    = CUBLAS_FILL_MODE_FULL;
        constexpr cublasDiagType_t DIAG_A    = CUBLAS_DIAG_NON_UNIT;
        constexpr common::MatStruct STRUCT_A = common::MatStruct::Full;

        using TB                             = THerm;
        const TB *const B                    = Herm;
        const size_t ldb                     = ldHerm;
        void *const workB                    = workHerm;
        const bool enable_skip_scalB         = enable_skip_scalHerm;
        const bool skip_scalB                = skip_scalHerm;
        constexpr cublasOperation_t op_B     = CUBLAS_OP_N;
        constexpr cublasDiagType_t DIAG_B    = CUBLAS_DIAG_NON_UNIT;
        constexpr common::MatStruct STRUCT_B = common::MatStruct::Hermitian;

        if (work == nullptr) {
            size_t workSize_A, workSize_B;
            size_t workSize_total = workSize<common::isComplex<TA>, BACKEND>(
                m, n, k, NUM_MODULI, enable_skip_scalA, enable_skip_scalB, &workSize_A, &workSize_B);
            std::vector<double> timer(4, 0.0);
            timer[0] = static_cast<double>(workSize_total);
            timer[1] = static_cast<double>(workSize_B);
            timer[2] = static_cast<double>(workSize_A);
            return timer;
        }

        if (uplo == CUBLAS_FILL_MODE_UPPER) {
            constexpr cublasFillMode_t UPLO_B = CUBLAS_FILL_MODE_UPPER;
            return core::oz2_core<Func::hemm, TA, TB, TC, BACKEND, NUM_MODULI, TC, TC,
                                  STRUCT_A, STRUCT_B, UPLO_A, UPLO_B, DIAG_A, DIAG_B>(
                handle, op_A, op_B, m, n, k,
                alpha, A, lda, B, ldb, beta, C, ldc,
                fastmode, work, workA, workB,
                enable_skip_scalA, enable_skip_scalB,
                skip_scalA, skip_scalB, stream);
        } else {
            constexpr cublasFillMode_t UPLO_B = CUBLAS_FILL_MODE_LOWER;
            return core::oz2_core<Func::hemm, TA, TB, TC, BACKEND, NUM_MODULI, TC, TC,
                                  STRUCT_A, STRUCT_B, UPLO_A, UPLO_B, DIAG_A, DIAG_B>(
                handle, op_A, op_B, m, n, k,
                alpha, A, lda, B, ldb, beta, C, ldc,
                fastmode, work, workA, workB,
                enable_skip_scalA, enable_skip_scalB,
                skip_scalA, skip_scalB, stream);
        }
    }
}

} // namespace gemmul8::oz2::hemm
