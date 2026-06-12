#pragma once
#include "../common/common.hpp"
#include "../core/oz2_core.hpp"
#include "worksize.hpp"

namespace gemmul8::oz2::syrk {

template <typename TA, typename TC, Backend BACKEND, unsigned NUM_MODULI>
std::vector<double> syrk_core(
    common::Handle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans,
    size_t n, size_t k,
    const TC *alpha,
    const TA *const A, size_t lda,
    const TC *beta,
    TC *const C, size_t ldc,
    bool fastmode,
    void *const work, void *const workA,
    bool enable_skip_scalA,
    bool skip_scalA,
    cudaStream_t stream //
) {
    static_assert(common::isComplex<TA> == common::isComplex<TC>,
                  "TA and TC must be both real or both complex");

    if (uplo == CUBLAS_FILL_MODE_FULL || trans == CUBLAS_OP_C) {
        assert(false && "unsupported");
        return std::vector<double>(4, 0.0);
    }

    const cublasOperation_t op_A = trans;

    // Return workspace size
    if (work == nullptr) {
        size_t workSize_A;
        size_t workSize_total = workSize<common::isComplex<TA>, BACKEND>(
            n, n, k, NUM_MODULI, enable_skip_scalA, false, &workSize_A, nullptr);
        std::vector<double> timer(4, 0.0);
        timer[0] = static_cast<double>(workSize_total);
        timer[1] = static_cast<double>(workSize_A);
        timer[2] = 0.0;
        return timer;
    }

    if (uplo == CUBLAS_FILL_MODE_UPPER) {
        constexpr cublasFillMode_t UPLO_C = CUBLAS_FILL_MODE_UPPER;
        return core::oz2_core_rk<Func::syrk, TA, TC, BACKEND, NUM_MODULI, TC, TC, UPLO_C>(
            handle, op_A, n, k,
            alpha, A, lda, beta, C, ldc,
            fastmode, work, workA,
            enable_skip_scalA, skip_scalA,
            stream);
    } else {
        constexpr cublasFillMode_t UPLO_C = CUBLAS_FILL_MODE_LOWER;
        return core::oz2_core_rk<Func::syrk, TA, TC, BACKEND, NUM_MODULI, TC, TC, UPLO_C>(
            handle, op_A, n, k,
            alpha, A, lda, beta, C, ldc,
            fastmode, work, workA,
            enable_skip_scalA, skip_scalA,
            stream);
    }
}

} // namespace gemmul8::oz2::syrk
