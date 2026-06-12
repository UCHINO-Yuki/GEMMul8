#pragma once
#include "../common/common.hpp"
#include "../core/oz2_core.hpp"
#include "worksize.hpp"

namespace gemmul8::oz2::gemm {

template <typename TA, typename TB, typename TC, Backend BACKEND, unsigned NUM_MODULI>
std::vector<double> gemm_core(
    common::Handle_t handle,
    cublasOperation_t op_A, cublasOperation_t op_B,
    size_t m, size_t n, size_t k,
    const TC *alpha,
    const TA *const A, size_t lda,
    const TB *const B, size_t ldb,
    const TC *beta,
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

    // Return workspace size
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

    return core::oz2_core<Func::gemm, TA, TB, TC, BACKEND, NUM_MODULI>(
        handle, op_A, op_B, m, n, k,
        alpha, A, lda, B, ldb, beta, C, ldc,
        fastmode, work, workA, workB,
        enable_skip_scalA, enable_skip_scalB,
        skip_scalA, skip_scalB, stream);
}

} // namespace gemmul8::oz2::gemm
