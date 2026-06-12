#pragma once
#include "../common/common.hpp"
#include "../core/oz2_core.hpp"
#include "worksize.hpp"

namespace gemmul8::oz2::trtrmm {

template <typename TA, typename TB, typename TC, Backend BACKEND, unsigned NUM_MODULI>
std::vector<double> trtrmm_core(
    common::Handle_t handle,
    cublasFillMode_t uplo_A, cublasFillMode_t uplo_B,
    cublasOperation_t trans_A, cublasOperation_t trans_B,
    cublasDiagType_t diag_A, cublasDiagType_t diag_B,
    size_t n,
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

    if (uplo_A == CUBLAS_FILL_MODE_FULL || uplo_B == CUBLAS_FILL_MODE_FULL) {
        assert(false && "unsupported");
        return std::vector<double>(4, 0.0);
    }

    const size_t m = n;
    const size_t k = n;

    constexpr common::MatStruct STRUCT_A = common::MatStruct::Triangular;
    constexpr common::MatStruct STRUCT_B = common::MatStruct::Triangular;
    constexpr cublasFillMode_t UPLO_C    = CUBLAS_FILL_MODE_FULL;

    if (work == nullptr) {
        size_t workSize_A, workSize_B;
        const size_t workSize_total = workSize<common::isComplex<TA>, BACKEND>(
            m, n, k, NUM_MODULI, enable_skip_scalA, enable_skip_scalB, &workSize_A, &workSize_B);

        std::vector<double> timer(4, 0.0);
        timer[0] = static_cast<double>(workSize_total);
        timer[1] = static_cast<double>(workSize_A);
        timer[2] = static_cast<double>(workSize_B);
        return timer;
    }

#define GEMMUL8_TRTRMM_CALL(UA, UB, DA, DB)                                      \
    return core::oz2_core<Func::trtrmm, TA, TB, TC, BACKEND, NUM_MODULI, TC, TC, \
                          STRUCT_A, STRUCT_B, UA, UB, DA, DB, UPLO_C>(           \
        handle, trans_A, trans_B, m, n, k,                                       \
        alpha, A, lda, B, ldb, beta, C, ldc,                                     \
        fastmode, work, workA, workB,                                            \
        enable_skip_scalA, enable_skip_scalB,                                    \
        skip_scalA, skip_scalB, stream)

    if (uplo_A == CUBLAS_FILL_MODE_UPPER) {
        constexpr cublasFillMode_t UA = CUBLAS_FILL_MODE_UPPER;

        if (uplo_B == CUBLAS_FILL_MODE_UPPER) {
            constexpr cublasFillMode_t UB = CUBLAS_FILL_MODE_UPPER;

            if (diag_A == CUBLAS_DIAG_NON_UNIT) {
                constexpr cublasDiagType_t DA = CUBLAS_DIAG_NON_UNIT;

                if (diag_B == CUBLAS_DIAG_NON_UNIT) {
                    constexpr cublasDiagType_t DB = CUBLAS_DIAG_NON_UNIT;
                    GEMMUL8_TRTRMM_CALL(UA, UB, DA, DB);
                } else {
                    constexpr cublasDiagType_t DB = CUBLAS_DIAG_UNIT;
                    GEMMUL8_TRTRMM_CALL(UA, UB, DA, DB);
                }
            } else {
                constexpr cublasDiagType_t DA = CUBLAS_DIAG_UNIT;

                if (diag_B == CUBLAS_DIAG_NON_UNIT) {
                    constexpr cublasDiagType_t DB = CUBLAS_DIAG_NON_UNIT;
                    GEMMUL8_TRTRMM_CALL(UA, UB, DA, DB);
                } else {
                    constexpr cublasDiagType_t DB = CUBLAS_DIAG_UNIT;
                    GEMMUL8_TRTRMM_CALL(UA, UB, DA, DB);
                }
            }

        } else {
            constexpr cublasFillMode_t UB = CUBLAS_FILL_MODE_LOWER;

            if (diag_A == CUBLAS_DIAG_NON_UNIT) {
                constexpr cublasDiagType_t DA = CUBLAS_DIAG_NON_UNIT;

                if (diag_B == CUBLAS_DIAG_NON_UNIT) {
                    constexpr cublasDiagType_t DB = CUBLAS_DIAG_NON_UNIT;
                    GEMMUL8_TRTRMM_CALL(UA, UB, DA, DB);
                } else {
                    constexpr cublasDiagType_t DB = CUBLAS_DIAG_UNIT;
                    GEMMUL8_TRTRMM_CALL(UA, UB, DA, DB);
                }
            } else {
                constexpr cublasDiagType_t DA = CUBLAS_DIAG_UNIT;

                if (diag_B == CUBLAS_DIAG_NON_UNIT) {
                    constexpr cublasDiagType_t DB = CUBLAS_DIAG_NON_UNIT;
                    GEMMUL8_TRTRMM_CALL(UA, UB, DA, DB);
                } else {
                    constexpr cublasDiagType_t DB = CUBLAS_DIAG_UNIT;
                    GEMMUL8_TRTRMM_CALL(UA, UB, DA, DB);
                }
            }
        }

    } else {
        constexpr cublasFillMode_t UA = CUBLAS_FILL_MODE_LOWER;

        if (uplo_B == CUBLAS_FILL_MODE_UPPER) {
            constexpr cublasFillMode_t UB = CUBLAS_FILL_MODE_UPPER;

            if (diag_A == CUBLAS_DIAG_NON_UNIT) {
                constexpr cublasDiagType_t DA = CUBLAS_DIAG_NON_UNIT;

                if (diag_B == CUBLAS_DIAG_NON_UNIT) {
                    constexpr cublasDiagType_t DB = CUBLAS_DIAG_NON_UNIT;
                    GEMMUL8_TRTRMM_CALL(UA, UB, DA, DB);
                } else {
                    constexpr cublasDiagType_t DB = CUBLAS_DIAG_UNIT;
                    GEMMUL8_TRTRMM_CALL(UA, UB, DA, DB);
                }
            } else {
                constexpr cublasDiagType_t DA = CUBLAS_DIAG_UNIT;

                if (diag_B == CUBLAS_DIAG_NON_UNIT) {
                    constexpr cublasDiagType_t DB = CUBLAS_DIAG_NON_UNIT;
                    GEMMUL8_TRTRMM_CALL(UA, UB, DA, DB);
                } else {
                    constexpr cublasDiagType_t DB = CUBLAS_DIAG_UNIT;
                    GEMMUL8_TRTRMM_CALL(UA, UB, DA, DB);
                }
            }

        } else {
            constexpr cublasFillMode_t UB = CUBLAS_FILL_MODE_LOWER;

            if (diag_A == CUBLAS_DIAG_NON_UNIT) {
                constexpr cublasDiagType_t DA = CUBLAS_DIAG_NON_UNIT;

                if (diag_B == CUBLAS_DIAG_NON_UNIT) {
                    constexpr cublasDiagType_t DB = CUBLAS_DIAG_NON_UNIT;
                    GEMMUL8_TRTRMM_CALL(UA, UB, DA, DB);
                } else {
                    constexpr cublasDiagType_t DB = CUBLAS_DIAG_UNIT;
                    GEMMUL8_TRTRMM_CALL(UA, UB, DA, DB);
                }
            } else {
                constexpr cublasDiagType_t DA = CUBLAS_DIAG_UNIT;

                if (diag_B == CUBLAS_DIAG_NON_UNIT) {
                    constexpr cublasDiagType_t DB = CUBLAS_DIAG_NON_UNIT;
                    GEMMUL8_TRTRMM_CALL(UA, UB, DA, DB);
                } else {
                    constexpr cublasDiagType_t DB = CUBLAS_DIAG_UNIT;
                    GEMMUL8_TRTRMM_CALL(UA, UB, DA, DB);
                }
            }
        }
    }

#undef GEMMUL8_TRTRMM_CALL
}

} // namespace gemmul8::oz2::trtrmm
