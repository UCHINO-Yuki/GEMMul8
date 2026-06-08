#pragma once

#include "../general/scaling_general_declaration.hpp"

namespace gemmul8::scaling::accu {

template <typename T, Backend BACKEND, unsigned NUM_MODULI,
          cublasFillMode_t UPLO, cublasDiagType_t DIAG, bool CONJ>
__global__ void scaling_colwise(
    const unsigned rows_A,
    const T *const __restrict__ A, const size_t lda,
    common::matptr_t<common::low_t<BACKEND>, common::isComplex<T>> A_lo,
    const size_t lda_lo, const size_t incA_lo,
    int16_t *const __restrict__ sftA //
) {
    const unsigned col_idx         = blockIdx.x;
    const T *const __restrict__ in = A + col_idx * lda;
    const int32_t sft              = -sftA[col_idx];

    general::scaling_colwise_device<T, BACKEND, NUM_MODULI, UPLO, DIAG, CONJ>(
        rows_A, in, A_lo, lda_lo, incA_lo, sft);
}

template <typename T, Backend BACKEND, unsigned NUM_MODULI,
          cublasFillMode_t UPLO, cublasDiagType_t DIAG, bool CONJ>
inline void scaling_colwise_launch(
    const cudaStream_t stream,
    const unsigned rows_A,
    const unsigned cols_A,
    const T *const A,
    const size_t lda,
    common::matptr_t<common::low_t<BACKEND>, common::isComplex<T>> A_lo,
    const size_t lda_lo4,
    const size_t incA_lo4,
    int16_t *const sftA //
) {
    constexpr dim3 threads(general::threads_x_colwise_full_tiled,
                           general::threads_y_colwise_full_tiled);

    if constexpr (UPLO == CUBLAS_FILL_MODE_FULL) {

        bool vector_load_aligned;
        if constexpr (common::isComplex<T>) {
            vector_load_aligned = false;
        } else {
            vector_load_aligned = common::is_aligned_as<T>(A) &&
                                  ((lda * sizeof(T)) % common::vec4_alignment_v<T> == 0ULL);
        }

        const bool aligned_full =
            ((rows_A & 3U) == 0U) &&
            (lda_lo4 == (size_t(rows_A) >> 2)) &&
            ((lda_lo4 % general::threads_x_colwise_full_tiled) == 0U) &&
            vector_load_aligned;

        if (aligned_full) {
            const dim3 grid(
                unsigned(lda_lo4 / general::threads_x_colwise_full_tiled),
                (cols_A + general::threads_y_colwise_full_tiled - 1U) / general::threads_y_colwise_full_tiled);

            general::scaling_colwise_full_tiled_aligned_kernel<T, BACKEND, NUM_MODULI, CONJ>
                <<<grid, threads, 0, stream>>>(
                    rows_A, cols_A, A, lda, A_lo, lda_lo4, incA_lo4, sftA);

        } else {
            const dim3 grid(
                (unsigned(lda_lo4) + general::threads_x_colwise_full_tiled - 1U) / general::threads_x_colwise_full_tiled,
                (cols_A + general::threads_y_colwise_full_tiled - 1U) / general::threads_y_colwise_full_tiled);

            general::scaling_colwise_full_tiled_kernel<T, BACKEND, NUM_MODULI, CONJ>
                <<<grid, threads, 0, stream>>>(
                    rows_A, cols_A, A, lda, A_lo, lda_lo4, incA_lo4, sftA);
        }

    } else {

        const unsigned rows4 = (rows_A + 3U) >> 2;

        unsigned cols_grid = cols_A;
        if constexpr (UPLO == CUBLAS_FILL_MODE_LOWER) {
            cols_grid = (cols_A < rows_A) ? cols_A : rows_A;
        }

        if (rows4 == 0U || cols_grid == 0U) return;

        const dim3 grid(
            (rows4 + general::threads_x_colwise_full_tiled - 1U) / general::threads_x_colwise_full_tiled,
            (cols_grid + general::threads_y_colwise_full_tiled - 1U) / general::threads_y_colwise_full_tiled);

        general::scaling_colwise_tri_tiled_kernel<T, BACKEND, NUM_MODULI, UPLO, DIAG, CONJ>
            <<<grid, threads, 0, stream>>>(
                rows_A, cols_A, A, lda, A_lo, lda_lo4, incA_lo4, sftA);
    }
}

} // namespace gemmul8::scaling::accu
