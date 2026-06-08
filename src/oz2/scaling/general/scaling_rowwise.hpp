#pragma once
#include "../../common/common.hpp"
#include "../../mod/mod.hpp"

#include "config.hpp"
#include "roundup.hpp"
#include "helper_triangular.hpp"
#include "store.hpp"

namespace gemmul8::scaling::general {

template <typename T, Backend BACKEND, unsigned NUM_MODULI, bool CONJ>
__global__ void scaling_rowwise_full_kernel(
    const unsigned rows_A, const unsigned cols_A,
    const T *const __restrict__ A, const size_t lda,
    common::matptr_t<common::low_t<BACKEND>, common::isComplex<T>> A_lo,
    const size_t lda_lo, const size_t incA_lo,
    int16_t *const __restrict__ sftA //
) {
    using ValT               = decltype(trunc_scalbn<true, T, BACKEND, NUM_MODULI>::run(T{}, int32_t{}));
    constexpr bool cast_flag = (sizeof(ValT) <= sizeof(T)) && common::isCUDA;
    using shm_t              = std::conditional_t<(cast_flag), ValT, T>;

    __shared__ shm_t tile[common::TILE_DIM][common::TILE_DIM + 1];

    const unsigned rowBase = blockIdx.x * common::TILE_DIM;
    const unsigned colBase = blockIdx.y * common::TILE_DIM;

    const unsigned in_row = rowBase + threadIdx.x;
    const int32_t sft     = (in_row < rows_A) ? -sftA[in_row] : 0;

#pragma unroll
    for (unsigned j = 0; j < common::TILE_DIM; j += threads_y_rowwise<BACKEND>) {
        const unsigned yy = threadIdx.y + j;
        if (yy >= common::TILE_DIM) continue;

        const unsigned in_col = colBase + yy;
        const T Atmp          = (in_row < rows_A && in_col < cols_A) ? A[in_col * lda + in_row] : common::Tconst<T>::zero();
        const shm_t A_scaled  = trunc_scalbn<cast_flag, T, BACKEND, NUM_MODULI>::run(common::conj<T, CONJ>(Atmp), sft);
        tile[yy][threadIdx.x] = A_scaled;
    }
    __syncthreads();

#pragma unroll
    for (unsigned j = 0; j < common::TILE_DIM; j += threads_y_rowwise<BACKEND>) {
        const unsigned yy = threadIdx.y + j;
        if (yy >= common::TILE_DIM) continue;

        const unsigned out_col = rowBase + yy;
        const unsigned out_row = colBase + threadIdx.x;
        if (out_col >= rows_A || out_row >= cols_A) continue;

        const size_t idx = out_col * lda_lo + out_row;
        const ValT in    = trunc_scalbn<cast_flag, T, BACKEND, NUM_MODULI>::cast(tile[threadIdx.x][yy]);

        if constexpr (common::isComplex<T>) {
            common::low_t<BACKEND> *__restrict__ out_1 = A_lo.ptr0 + idx;
            common::low_t<BACKEND> *__restrict__ out_2 = A_lo.ptr1 + idx;
            common::low_t<BACKEND> *__restrict__ out_3 = A_lo.ptr2 + idx;
            mod::ModUnroll<NUM_MODULI, ValT>::run(out_1, out_2, out_3, incA_lo, in);
        } else {
            common::low_t<BACKEND> *__restrict__ out = A_lo.ptr0 + idx;
            mod::ModUnroll<NUM_MODULI, ValT>::run(out, incA_lo, in);
        }
    }
}

template <bool UPPER,
          typename T, Backend BACKEND, unsigned NUM_MODULI,
          cublasDiagType_t DIAG, bool CONJ>
__global__ void scaling_rowwise_tri_kernel(
    const unsigned rows_A, const unsigned cols_A,
    const T *const __restrict__ A, const size_t lda,
    common::matptr_t<common::low_t<BACKEND>, common::isComplex<T>> A_lo,
    const size_t lda_lo, const size_t incA_lo,
    int16_t *const __restrict__ sftA //
) {
    using ValT               = decltype(trunc_scalbn<true, T, BACKEND, NUM_MODULI>::run(T{}, int32_t{}));
    constexpr bool cast_flag = (sizeof(ValT) <= sizeof(T)) && common::isCUDA;
    using shm_t              = std::conditional_t<(cast_flag), ValT, T>;

    __shared__ shm_t tile[common::TILE_DIM][common::TILE_DIM + 1];

    const unsigned rowBase = blockIdx.x * common::TILE_DIM;
    const unsigned colBase = blockIdx.y * common::TILE_DIM;
    if (tri_tile_zero<UPPER>(rowBase, colBase)) return;
    const bool full_active = tri_tile_full_active<UPPER>(rowBase, colBase);

    const unsigned in_row = rowBase + threadIdx.x;
    const int32_t sft     = (in_row < rows_A) ? -sftA[in_row] : 0;

#pragma unroll
    for (unsigned j = 0; j < common::TILE_DIM; j += threads_y_rowwise<BACKEND>) {
        const unsigned yy = threadIdx.y + j;
        if (yy >= common::TILE_DIM) continue;

        const unsigned in_col = colBase + yy;
        T Atmp;
        if (full_active) {
            Atmp = (in_row < rows_A && in_col < cols_A) ? A[in_col * lda + in_row] : common::Tconst<T>::zero();
        } else {
            Atmp = tri_mat_value<UPPER, T, DIAG>(A, lda, in_row, in_col, rows_A, cols_A);
        }
        const shm_t A_scaled  = trunc_scalbn<cast_flag, T, BACKEND, NUM_MODULI>::run(common::conj<T, CONJ>(Atmp), sft);
        tile[yy][threadIdx.x] = A_scaled;
    }
    __syncthreads();

#pragma unroll
    for (unsigned j = 0; j < common::TILE_DIM; j += threads_y_rowwise<BACKEND>) {
        const unsigned yy = threadIdx.y + j;
        if (yy >= common::TILE_DIM) continue;

        const unsigned out_col = rowBase + yy;
        const unsigned out_row = colBase + threadIdx.x;
        if (out_col >= rows_A || out_row >= cols_A) continue;
        if (!full_active) {
            if (!tri_elem_active<UPPER>(out_col, out_row)) continue;
        }

        const size_t idx = out_col * lda_lo + out_row;
        const ValT in    = trunc_scalbn<cast_flag, T, BACKEND, NUM_MODULI>::cast(tile[threadIdx.x][yy]);

        if constexpr (common::isComplex<T>) {
            common::low_t<BACKEND> *__restrict__ out_1 = A_lo.ptr0 + idx;
            common::low_t<BACKEND> *__restrict__ out_2 = A_lo.ptr1 + idx;
            common::low_t<BACKEND> *__restrict__ out_3 = A_lo.ptr2 + idx;
            mod::ModUnroll<NUM_MODULI, ValT>::run(out_1, out_2, out_3, incA_lo, in);
        } else {
            common::low_t<BACKEND> *__restrict__ out = A_lo.ptr0 + idx;
            mod::ModUnroll<NUM_MODULI, ValT>::run(out, incA_lo, in);
        }
    }
}

template <typename T, Backend BACKEND, unsigned NUM_MODULI,
          cublasFillMode_t UPLO, cublasDiagType_t DIAG, bool CONJ>
void scaling_rowwise(
    const cudaStream_t stream,
    const unsigned rows_A, const unsigned cols_A,
    const T *const A, const size_t lda,
    common::matptr_t<common::low_t<BACKEND>, common::isComplex<T>> A_lo,
    const size_t lda_lo, const size_t incA_lo,
    int16_t *const sftA //
) {

    constexpr dim3 threads(threads_x_general, threads_y_rowwise<BACKEND>);
    dim3 grid((rows_A + threads_x_general - 1) / threads_x_general,
              (cols_A + common::TILE_DIM - 1) / common::TILE_DIM);

    if constexpr (UPLO == CUBLAS_FILL_MODE_FULL) {

        memset_padding_low_mats_2d_async<T, BACKEND, NUM_MODULI>(
            stream, A_lo, cols_A, lda_lo, incA_lo / lda_lo);

        scaling_rowwise_full_kernel<T, BACKEND, NUM_MODULI, CONJ>
            <<<grid, threads, 0, stream>>>(
                rows_A, cols_A, A, lda, A_lo, lda_lo, incA_lo, sftA);

    } else {

        constexpr bool isUPPER = UPLO == CUBLAS_FILL_MODE_UPPER;
        scaling_rowwise_tri_kernel<isUPPER, T, BACKEND, NUM_MODULI, DIAG, CONJ>
            <<<grid, threads, 0, stream>>>(
                rows_A, cols_A, A, lda, A_lo, lda_lo, incA_lo, sftA);
    }
}

} // namespace gemmul8::scaling::general
