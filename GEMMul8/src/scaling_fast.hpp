#pragma once
#include "common.hpp"

namespace oz2 {
namespace vecnorm {

template <typename T> __forceinline__ __device__ int compute_sft(T amax, T vecnrm, const float log2M);
template <> __forceinline__ __device__ int compute_sft<double>(double amax, double vecnrm, const float log2M) {
    const int exponent  = Tilogb<double>(vecnrm);
    const float vecnrmf = __double2float_ru(scalbn(vecnrm, -exponent));
    const int k         = __float2int_rd(__fmaf_rd(-0x1.0000060000000p-1F, __fadd_ru(__log2f(vecnrmf), exponent), log2M));
    return min(__float2int_rd(log2M - 1.0f), k) - Tilogb<double>(amax);
}
template <> __forceinline__ __device__ int compute_sft<float>(float amax, float vecnrm, const float log2M) {
    const int k = __float2int_rd(__fmaf_rd(-0x1.0000060000000p-1F, __log2f(vecnrm), log2M));
    return min(__float2int_rd(log2M - 1.0f), k) - Tilogb<float>(amax);
}

template <typename T>
__global__ void compute_sft_kernel(
    const size_t m,                   // size(A,1)
    const size_t k,                   // size(A,2)
    const T *const A,                 // input (lda * k)
    const size_t lda,                 // leading dimension
    int16_t *const __restrict__ sftA, // exponent of shift values
    const float log2M                 // log2(M-1)/2 - 1.5
) {
    const auto block_base = blockIdx.x << LOG2_TILE_DIM;
    const auto row        = block_base + threadIdx.x;

    __shared__ T samax[TILE_DIM][TILE_DIM + 1];
    __shared__ T ssum[TILE_DIM][TILE_DIM + 1];

    T amax = Tzero<T>::value;
    T sum  = Tzero<T>::value;
    if (row < m) {
        const T *row_ptr = A + row;
        for (unsigned col = threadIdx.y; col < k; col += blockDim.y) {
            const T tmp = Tabs<T>(__ldg(row_ptr + col * lda));
            amax        = max(amax, tmp);
            sum         = __Tfma_ru<T>(tmp, tmp, sum); // round-up mode
        }
    }
    samax[threadIdx.y][threadIdx.x] = amax;
    ssum[threadIdx.y][threadIdx.x]  = sum;
    __syncthreads();

    amax = samax[threadIdx.x][threadIdx.y];
    amax = inner_warp_max<T, TILE_DIM>(amax);

    sum = ssum[threadIdx.x][threadIdx.y];
    sum = inner_warp_sum<T, TILE_DIM>(sum);

    if (row < m && threadIdx.x == 0) {
        const int sft                  = compute_sft<T>(amax, sum, log2M);
        sftA[block_base + threadIdx.y] = -sft;
    }
}

// A(i,j) -> A8i(j,i)
template <typename T, int MODE>
__global__ void scalingA_kernel(
    const size_t m,                  // size(A,1)
    const size_t k,                  // size(A,2)
    const size_t incA8i,             // lda8i * m
    const unsigned num_moduli,       // #moduli
    const T *const __restrict__ A,   // input (lda * k)
    const size_t lda,                // leading dimension
    int8_t *const __restrict__ A8i,  // output (lda8i * m)
    const size_t lda8i,              // leading dimension
    int16_t *const __restrict__ sftA // exponent of shift values
) {
    __shared__ T tile[TILE_DIM][TILE_DIM + 1];

    const auto rowBase = blockIdx.x << LOG2_TILE_DIM;
    const auto colBase = blockIdx.y << LOG2_TILE_DIM;

    if constexpr (USE_CHAR4) {

        const auto in_row = rowBase + threadIdx.x;
        if (in_row < m) {
            const int sft = -sftA[in_row];
            for (unsigned t = threadIdx.y; t < TILE_DIM; t += blockDim.y) {
                const auto in_col    = colBase + t;
                tile[t][threadIdx.x] = (in_col < k) ? T2int_fp<T>(__ldg(A + in_row + in_col * lda), sft) : (Tzero<T>::value);
            }
        }
        __syncthreads();

        const auto tile_row = (threadIdx.x & (CHAR4_PER_ROW - 1)) << 2;
        const auto tile_col = (threadIdx.x >> LOG2_CHAR4_PER_ROW) + (threadIdx.y << 2);
        const auto out_row  = colBase + tile_row;
        const auto out_col  = rowBase + tile_col;

        Vec4<T> in4;
        in4.x = tile[tile_row][tile_col];
        in4.y = tile[tile_row + 1][tile_col];
        in4.z = tile[tile_row + 2][tile_col];
        in4.w = tile[tile_row + 3][tile_col];

        int8_t *const __restrict__ out = A8i + out_row + out_col * lda8i;
        if (out_col < m) {
            for (unsigned j = 0; j < num_moduli; ++j) {
                const auto val = readtab<T>(j);

                char4 out4;
                out4.x = mod_8i<T, MODE>(in4.x, val);
                out4.y = mod_8i<T, MODE>(in4.y, val);
                out4.z = mod_8i<T, MODE>(in4.z, val);
                out4.w = mod_8i<T, MODE>(in4.w, val);

                *reinterpret_cast<char4 *>(out + j * incA8i) = out4;
            }
        }

    } else {

        const auto in_row = rowBase + threadIdx.x;
        const auto in_col = colBase + threadIdx.y;
        if (in_row < m) {
            const int sft                  = -sftA[in_row];
            tile[threadIdx.y][threadIdx.x] = (in_col < k) ? T2int_fp<T>(__ldg(A + in_row + in_col * lda), sft) : (Tzero<T>::value);
        }
        __syncthreads();

        const auto out_row = colBase + threadIdx.x;
        const auto out_col = rowBase + threadIdx.y;
        const T in         = tile[threadIdx.x][threadIdx.y];

        int8_t *const __restrict__ out = A8i + out_row + out_col * lda8i;
        if (out_col < m) {
            for (unsigned j = 0; j < num_moduli; ++j) {
                const auto val  = readtab<T>(j);
                out[j * incA8i] = mod_8i<T, MODE>(in, val);
            }
        }
    }
}

// convert trunc(B*diag(2^sftB)) to B8i
template <typename T, int MODE>
__global__ void scalingB_kernel(
    const size_t k,                   // size(B,1)
    const size_t incB8i,              // ldb8i * n
    const unsigned num_moduli,        // #moduli
    const T *const __restrict__ B,    // input (ldb * n)
    const size_t ldb,                 // leading dimension
    int8_t *const __restrict__ B8i,   // output (ldb8i * n)
    const size_t ldb8i,               // leading dimension
    int16_t *const __restrict__ sftB, // exponent of shift values
    const float log2M                 // log2(M-1)/2 - 1.5
) {
    __shared__ T shm[64];
    const auto col_idx             = blockIdx.x;
    const T *const __restrict__ in = B + col_idx * ldb;
    T vecnrm;
    const T amax  = find_amax_and_nrm<T>(in, k, shm, vecnrm);
    const int sft = compute_sft<T>(amax, vecnrm, log2M);
    if (threadIdx.x == 0) {
        sftB[col_idx] = -sft;
    }

    int8_t *const __restrict__ out = B8i + col_idx * ldb8i;
    unsigned kmax                  = k >> 2;
    unsigned i                     = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4;
        in4.x = T2int_fp<T>(in[idx], sft);
        in4.y = T2int_fp<T>(in[idx + 1], sft);
        in4.z = T2int_fp<T>(in[idx + 2], sft);
        in4.w = T2int_fp<T>(in[idx + 3], sft);

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {
            const auto val = readtab<T>(j);

            out4.x = mod_8i<T, MODE>(in4.x, val);
            out4.y = mod_8i<T, MODE>(in4.y, val);
            out4.z = mod_8i<T, MODE>(in4.z, val);
            out4.w = mod_8i<T, MODE>(in4.w, val);

            *reinterpret_cast<char4 *>(out + j * incB8i + idx) = out4;
        }
    }
    kmax = ldb8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4;
        in4.x = (idx < k) ? T2int_fp<T>(in[idx], sft) : 0;
        in4.y = (idx + 1 < k) ? T2int_fp<T>(in[idx + 1], sft) : 0;
        in4.z = (idx + 2 < k) ? T2int_fp<T>(in[idx + 2], sft) : 0;
        in4.w = (idx + 3 < k) ? T2int_fp<T>(in[idx + 3], sft) : 0;

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {
            const auto val = readtab<T>(j);

            out4.x = (idx < k) ? mod_8i<T, MODE>(in4.x, val) : 0;
            out4.y = (idx + 1 < k) ? mod_8i<T, MODE>(in4.y, val) : 0;
            out4.z = (idx + 2 < k) ? mod_8i<T, MODE>(in4.z, val) : 0;
            out4.w = (idx + 3 < k) ? mod_8i<T, MODE>(in4.w, val) : 0;

            *reinterpret_cast<char4 *>(out + j * incB8i + idx) = out4;
        }
    }
}

template <typename T, int MODE>
__forceinline__ void scaling_launch(
    const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
    const cublasOperation_t op_B, // CUBLAS_OP_N or CUBLAS_OP_T
    const size_t m,               // Number of rows of C
    const size_t n,               // Number of columns of C
    const size_t k,               // Inner dimension
    const unsigned num_moduli,    // #moduli
    const T *const A,             // input
    const size_t lda,             // leading dimension
    const T *const B,             // input
    const size_t ldb,             // leading dimension
    int8_t *const A8i,            // output (lda8i * m)
    const size_t lda8i,           // leading dimension
    const size_t incA8i,          // increment between the A8i
    int16_t *const sftA,          // exponent of shift values for rows of A
    int8_t *const B8i,            // output (ldb8i * n)
    const size_t ldb8i,           // leading dimension
    const size_t incB8i,          // increment between the B8i
    int16_t *const sftB,          // exponent of shift values for cols of B
    const float log2M,            // fld(log2(M-1)/2 - 1.5)
    const bool skip_scalA,        // false (unskip scaling_A) or true (skip scaling_A)
    const bool skip_scalB         // false (unskip scaling_B) or true (skip scaling_B)
) {
    if (!skip_scalA) {
        if (op_A == CUBLAS_OP_N) {
            // m*k -> k*m
            dim3 grid((m + (TILE_DIM - 1)) / TILE_DIM, (lda8i + (TILE_DIM - 1)) / TILE_DIM);
            constexpr dim3 threads1(TILE_DIM, TILE_DIM);
            compute_sft_kernel<<<grid.x, threads1>>>(m, k, A, lda, sftA, log2M);

            if constexpr (USE_CHAR4) {
                constexpr dim3 threads2(TILE_DIM, CHAR4_PER_ROW);
                scalingA_kernel<T, MODE><<<grid, threads2>>>(m, k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA);
            } else {
                scalingA_kernel<T, MODE><<<grid, threads1>>>(m, k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA);
            }
        } else {
            // k*m -> k*m
            scalingB_kernel<T, MODE><<<m, threads_scaling>>>(k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA, log2M);
        }
    }

    if (!skip_scalB) {
        if (op_B == CUBLAS_OP_N) {
            // k*n -> k*n
            scalingB_kernel<T, MODE><<<n, threads_scaling>>>(k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB, log2M);
        } else {
            // n*k -> k*n
            dim3 grid((n + (TILE_DIM - 1)) / TILE_DIM, (ldb8i + (TILE_DIM - 1)) / TILE_DIM);
            constexpr dim3 threads1(TILE_DIM, TILE_DIM);
            compute_sft_kernel<<<grid.x, threads1>>>(n, k, B, ldb, sftB, log2M);

            if constexpr (USE_CHAR4) {
                constexpr dim3 threads2(TILE_DIM, CHAR4_PER_ROW);
                scalingA_kernel<T, MODE><<<grid, threads2>>>(n, k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB);
            } else {
                scalingA_kernel<T, MODE><<<grid, threads1>>>(n, k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB);
            }
        }
    }
}

template <typename T>
__inline__ void scaling(
    const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
    const cublasOperation_t op_B, // CUBLAS_OP_N or CUBLAS_OP_T
    const size_t m,               // Number of rows of C
    const size_t n,               // Number of columns of C
    const size_t k,               // Inner dimension
    const unsigned num_moduli,    // #moduli
    const T *const A,             // input
    const size_t lda,             // leading dimension
    const T *const B,             // input
    const size_t ldb,             // leading dimension
    int8_t *const A8i,            // output (lda8i * m)
    const size_t lda8i,           // leading dimension
    const size_t incA8i,          // increment between the A8i
    int16_t *const sftA,          // exponent of shift values for rows of A
    int8_t *const B8i,            // output (ldb8i * n)
    const size_t ldb8i,           // leading dimension
    const size_t incB8i,          // increment between the B8i
    int16_t *const sftB,          // exponent of shift values for cols of B
    const unsigned table_idx,     // index for table
    const bool skip_scalA,        // false (unskip scaling_A) or true (skip scaling_A)
    const bool skip_scalB         // false (unskip scaling_B) or true (skip scaling_B)
) {
    const float log2M = oz2_table::vecnorm::log2M[table_idx]; // fld(log2(M-1)/2 - 1.5)

    if (num_moduli <= threshold<T>::x) {
        scaling_launch<T, 1>(op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, incA8i, sftA, B8i, ldb8i, incB8i, sftB, log2M, skip_scalA, skip_scalB);
    } else if (num_moduli <= threshold<T>::y) {
        scaling_launch<T, 2>(op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, incA8i, sftA, B8i, ldb8i, incB8i, sftB, log2M, skip_scalA, skip_scalB);
    } else if (num_moduli <= threshold<T>::z) {
        scaling_launch<T, 3>(op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, incA8i, sftA, B8i, ldb8i, incB8i, sftB, log2M, skip_scalA, skip_scalB);
    } else {
        scaling_launch<T, 4>(op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, incA8i, sftA, B8i, ldb8i, incB8i, sftB, log2M, skip_scalA, skip_scalB);
    }
}

} // namespace vecnorm

} // namespace oz2
