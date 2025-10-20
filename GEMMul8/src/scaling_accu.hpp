#pragma once
#include "common.hpp"
#include "scaling_fast.hpp"

namespace oz2 {
namespace int8tc {

template <typename T>
__global__ void compute_sft_extract_kernel(const size_t m,                   // size(A,1)
                                           const size_t k,                   // size(A,2)
                                           const T *const A,                 // input (lda * k)
                                           const size_t lda,                 // leading dimension
                                           int16_t *const __restrict__ sftA) // exponent of shift values
{
    const auto block_base = blockIdx.x << LOG2_TILE_DIM;
    const auto row        = block_base + threadIdx.x;

    __shared__ T samax[TILE_DIM][TILE_DIM + 1];

    T amax = Tzero<T>::value;
    if (row < m) {
        const T *row_ptr = A + row;
        for (unsigned col = threadIdx.y; col < k; col += blockDim.y) {
            const T tmp = Tabs<T>(__ldg(row_ptr + col * lda));
            amax        = max(amax, tmp);
        }
    }
    samax[threadIdx.y][threadIdx.x] = amax;
    __syncthreads();

    amax = samax[threadIdx.x][threadIdx.y];
    amax = inner_warp_max<T, TILE_DIM>(amax);

    if (row < m && threadIdx.x == 0) {
        const int sft                  = 5 - Tilogb<T>(amax); // 6-bit
        sftA[block_base + threadIdx.y] = sft;
    }
}

// extract first 7-bit of A^T
template <typename T>
__global__ void extract_A8i_kernel(const size_t m,                   // size(A,1)
                                   const size_t k,                   // size(A,2)
                                   const T *const __restrict__ A,    // input (lda * k)
                                   const size_t lda,                 // leading dimension
                                   int8_t *const __restrict__ A8i,   // output (lda8i * m)
                                   const size_t lda8i,               // leading dimension
                                   int16_t *const __restrict__ sftA) // exponent of shift values
{
    __shared__ int8_t tile[TILE_DIM][TILE_DIM + 1];

    const auto rowBase = blockIdx.x << LOG2_TILE_DIM;
    const auto colBase = blockIdx.y << LOG2_TILE_DIM;

    if constexpr (USE_CHAR4) {

        const auto in_row = rowBase + threadIdx.x;
        if (in_row < m) {
            const int sft = sftA[in_row];
            for (unsigned t = threadIdx.y; t < TILE_DIM; t += blockDim.y) {
                const auto in_col    = colBase + t;
                tile[t][threadIdx.x] = (in_col < k) ? T2int8i<T>(__ldg(A + in_row + in_col * lda), sft) : 0;
            }
        }
        __syncthreads();

        const auto tile_row = (threadIdx.x & (CHAR4_PER_ROW - 1)) << 2;
        const auto tile_col = (threadIdx.x >> LOG2_CHAR4_PER_ROW) + (threadIdx.y << 2);
        const auto out_row  = colBase + tile_row;
        const auto out_col  = rowBase + tile_col;

        if (out_row < lda8i && out_col < m) {

            char4 out4;
            out4.x = tile[tile_row][tile_col];
            out4.y = tile[tile_row + 1][tile_col];
            out4.z = tile[tile_row + 2][tile_col];
            out4.w = tile[tile_row + 3][tile_col];

            *reinterpret_cast<char4 *>(A8i + out_row + out_col * lda8i) = out4;
        }

    } else {

        const auto in_row = rowBase + threadIdx.x;
        const auto in_col = colBase + threadIdx.y;
        if (in_row < m) {
            const int sft                  = sftA[in_row];
            tile[threadIdx.y][threadIdx.x] = (in_col < k) ? T2int8i<T>(__ldg(A + in_row + in_col * lda), sft) : 0;
        }
        __syncthreads();

        const auto out_row = colBase + threadIdx.x;
        const auto out_col = rowBase + threadIdx.y;
        if (out_row < lda8i && out_col < m) {
            A8i[out_row + out_col * lda8i] = tile[threadIdx.x][threadIdx.y];
        }
    }
}

// extract first 7-bit of B
template <typename T>
__global__ void extract_B8i_kernel(const size_t k,                   // size(B,1)
                                   const T *const __restrict__ B,    // input (ldb * n)
                                   const size_t ldb,                 // leading dimension
                                   int8_t *const __restrict__ B8i,   // output (ldb8i * n)
                                   const size_t ldb8i,               // leading dimension
                                   int16_t *const __restrict__ sftB) // exponent of shift values
{
    __shared__ T shm[32];
    const auto col_idx             = blockIdx.x;
    const T *const __restrict__ in = B + col_idx * ldb;
    const T amax                   = find_amax<T>(in, k, shm);
    const int sft                  = 5 - Tilogb<T>(amax); // 6-bit
    if (threadIdx.x == 0) {
        sftB[col_idx] = sft;
    }

    int8_t *const __restrict__ out = B8i + col_idx * ldb8i;
    unsigned kmax                  = k >> 2;
    unsigned i                     = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        char4 out4;
        out4.x = T2int8i<T>(in[idx], sft);
        out4.y = T2int8i<T>(in[idx + 1], sft);
        out4.z = T2int8i<T>(in[idx + 2], sft);
        out4.w = T2int8i<T>(in[idx + 3], sft);

        *reinterpret_cast<char4 *>(out + idx) = out4;
    }
    kmax = ldb8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        char4 out4;
        out4.x = (idx < k) ? T2int8i<T>(in[idx], sft) : 0;
        out4.y = (idx + 1 < k) ? T2int8i<T>(in[idx + 1], sft) : 0;
        out4.z = (idx + 2 < k) ? T2int8i<T>(in[idx + 2], sft) : 0;
        out4.w = (idx + 3 < k) ? T2int8i<T>(in[idx + 3], sft) : 0;

        *reinterpret_cast<char4 *>(out + idx) = out4;
    }
}

__forceinline__ __device__ int compute_sft(int amax, const float log2M) {
    return __float2int_rd(__fmaf_rd(-0x1.0000060000000p-1F, __log2f(__int2float_rn(amax)), log2M));
}

__global__ void compute_sft_scaling_rowwise_kernel(const size_t m,                  // size(C32i,1)
                                                   const size_t n,                  // size(C32i,2)
                                                   const int *const C32i,           // input (ldc32i * n)
                                                   const size_t ldc32i,             // leading dimension
                                                   int16_t *const __restrict__ sft, // exponent of shift values
                                                   const float log2M)               // log2(M-1)/2 - 0.5
{
    const auto block_base = blockIdx.x << LOG2_TILE_DIM;
    const auto row        = block_base + threadIdx.x;

    __shared__ int samax[TILE_DIM][TILE_DIM + 1];

    int amax = 0;
    if (row < m) {
        const int *row_ptr = C32i + row;
        for (unsigned c = threadIdx.y; c < n; c += blockDim.y) {
            const int tmp = abs(__ldg(row_ptr + c * ldc32i));
            amax          = max(amax, tmp);
        }
    }
    samax[threadIdx.y][threadIdx.x] = amax;
    __syncthreads();

    amax = samax[threadIdx.x][threadIdx.y];
    amax = inner_warp_max<int, TILE_DIM>(amax);

    if (row < m && threadIdx.x == 0) {
        const int s                   = sft[block_base + threadIdx.y] + compute_sft(amax, log2M);
        sft[block_base + threadIdx.y] = -s;
    }
}

__global__ void compute_sft_scaling_colwise_kernel(const size_t m,                  // size(C32i,1)
                                                   const int *const C32i,           // input (ldc32i * n)
                                                   const size_t ldc32i,             // leading dimension
                                                   int16_t *const __restrict__ sft, // exponent of shift values
                                                   const float log2M)               // log2(M-1)/2 - 0.5
{
    __shared__ int shm[32];
    const auto col_idx = blockIdx.x;
    int sft_tmp        = sft[col_idx];
    const int amax     = find_amax<int>(C32i + col_idx * ldc32i, m, shm);
    sft_tmp += compute_sft(amax, log2M);
    if (threadIdx.x == 0) {
        sft[col_idx] = -sft_tmp;
    }
}

// convert trunc(B*diag(2^sftB)) to B8i
template <typename T, int MODE>
__global__ void scalingB_kernel(const size_t k,                   // size(B,1)
                                const size_t incB8i,              // ldb8i * n
                                const unsigned num_moduli,        // #moduli
                                const T *const __restrict__ B,    // input (ldb * n)
                                const size_t ldb,                 // leading dimension
                                int8_t *const __restrict__ B8i,   // output (ldb8i * n)
                                const size_t ldb8i,               // leading dimension
                                int16_t *const __restrict__ sftB) // exponent of shift values
{
    const auto col_idx = blockIdx.x;
    const int sft      = -sftB[col_idx];

    const T *const __restrict__ in = B + col_idx * ldb;
    int8_t *const __restrict__ out = B8i + col_idx * ldb8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
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

template <typename T>
__forceinline__ void extract_8i_launch(const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
                                       const cublasOperation_t op_B, // CUBLAS_OP_N or CUBLAS_OP_T
                                       const size_t m,               // Number of rows of C
                                       const size_t n,               // Number of columns of C
                                       const size_t k,               // Inner dimension
                                       const T *const A,             // input
                                       const size_t lda,             // leading dimension
                                       const T *const B,             // input
                                       const size_t ldb,             // leading dimension
                                       int8_t *const A8i,            // output (lda8i * m)
                                       const size_t lda8i,           // leading dimension
                                       int16_t *const sftA,          // exponent of shift values for rows of A
                                       int8_t *const B8i,            // output (ldb8i * n)
                                       const size_t ldb8i,           // leading dimension
                                       int16_t *const sftB)          // exponent of shift values for cols of B
{
    if (op_A == CUBLAS_OP_N) {
        // m*k -> k*m
        constexpr dim3 threads1(TILE_DIM, TILE_DIM);
        dim3 grid((m + (TILE_DIM - 1)) / TILE_DIM, (lda8i + (TILE_DIM - 1)) / TILE_DIM);
        compute_sft_extract_kernel<<<grid.x, threads1>>>(m, k, A, lda, sftA);

        if constexpr (USE_CHAR4) {
            constexpr dim3 threads2(TILE_DIM, CHAR4_PER_ROW);
            extract_A8i_kernel<T><<<grid, threads2>>>(m, k, A, lda, A8i, lda8i, sftA);
        } else {
            extract_A8i_kernel<T><<<grid, threads1>>>(m, k, A, lda, A8i, lda8i, sftA);
        }
    } else {
        // k*m -> k*m
        extract_B8i_kernel<T><<<m, threads_scaling>>>(k, A, lda, A8i, lda8i, sftA);
    }

    if (op_B == CUBLAS_OP_N) {
        // k*n -> k*n
        extract_B8i_kernel<T><<<n, threads_scaling>>>(k, B, ldb, B8i, ldb8i, sftB);
    } else {
        // n*k -> k*n
        constexpr dim3 threads1(TILE_DIM, TILE_DIM);
        dim3 grid((n + (TILE_DIM - 1)) / TILE_DIM, (ldb8i + (TILE_DIM - 1)) / TILE_DIM);
        compute_sft_extract_kernel<<<grid.x, threads1>>>(n, k, B, ldb, sftB);

        if constexpr (USE_CHAR4) {
            constexpr dim3 threads2(TILE_DIM, CHAR4_PER_ROW);
            extract_A8i_kernel<T><<<grid, threads2>>>(n, k, B, ldb, B8i, ldb8i, sftB);
        } else {
            extract_A8i_kernel<T><<<grid, threads1>>>(n, k, B, ldb, B8i, ldb8i, sftB);
        }
    }
}

template <typename T, int MODE>
__forceinline__ void scaling_launch(const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
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
                                    int32_t *const C32i,          // tmp (ldc32i * n)
                                    const size_t ldc32i,          // ((m + 15) >> 4) << 4
                                    const float log2M)            // fld(log2(M-1)/2 - 0.5)
{
    if (op_A == CUBLAS_OP_N) {
        // m*k -> k*m
        dim3 grid((m + (TILE_DIM - 1)) / TILE_DIM, (lda8i + (TILE_DIM - 1)) / TILE_DIM);
        constexpr dim3 threads1(TILE_DIM, TILE_DIM);
        compute_sft_scaling_rowwise_kernel<<<grid.x, threads1>>>(m, n, C32i, ldc32i, sftA, log2M);

        if constexpr (USE_CHAR4) {
            constexpr dim3 threads2(TILE_DIM, CHAR4_PER_ROW);
            oz2::vecnorm::scalingA_kernel<T, MODE><<<grid, threads2>>>(m, k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA);
        } else {
            oz2::vecnorm::scalingA_kernel<T, MODE><<<grid, threads1>>>(m, k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA);
        }
    } else {
        // k*m -> k*m
        constexpr dim3 threads1(TILE_DIM, TILE_DIM);
        compute_sft_scaling_rowwise_kernel<<<(m + (TILE_DIM - 1)) / TILE_DIM, threads1>>>(m, n, C32i, ldc32i, sftA, log2M);
        scalingB_kernel<T, MODE><<<m, threads_scaling>>>(k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA);
    }

    if (op_B == CUBLAS_OP_N) {
        // k*n -> k*n
        compute_sft_scaling_colwise_kernel<<<n, threads_scaling>>>(m, C32i, ldc32i, sftB, log2M);
        scalingB_kernel<T, MODE><<<n, threads_scaling>>>(k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB);
    } else {
        // n*k -> k*n
        dim3 grid((n + (TILE_DIM - 1)) / TILE_DIM, (ldb8i + (TILE_DIM - 1)) / TILE_DIM);
        compute_sft_scaling_colwise_kernel<<<n, threads_scaling>>>(m, C32i, ldc32i, sftB, log2M);

        if constexpr (USE_CHAR4) {
            constexpr dim3 threads2(TILE_DIM, CHAR4_PER_ROW);
            oz2::vecnorm::scalingA_kernel<T, MODE><<<grid, threads2>>>(n, k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB);
        } else {
            constexpr dim3 threads2(TILE_DIM, TILE_DIM);
            oz2::vecnorm::scalingA_kernel<T, MODE><<<grid, threads2>>>(n, k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB);
        }
    }
}

template <typename T>
__inline__ void scaling(cublasHandle_t handle,        // Handle to the cuBLAS library context
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
                        int32_t *const C32i,          // tmp (ldc32i * n)
                        const size_t ldc32i,          // ((m + 15) >> 4) << 4
                        const unsigned table_idx)     //
{
    // extract first 7-bit from A and B
    extract_8i_launch<T>(op_A, op_B, m, n, k, A, lda, B, ldb, A8i, lda8i, sftA, B8i, ldb8i, sftB);

    // C32i := A8i^T*B8i
    constexpr int32_t alpha = 1;
    constexpr int32_t beta  = 0;
    cudaDeviceSynchronize();
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, ldc32i, n, lda8i, &alpha, A8i, CUDA_R_8I, lda8i, B8i, CUDA_R_8I, ldb8i, &beta, C32i, CUDA_R_32I, ldc32i, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);

    // extract high order bits from A and B
    const float log2M = oz2_table::int8tc::log2M[table_idx]; // fld(log2(M-1)/2 - 0.5)
    cudaDeviceSynchronize();
    if (num_moduli <= threshold<T>::x) {
        scaling_launch<T, 1>(op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, incA8i, sftA, B8i, ldb8i, incB8i, sftB, C32i, ldc32i, log2M);
    } else if (num_moduli <= threshold<T>::y) {
        scaling_launch<T, 2>(op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, incA8i, sftA, B8i, ldb8i, incB8i, sftB, C32i, ldc32i, log2M);
    } else if (num_moduli <= threshold<T>::z) {
        scaling_launch<T, 3>(op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, incA8i, sftA, B8i, ldb8i, incB8i, sftB, C32i, ldc32i, log2M);
    } else {
        scaling_launch<T, 4>(op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, incA8i, sftA, B8i, ldb8i, incB8i, sftB, C32i, ldc32i, log2M);
    }
}

} // namespace int8tc

} // namespace oz2
