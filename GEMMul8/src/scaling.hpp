#pragma once
#include "common.hpp"
#include "table.hpp"

namespace oz2_util {

namespace int8tc {

__forceinline__ __device__ int compute_sft(int amax, int16_t sftA, const float log2M) {
    return sftA + __float2int_rd(__fmaf_rd(minus_half, __log2f(__int2float_rn(amax)), log2M));
}

// extract first 7-bit of A^T
template <typename T>
__global__ void extract_A8i_kernel(const size_t k,                   // size(A,2)
                                   const T *const __restrict__ A,    // input (lda * k)
                                   const size_t lda,                 // leading dimension
                                   int8_t *const __restrict__ A8i,   // output (lda8i * m)
                                   const size_t lda8i,               // leading dimension
                                   int16_t *const __restrict__ sftA) // exponent of shift values
{
    __shared__ T smem[32];
    const auto row_idx             = blockIdx.x;
    const T *const __restrict__ in = A + row_idx;
    const T amax                   = find_amax<T>(in, k, lda, smem);
    const int sft                  = 5 - Tilogb<T>(amax); // 6-bit
    if (threadIdx.x == 0) {
        sftA[row_idx] = sft;
    }

    int8_t *const __restrict__ out = A8i + row_idx * lda8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        char4 out4;
        unsigned idx = i << 2;

        out4.x = T2int8i<T>(in[idx * lda], sft);
        out4.y = T2int8i<T>(in[(idx + 1) * lda], sft);
        out4.z = T2int8i<T>(in[(idx + 2) * lda], sft);
        out4.w = T2int8i<T>(in[(idx + 3) * lda], sft);

        *reinterpret_cast<char4 *>(out + idx) = out4;
    }
    kmax = lda8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        char4 out4;
        unsigned idx = i << 2;

        out4.x = (idx < k) ? T2int8i<T>(in[idx * lda], sft) : 0;
        out4.y = (idx + 1 < k) ? T2int8i<T>(in[(idx + 1) * lda], sft) : 0;
        out4.z = (idx + 2 < k) ? T2int8i<T>(in[(idx + 2) * lda], sft) : 0;
        out4.w = (idx + 3 < k) ? T2int8i<T>(in[(idx + 3) * lda], sft) : 0;

        *reinterpret_cast<char4 *>(out + idx) = out4;
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
    __shared__ T smem[32];
    const auto col_idx             = blockIdx.x;
    const T *const __restrict__ in = B + col_idx * ldb;
    const T amax                   = find_amax<T>(in, k, 1u, smem);
    const int sft                  = 5 - Tilogb<T>(amax); // 6-bit
    if (threadIdx.x == 0) {
        sftB[col_idx] = sft;
    }

    int8_t *const __restrict__ out = B8i + col_idx * ldb8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        char4 out4;
        unsigned idx = i << 2;

        Vec4<T> in4 = *reinterpret_cast<const Vec4<T> *>(in + idx);

        out4.x = T2int8i<T>(in4.x, sft);
        out4.y = T2int8i<T>(in4.y, sft);
        out4.z = T2int8i<T>(in4.z, sft);
        out4.w = T2int8i<T>(in4.w, sft);

        *reinterpret_cast<char4 *>(out + idx) = out4;
    }
    kmax = ldb8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        char4 out4;
        unsigned idx = i << 2;

        out4.x = (idx < k) ? T2int8i<T>(in[idx], sft) : 0;
        out4.y = (idx < k) ? T2int8i<T>(in[idx + 1], sft) : 0;
        out4.z = (idx < k) ? T2int8i<T>(in[idx + 2], sft) : 0;
        out4.w = (idx < k) ? T2int8i<T>(in[idx + 3], sft) : 0;

        *reinterpret_cast<char4 *>(out + idx) = out4;
    }
}

// convert trunc(diag(2^sftA)*A)^T to A8i
template <typename T, int MODE>
__global__ void scalingA_kernel(const size_t n,                         // size(C,2)
                                const size_t k,                         // size(A,2)
                                const size_t incA8i,                    // lda8i * m
                                const unsigned num_moduli,              // #moduli
                                const T *const __restrict__ A,          // input (lda * m)
                                const size_t lda,                       // leading dimension
                                const int32_t *const __restrict__ C32i, // input (ldc32i * n)
                                const size_t ldc32i,                    // leading dimension
                                int8_t *const __restrict__ A8i,         // output (lda8i * m)
                                const size_t lda8i,                     // leading dimension
                                int16_t *const __restrict__ sftA,       // exponent of shift values
                                const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[32];
    const auto row_idx = blockIdx.x;
    const int sftAi    = sftA[row_idx];
    const int32_t amax = find_amax<int32_t>(C32i + row_idx, n, ldc32i, smem);
    const int sft      = compute_sft(amax, sftAi, log2M);

    const T *const __restrict__ in = A + row_idx;
    int8_t *const __restrict__ out = A8i + row_idx * lda8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4;
        in4.x = T2int_fp<T>(in[idx * lda], sft);
        in4.y = T2int_fp<T>(in[(idx + 1) * lda], sft);
        in4.z = T2int_fp<T>(in[(idx + 2) * lda], sft);
        in4.w = T2int_fp<T>(in[(idx + 3) * lda], sft);

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {
            const auto val = readtab<T>(j);

            out4.x = mod_8i<T, MODE>(in4.x, val);
            out4.y = mod_8i<T, MODE>(in4.y, val);
            out4.z = mod_8i<T, MODE>(in4.z, val);
            out4.w = mod_8i<T, MODE>(in4.w, val);

            *reinterpret_cast<char4 *>(out + j * incA8i + idx) = out4;
        }
    }
    kmax = lda8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4;
        in4.x = (idx < k) ? T2int_fp<T>(in[idx * lda], sft) : 0;
        in4.y = (idx + 1 < k) ? T2int_fp<T>(in[(idx + 1) * lda], sft) : 0;
        in4.z = (idx + 2 < k) ? T2int_fp<T>(in[(idx + 2) * lda], sft) : 0;
        in4.w = (idx + 3 < k) ? T2int_fp<T>(in[(idx + 3) * lda], sft) : 0;

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {
            const auto val = readtab<T>(j);

            out4.x = (idx < k) ? mod_8i<T, MODE>(in4.x, val) : 0;
            out4.y = (idx + 1 < k) ? mod_8i<T, MODE>(in4.y, val) : 0;
            out4.z = (idx + 2 < k) ? mod_8i<T, MODE>(in4.z, val) : 0;
            out4.w = (idx + 3 < k) ? mod_8i<T, MODE>(in4.w, val) : 0;

            *reinterpret_cast<char4 *>(out + j * incA8i + idx) = out4;
        }
    }

    if (threadIdx.x == 0) {
        sftA[row_idx] = -sft;
    }
}

// convert trunc(B*diag(2^sftB)) to B8i
template <typename T, int MODE>
__global__ void scalingB_kernel(const size_t m,                         // size(C,1)
                                const size_t k,                         // size(B,1)
                                const size_t incB8i,                    // ldb8i * n
                                const unsigned num_moduli,              // #moduli
                                const T *const __restrict__ B,          // input (ldb * n)
                                const size_t ldb,                       // leading dimension
                                const int32_t *const __restrict__ C32i, // input (ldc32i * n)
                                const size_t ldc32i,                    // leading dimension
                                int8_t *const __restrict__ B8i,         // output (ldb8i * n)
                                const size_t ldb8i,                     // leading dimension
                                int16_t *const __restrict__ sftB,       // exponent of shift values
                                const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[32];
    const auto col_idx = blockIdx.x;
    const int sftBi    = sftB[col_idx];
    const int32_t amax = find_amax<int32_t>(C32i + col_idx * ldc32i, m, 1u, smem);
    const int sft      = compute_sft(amax, sftBi, log2M);

    const T *const __restrict__ in = B + col_idx * ldb;
    int8_t *const __restrict__ out = B8i + col_idx * ldb8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4 = *reinterpret_cast<const Vec4<T> *>(in + idx);

        in4.x = T2int_fp<T>(in4.x, sft);
        in4.y = T2int_fp<T>(in4.y, sft);
        in4.z = T2int_fp<T>(in4.z, sft);
        in4.w = T2int_fp<T>(in4.w, sft);

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

    if (threadIdx.x == 0) {
        sftB[col_idx] = -sft;
    }
}

// convert trunc(A^T*diag(2^sftA))^T to A8i
template <typename T, int MODE>
__global__ void scalingAT_kernel(const size_t n,                         // size(C,2)
                                 const size_t k,                         // size(AT,1)
                                 const size_t incA8i,                    // lda8i * n
                                 const unsigned num_moduli,              // #moduli
                                 const T *const __restrict__ A,          // input (lda * n)
                                 const size_t lda,                       // leading dimension
                                 const int32_t *const __restrict__ C32i, // input (ldc32i * n)
                                 const size_t ldc32i,                    // leading dimension
                                 int8_t *const __restrict__ A8i,         // output (lda8i * n)
                                 const size_t lda8i,                     // leading dimension
                                 int16_t *const __restrict__ sftA,       // exponent of shift values
                                 const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[32];
    const auto col_idx = blockIdx.x;
    const int sftAi    = sftA[col_idx];
    const int32_t amax = find_amax<int32_t>(C32i + col_idx, n, ldc32i, smem);
    const int sft      = compute_sft(amax, sftAi, log2M);

    const T *const __restrict__ in = A + col_idx * lda;
    int8_t *const __restrict__ out = A8i + col_idx * lda8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4 = *reinterpret_cast<const Vec4<T> *>(in + idx);

        in4.x = T2int_fp<T>(in4.x, sft);
        in4.y = T2int_fp<T>(in4.y, sft);
        in4.z = T2int_fp<T>(in4.z, sft);
        in4.w = T2int_fp<T>(in4.w, sft);

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {
            const auto val = readtab<T>(j);

            out4.x = mod_8i<T, MODE>(in4.x, val);
            out4.y = mod_8i<T, MODE>(in4.y, val);
            out4.z = mod_8i<T, MODE>(in4.z, val);
            out4.w = mod_8i<T, MODE>(in4.w, val);

            *reinterpret_cast<char4 *>(out + j * incA8i + idx) = out4;
        }
    }
    kmax = lda8i >> 2;
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

            *reinterpret_cast<char4 *>(out + j * incA8i + idx) = out4;
        }
    }

    if (threadIdx.x == 0) {
        sftA[col_idx] = -sft;
    }
}

// convert trunc(diag(2^sftB)*B^T) to B8i
template <typename T, int MODE>
__global__ void scalingBT_kernel(const size_t m,                         // size(C,1)
                                 const size_t k,                         // size(B,2)
                                 const size_t incB8i,                    // ldb8i * m
                                 const unsigned num_moduli,              // #moduli
                                 const T *const __restrict__ B,          // input (ldb * m)
                                 const size_t ldb,                       // leading dimension
                                 const int32_t *const __restrict__ C32i, // input (ldc32i * n)
                                 const size_t ldc32i,                    // leading dimension
                                 int8_t *const __restrict__ B8i,         // output (ldb8i * m)
                                 const size_t ldb8i,                     // leading dimension
                                 int16_t *const __restrict__ sftB,       // exponent of shift values
                                 const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[32];
    const auto row_idx = blockIdx.x;
    const int sftBi    = sftB[row_idx];
    const int32_t amax = find_amax<int32_t>(C32i + row_idx * ldc32i, m, 1u, smem);
    const int sft      = compute_sft(amax, sftBi, log2M);

    const T *const __restrict__ in = B + row_idx;
    int8_t *const __restrict__ out = B8i + row_idx * ldb8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4;
        in4.x = T2int_fp<T>(in[idx * ldb], sft);
        in4.y = T2int_fp<T>(in[(idx + 1) * ldb], sft);
        in4.z = T2int_fp<T>(in[(idx + 2) * ldb], sft);
        in4.w = T2int_fp<T>(in[(idx + 3) * ldb], sft);

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
        in4.x = (idx < k) ? T2int_fp<T>(in[idx * ldb], sft) : 0;
        in4.y = (idx + 1 < k) ? T2int_fp<T>(in[(idx + 1) * ldb], sft) : 0;
        in4.z = (idx + 2 < k) ? T2int_fp<T>(in[(idx + 2) * ldb], sft) : 0;
        in4.w = (idx + 3 < k) ? T2int_fp<T>(in[(idx + 3) * ldb], sft) : 0;

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

    if (threadIdx.x == 0) {
        sftB[row_idx] = -sft;
    }
}

template <typename T>
__inline__ void scaling(cublasHandle_t handle,        // Handle to the cuBLAS library context
                        const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
                        const cublasOperation_t op_B, // CUBLAS_OP_N or CUBLAS_OP_T
                        const size_t m,               // Number of rows of C
                        const size_t m_pad,           // ((m + 15) >> 4) << 4
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
                        int32_t *const C32i,          // tmp (m_pad * n)
                        const unsigned table_idx)     //
{
    // extract first 7-bit from A and B
    if (op_A == CUBLAS_OP_N) {
        extract_A8i_kernel<T><<<m, oz2_const::threads_scaling>>>(k, A, lda, A8i, lda8i, sftA);
    } else {
        extract_B8i_kernel<T><<<m, oz2_const::threads_scaling>>>(k, A, lda, A8i, lda8i, sftA);
    }
    if (op_B == CUBLAS_OP_N) {
        extract_B8i_kernel<T><<<n, oz2_const::threads_scaling>>>(k, B, ldb, B8i, ldb8i, sftB);
    } else {
        extract_A8i_kernel<T><<<n, oz2_const::threads_scaling>>>(k, B, ldb, B8i, ldb8i, sftB);
    }

    // C32i := A8i^T*B8i
    constexpr int32_t alpha = 1;
    constexpr int32_t beta  = 0;
    cudaDeviceSynchronize();
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m_pad, n, lda8i, &alpha, A8i, CUDA_R_8I, lda8i, B8i, CUDA_R_8I, ldb8i, &beta, C32i, CUDA_R_32I, m_pad, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);

    // extract high order bits from A and B
    cudaDeviceSynchronize();
    const float log2M = oz2_table::int8tc::log2M[table_idx]; // fld(log2(M-1)/2 - 0.5)
    if constexpr (std::is_same_v<T, double>) {
        if (num_moduli <= 12) {
            if (op_A == CUBLAS_OP_N) {
                scalingA_kernel<T, 1><<<m, oz2_const::threads_scaling>>>(n, k, incA8i, num_moduli, A, lda, C32i, m_pad, A8i, lda8i, sftA, log2M);
            } else {
                scalingAT_kernel<T, 1><<<m, oz2_const::threads_scaling>>>(n, k, incA8i, num_moduli, A, lda, C32i, m_pad, A8i, lda8i, sftA, log2M);
            }
            if (op_B == CUBLAS_OP_N) {
                scalingB_kernel<T, 1><<<n, oz2_const::threads_scaling>>>(m, k, incB8i, num_moduli, B, ldb, C32i, m_pad, B8i, ldb8i, sftB, log2M);
            } else {
                scalingBT_kernel<T, 1><<<n, oz2_const::threads_scaling>>>(m, k, incB8i, num_moduli, B, ldb, C32i, m_pad, B8i, ldb8i, sftB, log2M);
            }
        } else if (num_moduli <= 18) {
            if (op_A == CUBLAS_OP_N) {
                scalingA_kernel<T, 2><<<m, oz2_const::threads_scaling>>>(n, k, incA8i, num_moduli, A, lda, C32i, m_pad, A8i, lda8i, sftA, log2M);
            } else {
                scalingAT_kernel<T, 2><<<m, oz2_const::threads_scaling>>>(n, k, incA8i, num_moduli, A, lda, C32i, m_pad, A8i, lda8i, sftA, log2M);
            }
            if (op_B == CUBLAS_OP_N) {
                scalingB_kernel<T, 2><<<n, oz2_const::threads_scaling>>>(m, k, incB8i, num_moduli, B, ldb, C32i, m_pad, B8i, ldb8i, sftB, log2M);
            } else {
                scalingBT_kernel<T, 2><<<n, oz2_const::threads_scaling>>>(m, k, incB8i, num_moduli, B, ldb, C32i, m_pad, B8i, ldb8i, sftB, log2M);
            }
        } else {
            // num_moduli <= 25
            if (op_A == CUBLAS_OP_N) {
                scalingA_kernel<T, 3><<<m, oz2_const::threads_scaling>>>(n, k, incA8i, num_moduli, A, lda, C32i, m_pad, A8i, lda8i, sftA, log2M);
            } else {
                scalingAT_kernel<T, 3><<<m, oz2_const::threads_scaling>>>(n, k, incA8i, num_moduli, A, lda, C32i, m_pad, A8i, lda8i, sftA, log2M);
            }
            if (op_B == CUBLAS_OP_N) {
                scalingB_kernel<T, 3><<<n, oz2_const::threads_scaling>>>(m, k, incB8i, num_moduli, B, ldb, C32i, m_pad, B8i, ldb8i, sftB, log2M);
            } else {
                scalingBT_kernel<T, 3><<<n, oz2_const::threads_scaling>>>(m, k, incB8i, num_moduli, B, ldb, C32i, m_pad, B8i, ldb8i, sftB, log2M);
            }
        }
    } else if constexpr (std::is_same_v<T, float>) {
        if (num_moduli <= 5) {
            if (op_A == CUBLAS_OP_N) {
                scalingA_kernel<T, 1><<<m, oz2_const::threads_scaling>>>(n, k, incA8i, num_moduli, A, lda, C32i, m_pad, A8i, lda8i, sftA, log2M);
            } else {
                scalingAT_kernel<T, 1><<<m, oz2_const::threads_scaling>>>(n, k, incA8i, num_moduli, A, lda, C32i, m_pad, A8i, lda8i, sftA, log2M);
            }
            if (op_B == CUBLAS_OP_N) {
                scalingB_kernel<T, 1><<<n, oz2_const::threads_scaling>>>(m, k, incB8i, num_moduli, B, ldb, C32i, m_pad, B8i, ldb8i, sftB, log2M);
            } else {
                scalingBT_kernel<T, 1><<<n, oz2_const::threads_scaling>>>(m, k, incB8i, num_moduli, B, ldb, C32i, m_pad, B8i, ldb8i, sftB, log2M);
            }
        } else if (num_moduli <= 11) {
            if (op_A == CUBLAS_OP_N) {
                scalingA_kernel<T, 2><<<m, oz2_const::threads_scaling>>>(n, k, incA8i, num_moduli, A, lda, C32i, m_pad, A8i, lda8i, sftA, log2M);
            } else {
                scalingAT_kernel<T, 2><<<m, oz2_const::threads_scaling>>>(n, k, incA8i, num_moduli, A, lda, C32i, m_pad, A8i, lda8i, sftA, log2M);
            }
            if (op_B == CUBLAS_OP_N) {
                scalingB_kernel<T, 2><<<n, oz2_const::threads_scaling>>>(m, k, incB8i, num_moduli, B, ldb, C32i, m_pad, B8i, ldb8i, sftB, log2M);
            } else {
                scalingBT_kernel<T, 2><<<n, oz2_const::threads_scaling>>>(m, k, incB8i, num_moduli, B, ldb, C32i, m_pad, B8i, ldb8i, sftB, log2M);
            }
        } else {
            // num_moduli <= 18
            if (op_A == CUBLAS_OP_N) {
                scalingA_kernel<T, 3><<<m, oz2_const::threads_scaling>>>(n, k, incA8i, num_moduli, A, lda, C32i, m_pad, A8i, lda8i, sftA, log2M);
            } else {
                scalingAT_kernel<T, 3><<<m, oz2_const::threads_scaling>>>(n, k, incA8i, num_moduli, A, lda, C32i, m_pad, A8i, lda8i, sftA, log2M);
            }
            if (op_B == CUBLAS_OP_N) {
                scalingB_kernel<T, 3><<<n, oz2_const::threads_scaling>>>(m, k, incB8i, num_moduli, B, ldb, C32i, m_pad, B8i, ldb8i, sftB, log2M);
            } else {
                scalingBT_kernel<T, 3><<<n, oz2_const::threads_scaling>>>(m, k, incB8i, num_moduli, B, ldb, C32i, m_pad, B8i, ldb8i, sftB, log2M);
            }
        }
    }
}

} // namespace int8tc

namespace vecnorm {

template <typename T> __forceinline__ __device__ int compute_sft(T amax, T vecnrm, const float log2M);
template <> __forceinline__ __device__ int compute_sft<double>(double amax, double vecnrm, const float log2M) {
    const int exponent  = Tilogb<double>(vecnrm);
    const float vecnrmf = __double2float_ru(scalbn(vecnrm, -exponent));
    const int k         = __float2int_rd(__fmaf_rd(minus_half, __fadd_ru(__log2f(vecnrmf), exponent), log2M));
    return min(__float2int_rd(log2M - 1.0f), k) - Tilogb<double>(amax);
}
template <> __forceinline__ __device__ int compute_sft<float>(float amax, float vecnrm, const float log2M) {
    const int k = __float2int_rd(__fmaf_rd(minus_half, __log2f(vecnrm), log2M));
    return min(__float2int_rd(log2M - 1.0f), k) - Tilogb<float>(amax);
}

// convert trunc(diag(2^sftA)*A)^T to A8i
template <typename T, int MODE>
__global__ void scalingA_kernel(const size_t k,                   // size(A,2)
                                const size_t incA8i,              // lda8i * m
                                const unsigned num_moduli,        // #moduli
                                const T *const __restrict__ A,    // input (lda * n)
                                const size_t lda,                 // leading dimension
                                int8_t *const __restrict__ A8i,   // output (lda8i * m)
                                const size_t lda8i,               // leading dimension
                                int16_t *const __restrict__ sftA, // exponent of shift values
                                const float log2M)                // log2(M-1)/2 - 1.5
{
    __shared__ T smem[64];
    const auto row_idx             = blockIdx.x;
    const T *const __restrict__ in = A + row_idx;
    T vecnrm;
    const T amax  = find_amax_and_nrm<T>(in, k, lda, smem, vecnrm);
    const int sft = compute_sft<T>(amax, vecnrm, log2M);
    if (threadIdx.x == 0) {
        sftA[row_idx] = -sft;
    }

    int8_t *const __restrict__ out = A8i + row_idx * lda8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4;
        in4.x = T2int_fp<T>(in[idx * lda], sft);
        in4.y = T2int_fp<T>(in[(idx + 1) * lda], sft);
        in4.z = T2int_fp<T>(in[(idx + 2) * lda], sft);
        in4.w = T2int_fp<T>(in[(idx + 3) * lda], sft);

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {
            const auto val = readtab<T>(j);

            out4.x = mod_8i<T, MODE>(in4.x, val);
            out4.y = mod_8i<T, MODE>(in4.y, val);
            out4.z = mod_8i<T, MODE>(in4.z, val);
            out4.w = mod_8i<T, MODE>(in4.w, val);

            *reinterpret_cast<char4 *>(out + j * incA8i + idx) = out4;
        }
    }
    kmax = lda8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4;
        in4.x = (idx < k) ? T2int_fp<T>(in[idx * lda], sft) : 0;
        in4.y = (idx + 1 < k) ? T2int_fp<T>(in[(idx + 1) * lda], sft) : 0;
        in4.z = (idx + 2 < k) ? T2int_fp<T>(in[(idx + 2) * lda], sft) : 0;
        in4.w = (idx + 3 < k) ? T2int_fp<T>(in[(idx + 3) * lda], sft) : 0;

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {
            const auto val = readtab<T>(j);

            out4.x = (idx < k) ? mod_8i<T, MODE>(in4.x, val) : 0;
            out4.y = (idx + 1 < k) ? mod_8i<T, MODE>(in4.y, val) : 0;
            out4.z = (idx + 2 < k) ? mod_8i<T, MODE>(in4.z, val) : 0;
            out4.w = (idx + 3 < k) ? mod_8i<T, MODE>(in4.w, val) : 0;

            *reinterpret_cast<char4 *>(out + j * incA8i + idx) = out4;
        }
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
                                int16_t *const __restrict__ sftB, // exponent of shift values
                                const float log2M)                // log2(M-1)/2 - 1.5
{
    __shared__ T smem[64];
    const auto col_idx             = blockIdx.x;
    const T *const __restrict__ in = B + col_idx * ldb;
    T vecnrm;
    const T amax  = find_amax_and_nrm<T>(in, k, 1u, smem, vecnrm);
    const int sft = compute_sft<T>(amax, vecnrm, log2M);
    if (threadIdx.x == 0) {
        sftB[col_idx] = -sft;
    }

    int8_t *const __restrict__ out = B8i + col_idx * ldb8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4 = *reinterpret_cast<const Vec4<T> *>(in + idx);

        in4.x = T2int_fp<T>(in4.x, sft);
        in4.y = T2int_fp<T>(in4.y, sft);
        in4.z = T2int_fp<T>(in4.z, sft);
        in4.w = T2int_fp<T>(in4.w, sft);

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
__inline__ void scaling(const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
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
                        const unsigned table_idx)     //
{
    const float log2M = oz2_table::vecnorm::log2M[table_idx]; // fld(log2(M-1)/2 - 1.5)

    if constexpr (std::is_same_v<T, double>) {
        if (num_moduli <= 12) {
            if (op_A == CUBLAS_OP_N) {
                scalingA_kernel<T, 1><<<m, oz2_const::threads_scaling>>>(k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA, log2M);
            } else {
                scalingB_kernel<T, 1><<<m, oz2_const::threads_scaling>>>(k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA, log2M);
            }
            if (op_B == CUBLAS_OP_N) {
                scalingB_kernel<T, 1><<<n, oz2_const::threads_scaling>>>(k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB, log2M);
            } else {
                scalingA_kernel<T, 1><<<n, oz2_const::threads_scaling>>>(k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB, log2M);
            }
        } else if (num_moduli <= 18) {
            if (op_A == CUBLAS_OP_N) {
                scalingA_kernel<T, 2><<<m, oz2_const::threads_scaling>>>(k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA, log2M);
            } else {
                scalingB_kernel<T, 2><<<m, oz2_const::threads_scaling>>>(k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA, log2M);
            }
            if (op_B == CUBLAS_OP_N) {
                scalingB_kernel<T, 2><<<n, oz2_const::threads_scaling>>>(k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB, log2M);
            } else {
                scalingA_kernel<T, 2><<<n, oz2_const::threads_scaling>>>(k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB, log2M);
            }
        } else {
            // num_moduli <= 25
            if (op_A == CUBLAS_OP_N) {
                scalingA_kernel<T, 3><<<m, oz2_const::threads_scaling>>>(k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA, log2M);
            } else {
                scalingB_kernel<T, 3><<<m, oz2_const::threads_scaling>>>(k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA, log2M);
            }
            if (op_B == CUBLAS_OP_N) {
                scalingB_kernel<T, 3><<<n, oz2_const::threads_scaling>>>(k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB, log2M);
            } else {
                scalingA_kernel<T, 3><<<n, oz2_const::threads_scaling>>>(k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB, log2M);
            }
        }
    } else if constexpr (std::is_same_v<T, float>) {
        if (num_moduli <= 5) {
            if (op_A == CUBLAS_OP_N) {
                scalingA_kernel<T, 1><<<m, oz2_const::threads_scaling>>>(k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA, log2M);
            } else {
                scalingB_kernel<T, 1><<<m, oz2_const::threads_scaling>>>(k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA, log2M);
            }
            if (op_B == CUBLAS_OP_N) {
                scalingB_kernel<T, 1><<<n, oz2_const::threads_scaling>>>(k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB, log2M);
            } else {
                scalingA_kernel<T, 1><<<n, oz2_const::threads_scaling>>>(k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB, log2M);
            }
        } else if (num_moduli <= 11) {
            if (op_A == CUBLAS_OP_N) {
                scalingA_kernel<T, 2><<<m, oz2_const::threads_scaling>>>(k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA, log2M);
            } else {
                scalingB_kernel<T, 2><<<m, oz2_const::threads_scaling>>>(k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA, log2M);
            }
            if (op_B == CUBLAS_OP_N) {
                scalingB_kernel<T, 2><<<n, oz2_const::threads_scaling>>>(k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB, log2M);
            } else {
                scalingA_kernel<T, 2><<<n, oz2_const::threads_scaling>>>(k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB, log2M);
            }
        } else { // num_moduli <= 18
            if (op_A == CUBLAS_OP_N) {
                scalingA_kernel<T, 3><<<m, oz2_const::threads_scaling>>>(k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA, log2M);
            } else {
                scalingB_kernel<T, 3><<<m, oz2_const::threads_scaling>>>(k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA, log2M);
            }
            if (op_B == CUBLAS_OP_N) {
                scalingB_kernel<T, 3><<<n, oz2_const::threads_scaling>>>(k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB, log2M);
            } else {
                scalingA_kernel<T, 3><<<n, oz2_const::threads_scaling>>>(k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB, log2M);
            }
        }
    }
}

} // namespace vecnorm

} // namespace oz2_util
