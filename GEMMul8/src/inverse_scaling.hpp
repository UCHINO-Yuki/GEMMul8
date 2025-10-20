#pragma once
#include "common.hpp"

namespace oz2 {

// C := C64f - round(C64f/M)*M
template <typename TC, typename TM>
__forceinline__ __device__ TC invscal_device(const size_t mem_idx,                  //
                                             const unsigned num_moduli,             //
                                             const size_t sizeC,                    //
                                             const size_t incC8u,                   //
                                             const uint8_t *const __restrict__ C8u, // input
                                             const size_t ldc8u,                    // leading dim of C8u
                                             const double invM,                     //
                                             const TM M,                            //
                                             const int sft)                         // exponent of shift values
{
    if constexpr (std::is_same_v<TM, double>) {

        double C64f = 0.0;
        for (unsigned i = 0; i < num_moduli; ++i) {
            const double C8u_tmp = __uint2double_rn(0u + C8u[i * incC8u + mem_idx]); // __uint2double_rn(static_cast<uint32_t>(C8u[i * incC8u + idx]));
            const double NMi     = oz2_table::NMi_dev[i];                            // table
            C64f                 = fma(NMi, C8u_tmp, C64f);                          // error-free
        }

        const double quot = -rint(C64f * invM);
        TC tmpC           = Tcast<TC>(fma(quot, M, C64f));
        return Tscalbn<TC>(tmpC, sft);

    } else if constexpr (std::is_same_v<TM, double2>) {

        double2 C64f{0.0, 0.0};
        for (unsigned i = 0; i < num_moduli; ++i) {
            const double C8u_tmp = __uint2double_rn(0u + C8u[i * incC8u + mem_idx]); // __uint2double_rn(static_cast<uint32_t>(C8u[i * incC8u + idx]));
            double2 NMi{oz2_table::NMi_dev[i * 2], oz2_table::NMi_dev[i * 2 + 1]};   // table
            C64f.x = fma(NMi.x, C8u_tmp, C64f.x);                                    // error-free
            C64f.y = fma(NMi.y, C8u_tmp, C64f.y);                                    // not error-free
        }

        const double quot = -rint(C64f.x * invM);
        double tmpC1      = fma(quot, M.x, C64f.x) + C64f.y;
        TC tmpC2          = Tcast<TC>(fma(quot, M.y, tmpC1));
        return Tscalbn<TC>(tmpC2, sft);

    } else {

        return Tzero<TC>::value;
    }
}

// C := diag(2^sftA) * C * diag(2^sftB)
template <typename TC, typename TM, int ALPHA, int BETA>
__global__ void invscal_kernel_special(const unsigned num_moduli,              //
                                       const size_t m,                         // size(C64f,1)
                                       const size_t sizeC,                     //
                                       const size_t incC8u,                    //
                                       const uint8_t *const __restrict__ C8u,  // input
                                       const size_t ldc8u,                     // leading dim of C8u
                                       TC *const __restrict__ C,               // output
                                       const size_t ldc,                       // leading dimension
                                       const double invM,                      //
                                       const TM M,                             //
                                       const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                       const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col     = idx / m;
    const auto row     = idx - col * m;
    const auto mem_idx = col * ldc8u + row;
    TC tmpC            = invscal_device<TC, TM>(mem_idx, num_moduli, sizeC, incC8u, C8u, ldc8u, invM, M, sftA[row] + sftB[col]);

    const auto idxC = col * ldc + row;
    if constexpr (ALPHA == 1 && BETA == 0) {
        C[idxC] = tmpC;
    } else if constexpr (ALPHA == 1 && BETA == 1) {
        C[idxC] += tmpC;
    } else if constexpr (ALPHA == -1 && BETA == 0) {
        C[idxC] = -tmpC;
    } else if constexpr (ALPHA == -1 && BETA == 1) {
        C[idxC] -= tmpC;
    }
}

template <typename TC, typename TM>
__global__ void invscal_kernel(const TC alpha,                         //
                               const TC beta,                          //
                               const unsigned num_moduli,              //
                               const size_t m,                         // size(C64f,1)
                               const size_t sizeC,                     //
                               const size_t incC8u,                    //
                               const uint8_t *const __restrict__ C8u,  // input
                               const size_t ldc8u,                     // leading dim of C8u
                               TC *const __restrict__ C,               // output
                               const size_t ldc,                       // leading dimension
                               const double invM,                      //
                               const TM M,                             //
                               const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                               const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col     = idx / m;
    const auto row     = idx - col * m;
    const auto mem_idx = col * ldc8u + row;
    TC tmpC            = invscal_device<TC, TM>(mem_idx, num_moduli, sizeC, incC8u, C8u, ldc8u, invM, M, sftA[row] + sftB[col]);

    const auto idxC = col * ldc + row;
    C[idxC]         = Tfma<TC>(beta, C[idxC], alpha * tmpC);
}

// interface!!
template <typename T>
__inline__ void inverse_scaling(const bool is_numM_1,
                                const unsigned num_moduli,
                                const size_t m,            // size(C,1)
                                const size_t n,            // size(C,2)
                                const uint8_t *const C8u,  // input
                                const size_t ldc8u,        // leading dim of C8u
                                const size_t incC8u,       //
                                T *const C,                // output
                                const size_t ldc,          // leading dimension
                                const int16_t *const sftA, // exponent of shift values for rows of A
                                const int16_t *const sftB, // exponent of shift values for cols of B
                                const T alpha,             //
                                const T beta)              //
{
    const unsigned table_idx = num_moduli - 2;
    const size_t sizeC       = m * n;
    const double invM        = oz2_table::invM[table_idx];
    if (is_numM_1) {
        const double M = oz2_table::M[table_idx][0];
        if (alpha == Tone<T>::value) {
            if (beta == Tzero<T>::value) {
                // C = A*B
                invscal_kernel_special<T, double, 1, 0><<<oz2::grid_invscal, oz2::threads_invscal>>>(num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
                return;
            } else if (beta == Tone<T>::value) {
                // C += A*B
                invscal_kernel_special<T, double, 1, 1><<<oz2::grid_invscal, oz2::threads_invscal>>>(num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
                return;
            }
        } else if (alpha == Tmone<T>::value) {
            if (beta == Tzero<T>::value) {
                // C = -A*B
                invscal_kernel_special<T, double, -1, 0><<<oz2::grid_invscal, oz2::threads_invscal>>>(num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
                return;
            } else if (beta == Tone<T>::value) {
                // C -= A*B
                invscal_kernel_special<T, double, -1, 1><<<oz2::grid_invscal, oz2::threads_invscal>>>(num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
                return;
            }
        }
        invscal_kernel<T, double><<<oz2::grid_invscal, oz2::threads_invscal>>>(alpha, beta, num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
    } else {
        const double2 M{oz2_table::M[table_idx][0], oz2_table::M[table_idx][1]};
        if (alpha == Tone<T>::value) {
            if (beta == Tzero<T>::value) {
                // C = A*B
                invscal_kernel_special<T, double2, 1, 0><<<oz2::grid_invscal, oz2::threads_invscal>>>(num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
                return;
            } else if (beta == Tone<T>::value) {
                // C += A*B
                invscal_kernel_special<T, double2, 1, 1><<<oz2::grid_invscal, oz2::threads_invscal>>>(num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
                return;
            }
        } else if (alpha == Tmone<T>::value) {
            if (beta == Tzero<T>::value) {
                // C = -A*B
                invscal_kernel_special<T, double2, -1, 0><<<oz2::grid_invscal, oz2::threads_invscal>>>(num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
                return;
            } else if (beta == Tone<T>::value) {
                // C -= A*B
                invscal_kernel_special<T, double2, -1, 1><<<oz2::grid_invscal, oz2::threads_invscal>>>(num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
                return;
            }
        }
        invscal_kernel<T, double2><<<oz2::grid_invscal, oz2::threads_invscal>>>(alpha, beta, num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
    }
}

} // namespace oz2
