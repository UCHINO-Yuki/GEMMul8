#include "../include/gemmul8.hpp"
#include "common.hpp"
#include "conv_32i_2_8u.hpp"
#include "inverse_scaling.hpp"
#include "scaling_accu.hpp"
#include "scaling_fast.hpp"

namespace gemmul8 {

//------------------------------
// Calculating required work size
//------------------------------
size_t workSize(const size_t m,            // Number of rows of C
                const size_t n,            // Number of columns of C
                const size_t k,            // Inner dimension <= 2^17
                const unsigned num_moduli) // #moduli, 2 <= num_moduli <= 20
{
    const size_t lda8i     = oz2::calc_ld8i(k);
    const size_t ldb8i     = lda8i;
    const size_t ldc32i    = oz2::calc_ld32i(m);
    const size_t sizeA     = lda8i * ldc32i;
    const size_t sizeB     = ldb8i * n;
    const size_t sizeC     = ldc32i * n;
    const size_t size_vecA = oz2::calc_sizevec(m);
    const size_t size_vecB = oz2::calc_sizevec(n);

    size_t total_size = 0;
    total_size += sizeof(int8_t) * (sizeA + sizeB) * num_moduli;
    total_size += sizeof(uint8_t) * sizeC * num_moduli;
    total_size += sizeof(int32_t) * sizeC;
    total_size += sizeof(int16_t) * (size_vecA + size_vecB);

    return total_size;
}

template <typename T>
__inline__ std::vector<double> gemm_(cublasHandle_t handle,        // Handle to the cuBLAS library context
                                     const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
                                     const cublasOperation_t op_B, // CUBLAS_OP_N or CUBLAS_OP_T
                                     const size_t m,               // Number of rows of C
                                     const size_t n,               // Number of columns of C
                                     const size_t k,               // Inner dimension <= 2^17
                                     const T *alpha,               // Scaling factor for op(A)*op(B)
                                     const T *const A,             // 1-D device array of dimensions lda*k (CUBLAS_OP_N) or lda*m (CUBLAS_OP_T)
                                     const size_t lda,             // Leading dimension of A
                                     const T *const B,             // 1-D device array of dimensions ldb*n (CUBLAS_OP_N) or ldb*k (CUBLAS_OP_T)
                                     const size_t ldb,             // Leading dimension of B
                                     const T *beta,                // Scaling factor for C
                                     T *const C,                   // 1-D device array of dimensions ldc*n
                                     const size_t ldc,             // Leading dimension of C
                                     const unsigned num_moduli,    // #moduli, 2 <= num_moduli <= 20
                                     const bool fastmode,          // false (accurate mode) or true (fast mode)
                                     void *const work)             // workspace allocated in advance
{
    //------------------------------
    // timer
    //------------------------------
    std::chrono::system_clock::time_point time_stamp;
    std::vector<double> timer(4, 0.0);

    //------------------------------
    // set constants
    //------------------------------
    const size_t lda8i       = oz2::calc_ld8i(k);
    const size_t ldb8i       = lda8i;
    const size_t ldc32i      = oz2::calc_ld32i(m);
    const size_t sizeA       = lda8i * ldc32i;
    const size_t sizeB       = ldb8i * n;
    const size_t sizeC       = ldc32i * n;
    const size_t size_vecA   = oz2::calc_sizevec(m);
    const unsigned table_idx = num_moduli - 2;
    constexpr int32_t one    = 1;
    constexpr int32_t zero   = 0;

    oz2::grid_invscal   = (m * n + oz2::threads_invscal - 1) / oz2::threads_invscal;
    oz2::grid_conv32i8u = ((sizeC >> 2) + oz2::threads_conv32i8u - 1) / oz2::threads_conv32i8u;

    bool is_numM_1;
    if constexpr (std::is_same_v<T, double>) {
        is_numM_1 = oz2_table::numM[table_idx] == 1;
        if (is_numM_1) {
            cudaMemcpyToSymbol(oz2_table::NMi_dev, &oz2_table::NMi_1[table_idx][0], num_moduli * sizeof(double));
        } else {
            cudaMemcpyToSymbol(oz2_table::NMi_dev, &oz2_table::NMi_2[num_moduli - 8][0][0], 2 * num_moduli * sizeof(double));
        }
        cudaMemcpyToSymbol(oz2_table::moduli_dev, oz2_table::moduli, num_moduli * sizeof(oz2_table::tab_t<double>));
    } else {
        is_numM_1 = true;
        cudaMemcpyToSymbol(oz2_table::NMi_dev, &oz2_table::NMi_1[table_idx][0], num_moduli * sizeof(double));
        cudaMemcpyToSymbol(oz2_table::modulif_dev, oz2_table::modulif, num_moduli * sizeof(oz2_table::tab_t<float>));
    }

    //------------------------------
    // set workspace (16byte align)
    //------------------------------
    int8_t *A8i   = reinterpret_cast<int8_t *>(work);                      // lda8i*m*sizeod(int8_t)*num_moduli
    int8_t *B8i   = A8i + sizeA * num_moduli;                              // ldb8i*n*sizeod(int8_t)*num_moduli
    uint8_t *C8u  = reinterpret_cast<uint8_t *>(B8i + sizeB * num_moduli); // (m*n+15)/16*16*sizeof(uint8_t)*num_moduli
    int32_t *C32i = reinterpret_cast<int32_t *>(C8u + sizeC * num_moduli); // (m*n+15)/16*16*sizeof(int32_t)
    int16_t *sftA = reinterpret_cast<int16_t *>(C32i + sizeC);             // (m+15)/16*16*sizeof(int16_t)
    int16_t *sftB = sftA + size_vecA;                                      // (n+15)/16*16*sizeof(int16_t)

    //------------------------------
    // Scaling
    // A =: diag(2^sftA) * A', A' is integer
    // B =: B' * diag(2^sftB), B' is integer
    // Then, calculating mod for all moduli
    // A8i := mod(A', modulus[i]) - 128 (-128 <= A8i <= 127)
    // B8i := mod(B', modulus[i]) - 128 (-128 <= A8i <= 127)
    //------------------------------
    oz2::timing(time_stamp);
    if (fastmode) {
        oz2::vecnorm::scaling<T>(op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, sizeA, sftA, B8i, ldb8i, ldb8i * n, sftB, table_idx);
    } else {
        oz2::int8tc::scaling<T>(handle, op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, sizeA, sftA, B8i, ldb8i, ldb8i * n, sftB, C32i, ldc32i, table_idx);
    }
    oz2::timing(time_stamp, timer[0]);

    for (unsigned i = 0; i < num_moduli; ++i) {
        //-----------------------------
        // Error-free matrix multiplication
        // C32i := A8i*B8i
        //------------------------------
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, ldc32i, n, lda8i, &one, A8i + i * sizeA, CUDA_R_8I, lda8i, B8i + i * sizeB, CUDA_R_8I, ldb8i, &zero, C32i, CUDA_R_32I, ldc32i, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
        oz2::timing(time_stamp, timer[1]);

        //------------------------------
        // Calculating mod
        // C8u[i] := mod(C32i, modulus[i]) >= 0
        //------------------------------
        oz2::conv_32i_2_8u(i, sizeC, C32i, C8u + i * sizeC);
        oz2::timing(time_stamp, timer[2]);
    }

    //------------------------------
    // Accumulation and Inverse scaling
    // C64f = sum(Ni*Mi*C8u[i]),
    //  where
    //      Mi := M/modulus[i],
    //      M := prod(modulus[all]),
    //      mod(Ni*Mi, modulus[i]) == 1.
    // C := C64f - round(C64f/M)*M
    // C := diag(2^-sftA) * C * diag(2^-sftB)
    //------------------------------
    oz2::inverse_scaling<T>(is_numM_1, num_moduli, m, n, C8u, ldc32i, sizeC, C, ldc, sftA, sftB, *alpha, *beta);
    oz2::timing(time_stamp, timer[3]);

    return timer;
}

template <> std::vector<double> gemm<double>(cublasHandle_t handle,
                                             const cublasOperation_t op_A,
                                             const cublasOperation_t op_B,
                                             const size_t m,
                                             const size_t n,
                                             const size_t k,
                                             const double *alpha,
                                             const double *const A,
                                             const size_t lda,
                                             const double *const B,
                                             const size_t ldb,
                                             const double *beta,
                                             double *const C,
                                             const size_t ldc,
                                             const unsigned num_moduli,
                                             const bool fastmode,
                                             void *const work) //
{ return gemm_<double>(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, fastmode, work); }
template <> std::vector<double> gemm<float>(cublasHandle_t handle,
                                            const cublasOperation_t op_A,
                                            const cublasOperation_t op_B,
                                            const size_t m,
                                            const size_t n,
                                            const size_t k,
                                            const float *alpha,
                                            const float *const A,
                                            const size_t lda,
                                            const float *const B,
                                            const size_t ldb,
                                            const float *beta,
                                            float *const C,
                                            const size_t ldc,
                                            const unsigned num_moduli,
                                            const bool fastmode,
                                            void *const work) //
{ return gemm_<float>(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, fastmode, work); }

} // namespace gemmul8
