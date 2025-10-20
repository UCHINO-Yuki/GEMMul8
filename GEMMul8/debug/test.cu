#include "gemmul8.hpp"
#include "matrixmarketio.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>

void test(char trans_a, char trans_b, int moduli, bool fastmode) {

    std::cout << ((trans_a == 'N') ? "N" : "T") << ","
              << ((trans_b == 'N') ? "N" : "T") << ","
              << moduli << ","
              << int(fastmode) << ","
              << std::endl;

    Matrix<double> A;
    Matrix<double> B;
    Matrix<double> C;

    std::string matname = "problem";
    load_matrix("mats/" + matname + "_A.mtx", COORDINATE, A);
    load_matrix("mats/" + matname + "_B.mtx", COORDINATE, B);

    cublasOperation_t TRANSA = (trans_a == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t TRANSB = (trans_b == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasHandle_t cublasH   = NULL;
    cudaStream_t stream      = NULL;
    const int M              = A.sz_rows;
    const int N              = B.sz_cols;
    const int K              = A.sz_cols;
    const double ALPHA       = 1.0;
    const double BETA        = 0.0;
    const int LDA            = (trans_a == 'N') ? M : K;
    const int LDB            = (trans_b == 'N') ? K : N;
    const int LDC            = M;
    const size_t LWORK       = gemmul8::workSize(M, N, K, moduli);

    const double *a = A.data;
    const double *b = B.data;
    double *dat_A   = nullptr;
    double *dat_B   = nullptr;
    double *dat_C   = nullptr;
    double *dat_Cg  = nullptr;
    void *WORK      = NULL;
    double *c       = new double[M * N];
    double *cg      = new double[M * N];

    cublasCreate(&cublasH);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasSetStream(cublasH, stream);

    cudaMalloc(&dat_A, sizeof(double) * M * K);
    cudaMalloc(&dat_B, sizeof(double) * K * N);
    cudaMalloc(&dat_C, sizeof(double) * M * N);

    cudaMalloc(&dat_Cg, sizeof(double) * M * N);
    cudaMalloc(&WORK, LWORK);

    cudaMemcpy(dat_A, a, sizeof(double) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dat_B, b, sizeof(double) * K * N, cudaMemcpyHostToDevice);

    cudaStreamSynchronize(stream);

    cublasDgemm(cublasH, TRANSA, TRANSB, M, N, K, &ALPHA, dat_A, LDA, dat_B, LDB, &BETA, dat_C, LDC);

    cudaStreamSynchronize(stream);

    gemmul8::gemm(cublasH, TRANSA, TRANSB, M, N, K, &ALPHA, dat_A, LDA, dat_B, LDB, &BETA, dat_Cg, LDC, moduli, fastmode, WORK);

    cudaStreamSynchronize(stream);

    cudaMemcpy(c, dat_C, sizeof(double) * M * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(cg, dat_Cg, sizeof(double) * M * N, cudaMemcpyDeviceToHost);

    cudaStreamSynchronize(stream);

    cudaFree(dat_A);
    cudaFree(dat_B);
    cudaFree(dat_C);
    cudaFree(dat_Cg);
    cudaFree(WORK);

    cublasDestroy(cublasH);
    cudaStreamDestroy(stream);

    double err_chk = 0.;
    for (int i = 0; i < M * N; i++) {
        err_chk += std::pow(c[i] - cg[i], 2);
    }

    std::cout << "(M,N,K): " << M << ", " << N << ", " << K << std::endl;
    std::cout << "L2: " << err_chk << std::endl;
    if (err_chk > 1e-4) {
        std::cout << "L2 Error Too High!" << std::endl;
    }

    delete[] c;
    delete[] cg;
}

int main() {
    test('N', 'N', 20, false);
    test('N', 'T', 20, false);
    test('T', 'N', 20, false);
    test('T', 'T', 20, false);

    test('N', 'N', 20, true);
    test('N', 'T', 20, true);
    test('T', 'N', 20, true);
    test('T', 'T', 20, true);
    return 0;
}
