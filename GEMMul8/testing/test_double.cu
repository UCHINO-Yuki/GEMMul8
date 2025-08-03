#include "../include/gemmul8.hpp"
#include "eval.hpp"
#include "getWatt.hpp"
#include "make_matrix.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <nvml.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define AVERAGE 100
#define SEED    123456
#define PHI     0.5, 1, 2, 3, 4
// #define NUM_MODULI 14, 15, 16
#define NUM_MODULI 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
#if defined(GPU_MEM_MB) && GPU_MEM_MB >= 21000
    // more than or equal to 24GB
    #define SIZE 1024, 2048, 4096, 8192, 16384
#else
    #define SIZE 1024, 2048, 4096, 8192
#endif

#if defined(ozIMMU_EF_FLAG)
    #include "ozimmu/ozimmu.hpp"
std::vector<mtk::ozimmu::compute_mode_t> mode_list{
    mtk::ozimmu::fp64_int8_6,
    mtk::ozimmu::fp64_int8_7,
    mtk::ozimmu::fp64_int8_8,
    mtk::ozimmu::fp64_int8_9,
    mtk::ozimmu::fp64_int8_10,
    mtk::ozimmu::fp64_int8_11,
    mtk::ozimmu::fp64_int8_12};
    #define ozimmu_MIN_SPLIT 6
    #define ozimmu_MAX_SPLIT mtk::ozimmu::fp64_int8_12
#endif

std::string getDeviceName() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::string deviceName = deviceProp.name;

    for (char &c : deviceName) {
        if (c == ' ' || c == '/' || c == '\\') {
            c = '_';
        }
    }
    return deviceName;
}

std::string getCurrentDateTime(std::chrono::system_clock::time_point &now) {
    now                  = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time), "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}

void accuracy_check(std::string &deviceName, std::string &dateTime) {
    std::string fileName = "oz2_results_d_accuracy_" + deviceName + "_" + dateTime + ".csv";
    std::ofstream outFile(fileName);
    outFile << std::scientific;
    std::cout << std::scientific;
    cublasHandle_t handle;
    cublasCreate(&handle);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    std::vector<double> phi_list{PHI};
    std::vector<size_t> k_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};
    const size_t m = 1024;
    const size_t n = 1024;

    //--------------------
    // workspace
    //--------------------
    const size_t k_max          = *max_element(begin(k_list), end(k_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double *work_cpu            = new double[m * n * 3];
    size_t worksize             = gemmul8::workSize(m, n, k_max, num_moduli_max);
    void *work_gpu;
    cudaMalloc(&work_gpu, (m * k_max + k_max * n + m * n * 2) * sizeof(double));
    cudaDeviceSynchronize();
    void *work_gemm;
    cudaMalloc(&work_gemm, worksize);
    cudaDeviceSynchronize();

    outFile << "phi,function,";
    std::cout << "phi,function,";
    for (auto &moduli : num_moduli_list) {
        outFile << moduli << ",";
        std::cout << moduli << ",";
    }
    outFile << std::endl;
    std::cout << std::endl;

    for (auto &phi : phi_list) {
        for (auto &k : k_list) {
            double *cpuC  = work_cpu;
            double *cpuC1 = cpuC + m * n;
            double *cpuC2 = cpuC1 + m * n;
            double *devA  = reinterpret_cast<double *>(work_gpu);
            double *devB  = devA + m * k;
            double *devC  = devB + k * n;
            double *devC1 = devC;
            double *devC2 = devC1 + m * n;
            double errmax, errmed;
            std::vector<double> err_OS2_fast;
            std::vector<double> err_OS2_accu;

            //--------------------
            // generate matrices
            //--------------------
            makemat::randmat<double>(m, k, devA, phi, seed);
            makemat::randmat<double>(k, n, devB, phi, seed);

            //--------------------
            // C1+C2 := A*B (double-double arithmetic)
            //--------------------
            eval::dd_gpu::simple_gemm(m, n, k, devA, devB, devC1, devC2);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC1, devC1, m * n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(cpuC2, devC2, m * n * sizeof(double), cudaMemcpyDeviceToHost);

            //--------------------
            // C := A*B by FP64
            //--------------------
            double alpha = 1.0;
            double beta  = 0.0;
            cudaDeviceSynchronize();
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, CUDA_R_64F, m, devB, CUDA_R_64F, k, &beta, devC, CUDA_R_64F, m, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, errmax, errmed);

            outFile << phi << ",DGEMM (k=" + std::to_string(k) + "),";
            std::cout << phi << ",DGEMM (k=" + std::to_string(k) + "),";
            for (int i = 0; i < num_moduli_list.size(); ++i) {
                outFile << errmax << ",";
                std::cout << errmax << ",";
            }
            outFile << std::endl;
            std::cout << std::endl;

            //--------------------
            // C := A*B by ozIMMU_EF
            //--------------------
#ifdef ozIMMU_EF_FLAG
            mtk::ozimmu::gemm_list_t fp64in_gemm;
            fp64in_gemm.push_back({mtk::ozimmu::op_n, mtk::ozimmu::op_n, m, n, k, mtk::ozimmu::real, ozimmu_MAX_SPLIT});
            mtk::ozimmu::handle_t ozimmu_handle;
            mtk::ozimmu::create(&ozimmu_handle);
            mtk::ozimmu::reallocate_working_memory(ozimmu_handle, fp64in_gemm);
            int slice = ozimmu_MIN_SPLIT;
            for (auto &mode : mode_list) {
                cudaDeviceSynchronize();
                mtk::ozimmu::gemm(ozimmu_handle,
                                  mtk::ozimmu::op_n,
                                  mtk::ozimmu::op_n,
                                  m,
                                  n,
                                  k,
                                  &alpha,
                                  devA,
                                  m,
                                  devB,
                                  k,
                                  &beta,
                                  devC,
                                  m,
                                  mode,
                                  mtk::ozimmu::real);
                cudaDeviceSynchronize();
                cudaMemcpy(cpuC, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, errmax, errmed);

                outFile << phi << ",ozIMMU_EF-" + std::to_string(slice) + "(k=" + std::to_string(k) + "),";
                std::cout << phi << ",ozIMMU_EF-" + std::to_string(slice) + "(k=" + std::to_string(k) + "),";
                for (int i = 0; i < num_moduli_list.size(); ++i) {
                    outFile << errmax << ",";
                    std::cout << errmax << ",";
                }
                outFile << std::endl;
                std::cout << std::endl;

                slice++;
            }
            mtk::ozimmu::destroy(ozimmu_handle);
#endif

            //--------------------
            // C := A*B by ozaki-scheme2
            //--------------------
            outFile << phi << ",OS2-fast (k=" + std::to_string(k) + "),";
            std::cout << phi << ",OS2-fast (k=" + std::to_string(k) + "),";
            for (auto &num_moduli : num_moduli_list) {
                std::vector<double> timestmp(4, 0);
                cudaDeviceSynchronize();
                timestmp = gemmul8::gemm<double>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, true, work_gemm);
                cudaDeviceSynchronize();
                cudaMemcpy(cpuC, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, errmax, errmed);
                outFile << errmax << ",";
                std::cout << errmax << ",";
            }
            outFile << std::endl;
            std::cout << std::endl;

            //--------------------
            // C := A*B by ozaki-scheme2
            //--------------------
            outFile << phi << ",OS2-accu (k=" + std::to_string(k) + "),";
            std::cout << phi << ",OS2-accu (k=" + std::to_string(k) + "),";
            for (auto &num_moduli : num_moduli_list) {
                std::vector<double> timestmp(4);
                cudaDeviceSynchronize();
                timestmp = gemmul8::gemm<double>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, false, work_gemm);
                cudaDeviceSynchronize();
                cudaMemcpy(cpuC, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, errmax, errmed);
                outFile << errmax << ",";
                std::cout << errmax << ",";
            }
            outFile << std::endl;
            std::cout << std::endl;
        }
    }

    delete[] work_cpu;
    cudaFree(work_gpu);
    cudaFree(work_gemm);
    cublasDestroy(handle);
    outFile.close();
}

void time_check(std::string &deviceName, std::string &dateTime) {
    std::string fileName = "oz2_results_d_time_" + deviceName + "_" + dateTime + ".csv";
    std::ofstream outFile(fileName);
    outFile << std::scientific;
    std::cout << std::scientific;
    outFile << "phi,m,n,k,"
            << "function,"
            << "relerr_max,relerr_med,"
            << "TFLOPS,"
            << "total_time [sec],"
            << "conv_64f_2_8i,"
            << "cublasGemmEx,"
            << "conv_32i_2_8u,"
            << "inverse_scaling,"
            << std::endl;
    std::cout << "phi,m,n,k,"
              << "function,"
              << "relerr_max,relerr_med,"
              << "TFLOPS,"
              << "total_time [sec],"
              << "conv_64f_2_8i,"
              << "cublasGemmEx,"
              << "conv_32i_2_8u,"
              << "inverse_scaling,"
              << std::endl;
    cublasHandle_t handle;
    cublasCreate(&handle);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    const double phi        = 0.5;
    std::vector<size_t> n_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};
    const int itermax = AVERAGE;

    //--------------------
    // workspace
    //--------------------
    const size_t n_max          = *max_element(begin(n_list), end(n_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double *work_cpu            = new double[n_max * n_max * 3];
    size_t worksize             = gemmul8::workSize(n_max, n_max, n_max, num_moduli_max);
    void *work_gpu;
    cudaMalloc(&work_gpu, n_max * n_max * sizeof(double) * ((num_moduli_max >= 5) ? 3 : 4));
    cudaDeviceSynchronize();
    void *work_gemm;
    cudaMalloc(&work_gemm, worksize);
    cudaDeviceSynchronize();

    for (auto &n : n_list) {
        size_t m           = n;
        size_t k           = n;
        double *cpuC       = work_cpu;
        double *cpuC1      = cpuC + m * n;
        double *cpuC2      = cpuC1 + m * n;
        double *devA       = reinterpret_cast<double *>(work_gpu);
        double *devB       = devA + m * k;
        double *devC       = devB + k * n;
        double *devC1      = devC;
        double *devC2      = (num_moduli_max >= 5) ? (reinterpret_cast<double *>(work_gemm)) : (devC1 + m * n);
        const size_t lda8i = ((k + 15) >> 4) << 4;
        const size_t ldb8i = lda8i;
        int8_t *A8i        = reinterpret_cast<int8_t *>(work_gemm);
        int8_t *B8i        = A8i + lda8i * m;
        int32_t *C32i      = reinterpret_cast<int32_t *>(B8i + ldb8i * n);
        double maxerr = 0.0, mederr = 0.0;
        double time = 0.0;
        std::chrono::system_clock::time_point start, stop;

        //--------------------
        // generate matrices
        //--------------------
        makemat::randmat<double>(m, k, devA, phi, seed);
        makemat::randmat<double>(k, n, devB, phi, seed);

        //--------------------
        // C1+C2 := A*B (double-double arithmetic)
        //--------------------
        eval::dd_gpu::simple_gemm(m, n, k, devA, devB, devC1, devC2);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuC1, devC1, m * n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpuC2, devC2, m * n * sizeof(double), cudaMemcpyDeviceToHost);

        //--------------------
        // C := A*B (int8-TC)
        //--------------------
        makemat::ones(lda8i * m + ldb8i * n, A8i);
        int32_t ialpha = 1;
        int32_t ibeta  = 0;
        cudaDeviceSynchronize();
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, lda8i, &ialpha, A8i, CUDA_R_8I, lda8i, B8i, CUDA_R_8I, ldb8i, &ibeta, C32i, CUDA_R_32I, m, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        time = 0.0;
        for (int iter = 0; iter < itermax; ++iter) {
            cudaDeviceSynchronize();
            start = std::chrono::system_clock::now();
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, lda8i, &ialpha, A8i, CUDA_R_8I, lda8i, B8i, CUDA_R_8I, ldb8i, &ibeta, C32i, CUDA_R_32I, m, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            stop = std::chrono::system_clock::now();
            time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }
        time = time / itermax * 1.e-9;

        outFile << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
        outFile << "," << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                << "," << "," << "," << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
        std::cout << "," << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                  << "," << "," << "," << "," << std::endl;

        //--------------------
        // C := A*B by FP64
        //--------------------
        double alpha = 1.0;
        double beta  = 0.0;
        cudaDeviceSynchronize();
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, CUDA_R_64F, m, devB, CUDA_R_64F, k, &beta, devC, CUDA_R_64F, m, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuC, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

        time = 0.0;
        for (int iter = 0; iter < itermax; ++iter) {
            cudaDeviceSynchronize();
            start = std::chrono::system_clock::now();
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, CUDA_R_64F, m, devB, CUDA_R_64F, k, &beta, devC, CUDA_R_64F, m, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            stop = std::chrono::system_clock::now();
            time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }
        time = time / itermax * 1.e-9;

        outFile << phi << "," << m << "," << n << "," << k << "," << "DGEMM" << ",";
        outFile << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                << "," << "," << "," << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "DGEMM" << ",";
        std::cout << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                  << "," << "," << "," << "," << std::endl;

        //--------------------
        // C := A*B by ozIMMU_EF
        //--------------------
#ifdef ozIMMU_EF_FLAG
        mtk::ozimmu::gemm_list_t fp64in_gemm;
        fp64in_gemm.push_back({mtk::ozimmu::op_n, mtk::ozimmu::op_n, m, n, k, mtk::ozimmu::real, ozimmu_MAX_SPLIT});
        mtk::ozimmu::handle_t ozimmu_handle;
        mtk::ozimmu::create(&ozimmu_handle);
        mtk::ozimmu::reallocate_working_memory(ozimmu_handle, fp64in_gemm);
        int slice = ozimmu_MIN_SPLIT;
        for (auto &mode : mode_list) {
            cudaDeviceSynchronize();
            mtk::ozimmu::gemm(ozimmu_handle,
                              mtk::ozimmu::op_n,
                              mtk::ozimmu::op_n,
                              m,
                              n,
                              k,
                              &alpha,
                              devA,
                              m,
                              devB,
                              k,
                              &beta,
                              devC,
                              m,
                              mode,
                              mtk::ozimmu::real);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                cudaDeviceSynchronize();
                start = std::chrono::system_clock::now();
                mtk::ozimmu::gemm(ozimmu_handle,
                                  mtk::ozimmu::op_n,
                                  mtk::ozimmu::op_n,
                                  m,
                                  n,
                                  k,
                                  &alpha,
                                  devA,
                                  m,
                                  devB,
                                  k,
                                  &beta,
                                  devC,
                                  m,
                                  mode,
                                  mtk::ozimmu::real);
                cudaDeviceSynchronize();
                stop = std::chrono::system_clock::now();
                time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
            }
            time = time / itermax * 1.e-9;

            outFile << phi << "," << m << "," << n << "," << k << "," << "ozIMMU_EF-" << slice << ",";
            outFile << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                    << "," << "," << "," << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "ozIMMU_EF-" << slice << ",";
            std::cout << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                      << "," << "," << "," << "," << std::endl;

            slice++;
        }
        mtk::ozimmu::destroy(ozimmu_handle);
#endif

        //--------------------
        // C := A*B by ozaki-scheme2
        //--------------------
        for (auto &num_moduli : num_moduli_list) {
            std::vector<double> times(4, 0);
            std::vector<double> timestmp(4, 0);

            cudaDeviceSynchronize();
            timestmp = gemmul8::gemm<double>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, true, work_gemm);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                cudaDeviceSynchronize();
                start    = std::chrono::system_clock::now();
                timestmp = gemmul8::gemm<double>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, true, work_gemm);
                cudaDeviceSynchronize();
                stop = std::chrono::system_clock::now();
                time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
                for (int j = 0; j < 4; ++j) times[j] += timestmp[j];
            }
            time = time / itermax * 1.e-9;
            for (int j = 0; j < 4; ++j) times[j] = times[j] / itermax * 1.e-9;

            outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
            outFile << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                    << times[0] << "," << times[1] << "," << times[2] << "," << times[3] << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
            std::cout << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                      << times[0] << "," << times[1] << "," << times[2] << "," << times[3] << "," << std::endl;
        }

        //--------------------
        // C := A*B by ozaki-scheme2
        //--------------------
        for (auto &num_moduli : num_moduli_list) {
            std::vector<double> times(4, 0);
            std::vector<double> timestmp(4);

            cudaDeviceSynchronize();
            timestmp = gemmul8::gemm<double>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, false, work_gemm);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                cudaDeviceSynchronize();
                start    = std::chrono::system_clock::now();
                timestmp = gemmul8::gemm<double>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, false, work_gemm);
                cudaDeviceSynchronize();
                stop = std::chrono::system_clock::now();
                time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
                for (int j = 0; j < 4; ++j) times[j] += timestmp[j];
            }
            time = time / itermax * 1.e-9;
            for (int j = 0; j < 4; ++j) times[j] = times[j] / itermax * 1.e-9;

            outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
            outFile << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                    << times[0] << "," << times[1] << "," << times[2] << "," << times[3] << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
            std::cout << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                      << times[0] << "," << times[1] << "," << times[2] << "," << times[3] << "," << std::endl;
        }
    }

    delete[] work_cpu;
    cudaFree(work_gpu);
    cudaFree(work_gemm);
    cublasDestroy(handle);
    outFile.close();
}

void watt_check(std::string &deviceName, std::string &dateTime) {
    std::string fileName = "oz2_results_d_watt_" + deviceName + "_" + dateTime + ".csv";
    std::ofstream outFile(fileName);
    outFile << std::scientific;
    std::cout << std::scientific;
    outFile << "phi,m,n,k,"
            << "function,"
            << "relerr_max,relerr_med,"
            << "watt,"
            << "GFLOPS/watt,"
            << std::endl;
    std::cout << "phi,m,n,k,"
              << "function,"
              << "relerr_max,relerr_med,"
              << "watt,"
              << "GFLOPS/watt,"
              << std::endl;
    cublasHandle_t handle;
    cublasCreate(&handle);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    const double phi        = 0.5;
    std::vector<size_t> n_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};

    //--------------------
    // workspace
    //--------------------
    const size_t n_max          = *max_element(begin(n_list), end(n_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double *work_cpu            = new double[n_max * n_max * 3];
    size_t worksize             = gemmul8::workSize(n_max, n_max, n_max, num_moduli_max);
    void *work_gpu;
    cudaMalloc(&work_gpu, n_max * n_max * 5 * sizeof(double));
    cudaDeviceSynchronize();
    void *work_gemm;
    cudaMalloc(&work_gemm, worksize);
    cudaDeviceSynchronize();

    for (auto &n : n_list) {
        size_t m           = n;
        size_t k           = n;
        double *cpuC       = work_cpu;
        double *cpuC1      = cpuC + m * n;
        double *cpuC2      = cpuC1 + m * n;
        double *devA       = reinterpret_cast<double *>(work_gpu);
        double *devB       = devA + m * k;
        double *devC       = devB + k * n;
        double *devC1      = devC + m * n;
        double *devC2      = devC1 + m * n;
        const size_t lda8i = ((k + 15) >> 4) << 4;
        const size_t ldb8i = lda8i;
        int8_t *A8i        = reinterpret_cast<int8_t *>(work_gemm);
        int8_t *B8i        = A8i + lda8i * m;
        int32_t *C32i      = reinterpret_cast<int32_t *>(B8i + ldb8i * n);
        double maxerr = 0.0, mederr = 0.0;

        //--------------------
        // generate matrices
        //--------------------
        makemat::randmat<double>(m, k, devA, phi, seed);
        makemat::randmat<double>(k, n, devB, phi, seed);

        //--------------------
        // C1+C2 := A*B (double-double arithmetic)
        //--------------------
        eval::dd_gpu::simple_gemm(m, n, k, devA, devB, devC1, devC2);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuC1, devC1, m * n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpuC2, devC2, m * n * sizeof(double), cudaMemcpyDeviceToHost);

        //--------------------
        // C := A*B (int8-TC)
        //--------------------
        makemat::ones(lda8i * m + ldb8i * n, A8i);
        int32_t ialpha          = 1;
        int32_t ibeta           = 0;
        std::vector<double> res = getWatt::getWatt(
            [&]() {
                cublasGemmEx(handle,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             m,
                             n,
                             lda8i,
                             &ialpha,
                             A8i,
                             CUDA_R_8I,
                             lda8i,
                             B8i,
                             CUDA_R_8I,
                             ldb8i,
                             &ibeta,
                             C32i,
                             CUDA_R_32I,
                             m,
                             CUBLAS_COMPUTE_32I,
                             CUBLAS_GEMM_DEFAULT);
            },
            m,
            n,
            k);

        outFile << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
        outFile << "," << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
        std::cout << "," << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;

        //--------------------
        // C := A*B by FP64
        //--------------------
        double alpha = 1.0;
        double beta  = 0.0;
        cudaDeviceSynchronize();
        res = getWatt::getWatt(
            [&]() {
                cublasGemmEx(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             m,
                             n,
                             k,
                             &alpha,
                             devA,
                             CUDA_R_64F,
                             m,
                             devB,
                             CUDA_R_64F,
                             k,
                             &beta,
                             devC,
                             CUDA_R_64F,
                             m,
                             CUBLAS_COMPUTE_64F,
                             CUBLAS_GEMM_DEFAULT);
            },
            m,
            n,
            k);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuC, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

        outFile << phi << "," << m << "," << n << "," << k << "," << "DGEMM" << ",";
        outFile << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "DGEMM" << ",";
        std::cout << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;

        //--------------------
        // C := A*B by ozIMMU_EF
        //--------------------
#ifdef ozIMMU_EF_FLAG
        mtk::ozimmu::gemm_list_t fp64in_gemm;
        fp64in_gemm.push_back({mtk::ozimmu::op_n, mtk::ozimmu::op_n, m, n, k, mtk::ozimmu::real, ozimmu_MAX_SPLIT});
        mtk::ozimmu::handle_t ozimmu_handle;
        mtk::ozimmu::create(&ozimmu_handle);
        mtk::ozimmu::reallocate_working_memory(ozimmu_handle, fp64in_gemm);
        int slice = ozimmu_MIN_SPLIT;
        for (auto &mode : mode_list) {
            cudaDeviceSynchronize();
            res = getWatt::getWatt(
                [&]() {
                    mtk::ozimmu::gemm(ozimmu_handle,
                                      mtk::ozimmu::op_n,
                                      mtk::ozimmu::op_n,
                                      m,
                                      n,
                                      k,
                                      &alpha,
                                      devA,
                                      m,
                                      devB,
                                      k,
                                      &beta,
                                      devC,
                                      m,
                                      mode,
                                      mtk::ozimmu::real);
                },
                m,
                n,
                k);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            outFile << phi << "," << m << "," << n << "," << k << "," << "ozIMMU_EF-" << slice << ",";
            outFile << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "ozIMMU_EF-" << slice << ",";
            std::cout << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;

            slice++;
        }
        mtk::ozimmu::destroy(ozimmu_handle);
#endif

        //--------------------
        // C := A*B by ozaki-scheme2
        //--------------------
        for (auto &num_moduli : num_moduli_list) {
            cudaDeviceSynchronize();

            res = getWatt::getWatt(
                [&]() {
                    gemmul8::gemm<double>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, true, work_gemm);
                },
                m,
                n,
                k);

            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
            outFile << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
            std::cout << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
        }

        //--------------------
        // C := A*B by ozaki-scheme2
        //--------------------
        for (auto &num_moduli : num_moduli_list) {
            cudaDeviceSynchronize();

            res = getWatt::getWatt(
                [&]() {
                    gemmul8::gemm<double>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, false, work_gemm);
                },
                m,
                n,
                k);

            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
            outFile << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
            std::cout << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
        }
    }

    delete[] work_cpu;
    cudaFree(work_gpu);
    cudaFree(work_gemm);
    cublasDestroy(handle);
    outFile.close();
}

void time_check_rect(std::string &deviceName, std::string &dateTime) {
    std::string fileName = "oz2_results_d_time-rect_" + deviceName + "_" + dateTime + ".csv";
    std::ofstream outFile(fileName);
    outFile << std::scientific;
    std::cout << std::scientific;
    outFile << "phi,m,n,k,"
            << "function,"
            << "relerr_max,relerr_med,"
            << "TFLOPS,"
            << "total_time [sec],"
            << "conv_64f_2_8i,"
            << "cublasGemmEx,"
            << "conv_32i_2_8u,"
            << "inverse_scaling,"
            << std::endl;
    std::cout << "phi,m,n,k,"
              << "function,"
              << "relerr_max,relerr_med,"
              << "TFLOPS,"
              << "total_time [sec],"
              << "conv_64f_2_8i,"
              << "cublasGemmEx,"
              << "conv_32i_2_8u,"
              << "inverse_scaling,"
              << std::endl;
    cublasHandle_t handle;
    cublasCreate(&handle);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    const double phi        = 0.5;
    std::vector<size_t> n_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};
    const int itermax = AVERAGE;

    //--------------------
    // workspace
    //--------------------
    const size_t n_max          = *max_element(begin(n_list), end(n_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double *work_cpu            = new double[n_max * n_max * 3];
    size_t worksize             = gemmul8::workSize(n_max, n_max, n_max, num_moduli_max);
    void *work_gpu;
    cudaMalloc(&work_gpu, n_max * n_max * sizeof(double) * ((num_moduli_max >= 5) ? 3 : 4));
    cudaDeviceSynchronize();
    void *work_gemm;
    cudaMalloc(&work_gemm, worksize);
    cudaDeviceSynchronize();

    for (auto &n : n_list) {
        size_t m = n;

        for (auto &k : n_list) {
            double *cpuC       = work_cpu;
            double *cpuC1      = cpuC + m * n;
            double *cpuC2      = cpuC1 + m * n;
            double *devA       = reinterpret_cast<double *>(work_gpu);
            double *devB       = devA + m * k;
            double *devC       = devB + k * n;
            double *devC1      = devC;
            double *devC2      = (num_moduli_max >= 5) ? (reinterpret_cast<double *>(work_gemm)) : (devC1 + m * n);
            const size_t lda8i = ((k + 15) >> 4) << 4;
            const size_t ldb8i = lda8i;
            int8_t *A8i        = reinterpret_cast<int8_t *>(work_gemm);
            int8_t *B8i        = A8i + lda8i * m;
            int32_t *C32i      = reinterpret_cast<int32_t *>(B8i + ldb8i * n);
            double maxerr = 0.0, mederr = 0.0;
            double time = 0.0;
            std::chrono::system_clock::time_point start, stop;

            //--------------------
            // generate matrices
            //--------------------
            makemat::randmat<double>(m, k, devA, phi, seed);
            makemat::randmat<double>(k, n, devB, phi, seed);

            //--------------------
            // C1+C2 := A*B (double-double arithmetic)
            //--------------------
            eval::dd_gpu::simple_gemm(m, n, k, devA, devB, devC1, devC2);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC1, devC1, m * n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(cpuC2, devC2, m * n * sizeof(double), cudaMemcpyDeviceToHost);

            //--------------------
            // C := A*B (int8-TC)
            //--------------------
            makemat::ones(lda8i * m + ldb8i * n, A8i);
            int32_t ialpha = 1;
            int32_t ibeta  = 0;
            cudaDeviceSynchronize();
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, lda8i, &ialpha, A8i, CUDA_R_8I, lda8i, B8i, CUDA_R_8I, ldb8i, &ibeta, C32i, CUDA_R_32I, m, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                cudaDeviceSynchronize();
                start = std::chrono::system_clock::now();
                cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, lda8i, &ialpha, A8i, CUDA_R_8I, lda8i, B8i, CUDA_R_8I, ldb8i, &ibeta, C32i, CUDA_R_32I, m, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
                cudaDeviceSynchronize();
                stop = std::chrono::system_clock::now();
                time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
            }
            time = time / itermax * 1.e-9;

            outFile << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
            outFile << "," << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                    << "," << "," << "," << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
            std::cout << "," << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                      << "," << "," << "," << "," << std::endl;

            //--------------------
            // C := A*B by FP64
            //--------------------
            double alpha = 1.0;
            double beta  = 0.0;
            cudaDeviceSynchronize();
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, CUDA_R_64F, m, devB, CUDA_R_64F, k, &beta, devC, CUDA_R_64F, m, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                cudaDeviceSynchronize();
                start = std::chrono::system_clock::now();
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, CUDA_R_64F, m, devB, CUDA_R_64F, k, &beta, devC, CUDA_R_64F, m, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
                cudaDeviceSynchronize();
                stop = std::chrono::system_clock::now();
                time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
            }
            time = time / itermax * 1.e-9;

            outFile << phi << "," << m << "," << n << "," << k << "," << "DGEMM" << ",";
            outFile << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                    << "," << "," << "," << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "DGEMM" << ",";
            std::cout << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                      << "," << "," << "," << "," << std::endl;

            //--------------------
            // C := A*B by ozIMMU_EF
            //--------------------
#ifdef ozIMMU_EF_FLAG
            mtk::ozimmu::gemm_list_t fp64in_gemm;
            fp64in_gemm.push_back({mtk::ozimmu::op_n, mtk::ozimmu::op_n, m, n, k, mtk::ozimmu::real, ozimmu_MAX_SPLIT});
            mtk::ozimmu::handle_t ozimmu_handle;
            mtk::ozimmu::create(&ozimmu_handle);
            mtk::ozimmu::reallocate_working_memory(ozimmu_handle, fp64in_gemm);
            int slice = ozimmu_MIN_SPLIT;
            for (auto &mode : mode_list) {
                cudaDeviceSynchronize();
                mtk::ozimmu::gemm(ozimmu_handle,
                                  mtk::ozimmu::op_n,
                                  mtk::ozimmu::op_n,
                                  m,
                                  n,
                                  k,
                                  &alpha,
                                  devA,
                                  m,
                                  devB,
                                  k,
                                  &beta,
                                  devC,
                                  m,
                                  mode,
                                  mtk::ozimmu::real);
                cudaDeviceSynchronize();
                cudaMemcpy(cpuC, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

                time = 0.0;
                for (int iter = 0; iter < itermax; ++iter) {
                    cudaDeviceSynchronize();
                    start = std::chrono::system_clock::now();
                    mtk::ozimmu::gemm(ozimmu_handle,
                                      mtk::ozimmu::op_n,
                                      mtk::ozimmu::op_n,
                                      m,
                                      n,
                                      k,
                                      &alpha,
                                      devA,
                                      m,
                                      devB,
                                      k,
                                      &beta,
                                      devC,
                                      m,
                                      mode,
                                      mtk::ozimmu::real);
                    cudaDeviceSynchronize();
                    stop = std::chrono::system_clock::now();
                    time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
                }
                time = time / itermax * 1.e-9;

                outFile << phi << "," << m << "," << n << "," << k << "," << "ozIMMU_EF-" << slice << ",";
                outFile << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                        << "," << "," << "," << "," << std::endl;
                std::cout << phi << "," << m << "," << n << "," << k << "," << "ozIMMU_EF-" << slice << ",";
                std::cout << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                          << "," << "," << "," << "," << std::endl;

                slice++;
            }
            mtk::ozimmu::destroy(ozimmu_handle);
#endif

            //--------------------
            // C := A*B by ozaki-scheme2
            //--------------------
            for (auto &num_moduli : num_moduli_list) {
                std::vector<double> times(4, 0);
                std::vector<double> timestmp(4, 0);

                cudaDeviceSynchronize();
                timestmp = gemmul8::gemm<double>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, true, work_gemm);
                cudaDeviceSynchronize();
                cudaMemcpy(cpuC, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

                time = 0.0;
                for (int iter = 0; iter < itermax; ++iter) {
                    cudaDeviceSynchronize();
                    start    = std::chrono::system_clock::now();
                    timestmp = gemmul8::gemm<double>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, true, work_gemm);
                    cudaDeviceSynchronize();
                    stop = std::chrono::system_clock::now();
                    time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
                    for (int j = 0; j < 4; ++j) times[j] += timestmp[j];
                }
                time = time / itermax * 1.e-9;
                for (int j = 0; j < 4; ++j) times[j] = times[j] / itermax * 1.e-9;

                outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
                outFile << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                        << times[0] << "," << times[1] << "," << times[2] << "," << times[3] << "," << std::endl;
                std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
                std::cout << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                          << times[0] << "," << times[1] << "," << times[2] << "," << times[3] << "," << std::endl;
            }

            //--------------------
            // C := A*B by ozaki-scheme2
            //--------------------
            for (auto &num_moduli : num_moduli_list) {
                std::vector<double> times(4, 0);
                std::vector<double> timestmp(4);

                cudaDeviceSynchronize();
                timestmp = gemmul8::gemm<double>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, false, work_gemm);
                cudaDeviceSynchronize();
                cudaMemcpy(cpuC, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

                time = 0.0;
                for (int iter = 0; iter < itermax; ++iter) {
                    cudaDeviceSynchronize();
                    start    = std::chrono::system_clock::now();
                    timestmp = gemmul8::gemm<double>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, false, work_gemm);
                    cudaDeviceSynchronize();
                    stop = std::chrono::system_clock::now();
                    time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
                    for (int j = 0; j < 4; ++j) times[j] += timestmp[j];
                }
                time = time / itermax * 1.e-9;
                for (int j = 0; j < 4; ++j) times[j] = times[j] / itermax * 1.e-9;

                outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
                outFile << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                        << times[0] << "," << times[1] << "," << times[2] << "," << times[3] << "," << std::endl;
                std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
                std::cout << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                          << times[0] << "," << times[1] << "," << times[2] << "," << times[3] << "," << std::endl;
            }
        }
    }

    delete[] work_cpu;
    cudaFree(work_gpu);
    cudaFree(work_gemm);
    cublasDestroy(handle);
    outFile.close();
}

void watt_check_rect(std::string &deviceName, std::string &dateTime) {
    std::string fileName = "oz2_results_d_watt-rect_" + deviceName + "_" + dateTime + ".csv";
    std::ofstream outFile(fileName);
    outFile << std::scientific;
    std::cout << std::scientific;
    outFile << "phi,m,n,k,"
            << "function,"
            << "relerr_max,relerr_med,"
            << "watt,"
            << "GFLOPS/watt,"
            << std::endl;
    std::cout << "phi,m,n,k,"
              << "function,"
              << "relerr_max,relerr_med,"
              << "watt,"
              << "GFLOPS/watt,"
              << std::endl;
    cublasHandle_t handle;
    cublasCreate(&handle);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    const double phi        = 0.5;
    std::vector<size_t> n_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};

    //--------------------
    // workspace
    //--------------------
    const size_t n_max          = *max_element(begin(n_list), end(n_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double *work_cpu            = new double[n_max * n_max * 3];
    size_t worksize             = gemmul8::workSize(n_max, n_max, n_max, num_moduli_max);
    void *work_gpu;
    cudaMalloc(&work_gpu, n_max * n_max * 5 * sizeof(double));
    cudaDeviceSynchronize();
    void *work_gemm;
    cudaMalloc(&work_gemm, worksize);
    cudaDeviceSynchronize();

    for (auto &n : n_list) {
        size_t m = n;

        for (auto &k : n_list) {
            double *cpuC       = work_cpu;
            double *cpuC1      = cpuC + m * n;
            double *cpuC2      = cpuC1 + m * n;
            double *devA       = reinterpret_cast<double *>(work_gpu);
            double *devB       = devA + m * k;
            double *devC       = devB + k * n;
            double *devC1      = devC + m * n;
            double *devC2      = devC1 + m * n;
            const size_t lda8i = ((k + 15) >> 4) << 4;
            const size_t ldb8i = lda8i;
            int8_t *A8i        = reinterpret_cast<int8_t *>(work_gemm);
            int8_t *B8i        = A8i + lda8i * m;
            int32_t *C32i      = reinterpret_cast<int32_t *>(B8i + ldb8i * n);
            double maxerr = 0.0, mederr = 0.0;

            //--------------------
            // generate matrices
            //--------------------
            makemat::randmat<double>(m, k, devA, phi, seed);
            makemat::randmat<double>(k, n, devB, phi, seed);

            //--------------------
            // C1+C2 := A*B (double-double arithmetic)
            //--------------------
            eval::dd_gpu::simple_gemm(m, n, k, devA, devB, devC1, devC2);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC1, devC1, m * n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(cpuC2, devC2, m * n * sizeof(double), cudaMemcpyDeviceToHost);

            //--------------------
            // C := A*B (int8-TC)
            //--------------------
            makemat::ones(lda8i * m + ldb8i * n, A8i);
            int32_t ialpha          = 1;
            int32_t ibeta           = 0;
            std::vector<double> res = getWatt::getWatt(
                [&]() {
                    cublasGemmEx(handle,
                                 CUBLAS_OP_T,
                                 CUBLAS_OP_N,
                                 m,
                                 n,
                                 lda8i,
                                 &ialpha,
                                 A8i,
                                 CUDA_R_8I,
                                 lda8i,
                                 B8i,
                                 CUDA_R_8I,
                                 ldb8i,
                                 &ibeta,
                                 C32i,
                                 CUDA_R_32I,
                                 m,
                                 CUBLAS_COMPUTE_32I,
                                 CUBLAS_GEMM_DEFAULT);
                },
                m,
                n,
                k);

            outFile << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
            outFile << "," << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
            std::cout << "," << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;

            //--------------------
            // C := A*B by FP64
            //--------------------
            double alpha = 1.0;
            double beta  = 0.0;
            cudaDeviceSynchronize();
            res = getWatt::getWatt(
                [&]() {
                    cublasGemmEx(handle,
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 m,
                                 n,
                                 k,
                                 &alpha,
                                 devA,
                                 CUDA_R_64F,
                                 m,
                                 devB,
                                 CUDA_R_64F,
                                 k,
                                 &beta,
                                 devC,
                                 CUDA_R_64F,
                                 m,
                                 CUBLAS_COMPUTE_64F,
                                 CUBLAS_GEMM_DEFAULT);
                },
                m,
                n,
                k);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuC, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            outFile << phi << "," << m << "," << n << "," << k << "," << "DGEMM" << ",";
            outFile << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "DGEMM" << ",";
            std::cout << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;

            //--------------------
            // C := A*B by ozIMMU_EF
            //--------------------
#ifdef ozIMMU_EF_FLAG
            mtk::ozimmu::gemm_list_t fp64in_gemm;
            fp64in_gemm.push_back({mtk::ozimmu::op_n, mtk::ozimmu::op_n, m, n, k, mtk::ozimmu::real, ozimmu_MAX_SPLIT});
            mtk::ozimmu::handle_t ozimmu_handle;
            mtk::ozimmu::create(&ozimmu_handle);
            mtk::ozimmu::reallocate_working_memory(ozimmu_handle, fp64in_gemm);
            int slice = ozimmu_MIN_SPLIT;
            for (auto &mode : mode_list) {
                cudaDeviceSynchronize();
                res = getWatt::getWatt(
                    [&]() {
                        mtk::ozimmu::gemm(ozimmu_handle,
                                          mtk::ozimmu::op_n,
                                          mtk::ozimmu::op_n,
                                          m,
                                          n,
                                          k,
                                          &alpha,
                                          devA,
                                          m,
                                          devB,
                                          k,
                                          &beta,
                                          devC,
                                          m,
                                          mode,
                                          mtk::ozimmu::real);
                    },
                    m,
                    n,
                    k);
                cudaDeviceSynchronize();
                cudaMemcpy(cpuC, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

                outFile << phi << "," << m << "," << n << "," << k << "," << "ozIMMU_EF-" << slice << ",";
                outFile << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
                std::cout << phi << "," << m << "," << n << "," << k << "," << "ozIMMU_EF-" << slice << ",";
                std::cout << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;

                slice++;
            }
            mtk::ozimmu::destroy(ozimmu_handle);
#endif

            //--------------------
            // C := A*B by ozaki-scheme2
            //--------------------
            for (auto &num_moduli : num_moduli_list) {
                cudaDeviceSynchronize();

                res = getWatt::getWatt(
                    [&]() {
                        gemmul8::gemm<double>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, true, work_gemm);
                    },
                    m,
                    n,
                    k);

                cudaDeviceSynchronize();
                cudaMemcpy(cpuC, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

                outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
                outFile << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
                std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
                std::cout << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
            }

            //--------------------
            // C := A*B by ozaki-scheme2
            //--------------------
            for (auto &num_moduli : num_moduli_list) {
                cudaDeviceSynchronize();

                res = getWatt::getWatt(
                    [&]() {
                        gemmul8::gemm<double>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, false, work_gemm);
                    },
                    m,
                    n,
                    k);

                cudaDeviceSynchronize();
                cudaMemcpy(cpuC, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

                outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
                outFile << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
                std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
                std::cout << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
            }
        }
    }

    delete[] work_cpu;
    cudaFree(work_gpu);
    cudaFree(work_gemm);
    cublasDestroy(handle);
    outFile.close();
}

int main(int argc, char **argv) {
    std::chrono::system_clock::time_point start, stop;
    std::string deviceName = getDeviceName();
    std::string startTime  = getCurrentDateTime(start);

    bool run_accuracy   = false;
    bool run_flops      = false;
    bool run_watt       = false;
    bool run_flops_rect = false;
    bool run_watt_rect  = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "accuracy") {
            run_accuracy = true;
        } else if (arg == "flops") {
            run_flops = true;
        } else if (arg == "watt") {
            run_watt = true;
        } else if (arg == "flops_rect") {
            run_flops_rect = true;
        } else if (arg == "watt_rect") {
            run_watt_rect = true;
        } else if (arg == "all") {
            run_accuracy   = true;
            run_flops      = true;
            run_watt       = true;
            run_flops_rect = true;
            run_watt_rect  = true;
        }
    }

    if (run_accuracy)
        accuracy_check(deviceName, startTime);
    if (run_flops)
        time_check(deviceName, startTime);
    if (run_watt)
        watt_check(deviceName, startTime);
    if (run_flops_rect)
        time_check_rect(deviceName, startTime);
    if (run_watt_rect)
        watt_check_rect(deviceName, startTime);

    std::string endTime = getCurrentDateTime(stop);
    auto sec            = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() * 1.e-9;
    std::cout << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << "start        : " << startTime << std::endl;
    std::cout << "end          : " << endTime << std::endl;
    std::cout << "elapsed time : " << sec << " [sec]" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << std::endl;

    return 0;
}
