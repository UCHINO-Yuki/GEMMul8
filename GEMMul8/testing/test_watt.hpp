#pragma once
#include "getWatt.hpp"

template <typename T, gemmul8::Backend backend = gemmul8::Backend::FP8>
__inline__ void watt_check(std::string &deviceName, std::string &dateTime) {
    std::string fileName = std::string("oz2_results_") + backendType<backend> + "_" + gemmTraits<T>::prefix + "_watt_" + deviceName + "_" + dateTime + ".csv";
    std::ofstream outFile(fileName);

    CHECK_CUDA(cudaSetDevice(0));
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    cublasHandle_t handle_Emu;
    CHECK_CUBLAS(cublasCreate(&handle_Emu));
    cublasLtHandle_t handleLt;
    CHECK_CUBLASLT(cublasLtCreate(&handleLt));

    outFile << std::scientific;
    std::cout << std::scientific;
    outFile << "phi,m,n,k,function,watt,GFLOPS/watt," << std::endl;
    std::cout << "phi,m,n,k,function,watt,GFLOPS/watt," << std::endl;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    std::vector<float> times(mainloop, 0.0);

    using accu_t        = typename gemmTraits<T>::ACCU_TYPE;
    size_t total_memory = size_t(GPU_MEM_MB);

#if defined(__NVCC__)
    int cuBLASversion;
    cublasGetVersion(handle, &cuBLASversion);
#endif

    const double phi = -1.0;
    const T alpha    = gemmTraits<T>::one();
    const T beta     = gemmTraits<T>::zero();

    for (size_t m = 1024; m <= size_max; m *= 2) {
        size_t n = m;
        size_t k = m;

        const size_t size_A           = m * k;
        const size_t size_B           = k * n;
        const size_t size_C           = m * n;
        const unsigned num_moduli_max = NUM_MODULI_MAX<T>;
        const size_t lwork_gemmul8    = gemmul8::workSize<gemmTraits<T>::is_complex, backend>(m, n, k, num_moduli_max);
#if defined(__NVCC__) && CUBLAS_GE_13_1
        const size_t lwork_ozaki1 = (gemmTraits<T>::is_double) ? ozaki1::workSize(m, n, k, 1, gemmTraits<T>::is_complex, CUDA_EMULATION_MANTISSA_CONTROL_FIXED, 8 * 9 - 1) : 0;
#else
        const size_t lwork_ozaki1 = 0;
#endif
        const size_t lwork      = std::max(lwork_gemmul8, lwork_ozaki1);
        const size_t total_work = lwork + (size_A + size_B + size_C) * sizeof(T);
        if ((total_work + 256ULL * sizeof(accu_t)) * 1.e-6 > total_memory) {
            continue;
        }

        T *A, *B, *C;
        void *work;

        CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&A), size_A * sizeof(T)));
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&B), size_B * sizeof(T)));
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&C), size_C * sizeof(T)));
        CHECK_CUDA(cudaMalloc(&work, lwork));

        const int64_t mi = static_cast<int64_t>(m);
        const int64_t ni = static_cast<int64_t>(n);
        const int64_t ki = static_cast<int64_t>(k);

        // test matrices
        makemat::randmat<T>(m, k, A, phi, seedA);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        makemat::randmat<T>(k, n, B, phi, seedB);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // native gemm
        {
            for (int i = 0; i < warmup; ++i) {
                CHECK_CUBLAS(gemmTraits<T>::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, mi, ni, ki, &alpha, A, mi, B, ki, &beta, C, mi));
            }

            std::vector<double> res = getWatt::getWatt(
                [&]() { CHECK_CUBLAS(gemmTraits<T>::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, mi, ni, ki, &alpha, A, mi, B, ki, &beta, C, mi)); },
                m, n, k);

            outFile << std::scientific;
            std::cout << std::scientific;
            outFile << phi << "," << m << "," << n << "," << k << "," << gemmTraits<T>::prefix_upper() << "GEMM" << ",";
            outFile << res[0] << "," << ((gemmTraits<T>::is_complex) ? 4.0 : 1.0) * res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << gemmTraits<T>::prefix_upper() << "GEMM" << ",";
            std::cout << res[0] << "," << ((gemmTraits<T>::is_complex) ? 4.0 : 1.0) * res[1] * 1.e-9 << "," << std::endl;
        }

        // native gemm3m
#if defined(__NVCC__)
        if constexpr (gemmTraits<T>::is_double && gemmTraits<T>::is_complex) {
            for (int i = 0; i < warmup; ++i) {
                CHECK_CUBLAS(gemmTraits<T>::gemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, mi, ni, ki, &alpha, A, mi, B, ki, &beta, C, mi));
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaDeviceSynchronize());
            }

            std::vector<double> res = getWatt::getWatt(
                [&]() { CHECK_CUBLAS(gemmTraits<T>::gemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, mi, ni, ki, &alpha, A, mi, B, ki, &beta, C, mi)); },
                m, n, k);

            outFile << std::scientific;
            std::cout << std::scientific;
            outFile << phi << "," << m << "," << n << "," << k << "," << gemmTraits<T>::prefix_upper() << "GEMM3m" << ",";
            outFile << res[0] << "," << ((gemmTraits<T>::is_complex) ? 4.0 : 1.0) * res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << gemmTraits<T>::prefix_upper() << "GEMM3m" << ",";
            std::cout << res[0] << "," << ((gemmTraits<T>::is_complex) ? 4.0 : 1.0) * res[1] * 1.e-9 << "," << std::endl;
        }
#endif

        // fast mode
        for (unsigned num_moduli = NUM_MODULI_MIN<T>; num_moduli <= NUM_MODULI_MAX<T>; ++num_moduli) {

            std::vector<double> time0(mainloop, 0.0), time1(mainloop, 0.0), time2(mainloop, 0.0), time3(mainloop, 0.0), timestmp(4, 0.0);

            for (int i = 0; i < warmup; ++i) {
#if defined(__NVCC__)
                gemmul8::gemmLt<T, backend>(handleLt, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m, num_moduli, true, work);
#else
                if constexpr (backend == gemmul8::Backend::INT8) {
                    gemmul8::gemm<T, backend>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m, num_moduli, true, work);
                } else {
                    gemmul8::gemmLt<T, backend>(handleLt, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m, num_moduli, true, work);
                }
#endif
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaDeviceSynchronize());
            }

            std::vector<double> res = getWatt::getWatt(
                [&]() {
#if defined(__NVCC__)
                    gemmul8::gemmLt<T, backend>(handleLt, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m, num_moduli, true, work);
#else
                    if constexpr (backend == gemmul8::Backend::INT8) {
                        gemmul8::gemm<T, backend>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m, num_moduli, true, work);
                    } else {
                        gemmul8::gemmLt<T, backend>(handleLt, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m, num_moduli, true, work);
                    }
#endif
                },
                m, n, k);

            outFile << std::scientific;
            std::cout << std::scientific;
            outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
            outFile << res[0] << "," << ((gemmTraits<T>::is_complex) ? 4.0 : 1.0) * res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
            std::cout << res[0] << "," << ((gemmTraits<T>::is_complex) ? 4.0 : 1.0) * res[1] * 1.e-9 << "," << std::endl;
        }

        // accu mode
        for (unsigned num_moduli = NUM_MODULI_MIN<T>; num_moduli <= NUM_MODULI_MAX<T>; ++num_moduli) {

            std::vector<double> time0(mainloop, 0.0), time1(mainloop, 0.0), time2(mainloop, 0.0), time3(mainloop, 0.0), timestmp(4, 0.0);

            for (int i = 0; i < warmup; ++i) {
#if defined(__NVCC__)
                gemmul8::gemmLt<T, backend>(handleLt, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m, num_moduli, false, work);
#else
                if constexpr (backend == gemmul8::Backend::INT8) {
                    gemmul8::gemm<T, backend>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m, num_moduli, false, work);
                } else {
                    gemmul8::gemmLt<T, backend>(handleLt, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m, num_moduli, false, work);
                }
#endif
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaDeviceSynchronize());
            }

            std::vector<double> res = getWatt::getWatt(
                [&]() {
#if defined(__NVCC__)
                    gemmul8::gemmLt<T, backend>(handleLt, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m, num_moduli, false, work);
#else
                    if constexpr (backend == gemmul8::Backend::INT8) {
                        gemmul8::gemm<T, backend>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m, num_moduli, false, work);
                    } else {
                        gemmul8::gemmLt<T, backend>(handleLt, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m, num_moduli, false, work);
                    }
#endif
                },
                m, n, k);

            outFile << std::scientific;
            std::cout << std::scientific;
            outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
            outFile << res[0] << "," << ((gemmTraits<T>::is_complex) ? 4.0 : 1.0) * res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
            std::cout << res[0] << "," << ((gemmTraits<T>::is_complex) ? 4.0 : 1.0) * res[1] * 1.e-9 << "," << std::endl;
        }

// cuBLAS emulation
#if defined(__NVCC__)
        if (gemmTraits<T>::is_double) {
            if (cuBLASversion >= 130100) {
    #if CUBLAS_GE_13_1
                cublasSetMathMode(handle_Emu, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH);
                cublasSetEmulationStrategy(handle_Emu, CUBLAS_EMULATION_STRATEGY_EAGER);
                cublasSetFixedPointEmulationMantissaControl(handle_Emu, CUDA_EMULATION_MANTISSA_CONTROL_FIXED);
                cublasSetWorkspace(handle_Emu, work, lwork);

                for (int num_slice = 6; num_slice < 10; ++num_slice) {
                    int mantissaBitCount = num_slice * 8 - 1;
                    cublasSetFixedPointEmulationMaxMantissaBitCount(handle_Emu, mantissaBitCount);

                    for (int i = 0; i < warmup; ++i) {
                        CHECK_CUBLAS(gemmTraits<T>::gemm(handle_Emu, CUBLAS_OP_N, CUBLAS_OP_N, mi, ni, ki, &alpha, A, mi, B, ki, &beta, C, mi));
                    }

                    std::vector<double> res = getWatt::getWatt(
                        [&]() { CHECK_CUBLAS(gemmTraits<T>::gemm(handle_Emu, CUBLAS_OP_N, CUBLAS_OP_N, mi, ni, ki, &alpha, A, mi, B, ki, &beta, C, mi)); },
                        m, n, k);

                    outFile << std::scientific;
                    std::cout << std::scientific;
                    outFile << phi << "," << m << "," << n << "," << k << "," << "Oz1-" << num_slice << ",";
                    outFile << res[0] << "," << ((gemmTraits<T>::is_complex) ? 4.0 : 1.0) * res[1] * 1.e-9 << "," << std::endl;
                    std::cout << phi << "," << m << "," << n << "," << k << "," << "Oz1-" << num_slice << ",";
                    std::cout << res[0] << "," << ((gemmTraits<T>::is_complex) ? 4.0 : 1.0) * res[1] * 1.e-9 << "," << std::endl;
                }

                cublasSetWorkspace(handle_Emu, nullptr, 0);
                cublasSetMathMode(handle_Emu, CUBLAS_DEFAULT_MATH);
    #endif
            }

        } else {
            if (cuBLASversion >= 12.9) {
    #if CUBLAS_GE_12_9
                cublasSetMathMode(handle_Emu, CUBLAS_FP32_EMULATED_BF16X9_MATH);

                for (int i = 0; i < warmup; ++i) {
                    CHECK_CUBLAS(gemmTraits<T>::gemm(handle_Emu, CUBLAS_OP_N, CUBLAS_OP_N, mi, ni, ki, &alpha, A, mi, B, ki, &beta, C, mi));
                }

                std::vector<double> res = getWatt::getWatt(
                    [&]() { CHECK_CUBLAS(gemmTraits<T>::gemm(handle_Emu, CUBLAS_OP_N, CUBLAS_OP_N, mi, ni, ki, &alpha, A, mi, B, ki, &beta, C, mi)); },
                    m, n, k);

                outFile << std::scientific;
                std::cout << std::scientific;
                outFile << phi << "," << m << "," << n << "," << k << "," << "BF16x9" << ",";
                outFile << res[0] << "," << ((gemmTraits<T>::is_complex) ? 4.0 : 1.0) * res[1] * 1.e-9 << "," << std::endl;
                std::cout << phi << "," << m << "," << n << "," << k << "," << "BF16x9" << ",";
                std::cout << res[0] << "," << ((gemmTraits<T>::is_complex) ? 4.0 : 1.0) * res[1] * 1.e-9 << "," << std::endl;

                cublasSetMathMode(handle_Emu, CUBLAS_DEFAULT_MATH);
    #endif
            }
        }
#endif

        CHECK_CUDA(cudaFree(work));
        CHECK_CUDA(cudaFree(C));
        CHECK_CUDA(cudaFree(B));
        CHECK_CUDA(cudaFree(A));
    }

    std::cout << std::endl;
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUBLASLT(cublasLtDestroy(handleLt));
    CHECK_CUBLAS(cublasDestroy(handle_Emu));
    CHECK_CUBLAS(cublasDestroy(handle));
    outFile.close();
}
