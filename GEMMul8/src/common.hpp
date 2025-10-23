#pragma once
#if defined(__NVCC__)
    #include <cuComplex.h>
    #include <cublas_v2.h>
    #include <cuda_runtime.h>
#endif
#include "self_hipify.hpp"
#include "table.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace oz2 {

size_t grid_invscal;
size_t grid_conv32i8u;
inline constexpr size_t threads_scaling      = 256;
inline constexpr size_t threads_conv32i8u    = 256;
inline constexpr size_t threads_invscal      = 128;
inline constexpr unsigned TILE_DIM           = 16;
inline constexpr unsigned LOG2_TILE_DIM      = 4;
inline constexpr bool USE_CHAR4              = false;
inline constexpr unsigned CHAR4_PER_ROW      = TILE_DIM / 4;
inline constexpr unsigned LOG2_CHAR4_PER_ROW = LOG2_TILE_DIM - 2;

template <typename T> struct threshold;
template <> struct threshold<double> {
    static constexpr unsigned x = 12u;
    static constexpr unsigned y = 18u;
    static constexpr unsigned z = 25u;
};
template <> struct threshold<float> {
    static constexpr unsigned x = 5u;
    static constexpr unsigned y = 11u;
    static constexpr unsigned z = 18u;
};

// calculate work size
__inline__ size_t calc_ld8i(const size_t n) { return TILE_DIM * ((n + (TILE_DIM - 1)) / TILE_DIM); }
__inline__ size_t calc_ld32i(const size_t n) { return TILE_DIM * ((n + (TILE_DIM - 1)) / TILE_DIM); }
__inline__ size_t calc_sizevec(const size_t n) { return TILE_DIM * ((n + (TILE_DIM - 1)) / TILE_DIM); }

// define data type
template <typename T> struct Vec4Type;
template <> struct Vec4Type<double> {
#if CUBLAS_VER_MAJOR >= 13
    using type = double4_32a;
#else
    using type = double4;
#endif
};
template <> struct Vec4Type<float> {
    using type = float4;
};
template <typename T> using Vec4 = typename Vec4Type<T>::type;

// template math funcs
template <typename T> __forceinline__ __device__ T Tabs(T in);
template <> __forceinline__ __device__ double Tabs<double>(double in) { return fabs(in); }
template <> __forceinline__ __device__ float Tabs<float>(float in) { return fabsf(in); }
template <> __forceinline__ __device__ int32_t Tabs<int32_t>(int32_t in) { return abs(in); }

template <typename T> __forceinline__ __device__ int Tilogb(T in);
template <> __forceinline__ __device__ int Tilogb<double>(double in) { return (in == 0.0) ? 0 : ilogb(in); }
template <> __forceinline__ __device__ int Tilogb<float>(float in) { return (in == 0.0F) ? 0 : ilogbf(in); }

template <typename T> __forceinline__ __device__ T Tfma(const T in1, T in2, T in3);
template <> __forceinline__ __device__ double Tfma<double>(const double in1, double in2, double in3) { return fma(in1, in2, in3); }
template <> __forceinline__ __device__ float Tfma<float>(const float in1, float in2, float in3) { return __fmaf_rn(in1, in2, in3); }

template <typename T> __forceinline__ __device__ T __Tfma_ru(T in1, T in2, T in3);
template <> __forceinline__ __device__ double __Tfma_ru<double>(double in1, double in2, double in3) { return __fma_ru(in1, in2, in3); }
template <> __forceinline__ __device__ float __Tfma_ru<float>(float in1, float in2, float in3) { return __fmaf_ru(in1, in2, in3); }

template <typename T> __forceinline__ __device__ T __Tadd_ru(T in1, T in2);
template <> __forceinline__ __device__ double __Tadd_ru<double>(double in1, double in2) { return __dadd_ru(in1, in2); }
template <> __forceinline__ __device__ float __Tadd_ru<float>(float in1, float in2) { return __fadd_ru(in1, in2); }

template <typename T> __forceinline__ __device__ T Tcast(double in);
template <> __forceinline__ __device__ double Tcast<double>(double in) { return in; }
template <> __forceinline__ __device__ float Tcast<float>(double in) { return __double2float_rn(in); }

template <typename T> __forceinline__ __device__ T Tscalbn(T in, int sft);
template <> __forceinline__ __device__ double Tscalbn<double>(double in, int sft) { return scalbn(in, sft); }
template <> __forceinline__ __device__ float Tscalbn<float>(float in, int sft) { return scalbnf(in, sft); }

template <typename T> struct samesize_int;
template <> struct samesize_int<int32_t> {
    using type = int32_t;
};
template <> struct samesize_int<int64_t> {
    using type = int64_t;
};
template <> struct samesize_int<float> {
    using type = int32_t;
};
template <> struct samesize_int<double> {
    using type = int64_t;
};
template <typename T> using intT = typename samesize_int<T>::type;

template <typename T> struct samesize_fp;
template <> struct samesize_fp<int32_t> {
    using type = float;
};
template <> struct samesize_fp<int64_t> {
    using type = double;
};
template <> struct samesize_fp<float> {
    using type = float;
};
template <> struct samesize_fp<double> {
    using type = double;
};
template <typename T> using fpT = typename samesize_fp<T>::type;

template <typename T> __forceinline__ __device__ intT<T> __fp_as_int(T in);
template <> __forceinline__ __device__ intT<double> __fp_as_int<double>(double in) { return __double_as_longlong(in); }
template <> __forceinline__ __device__ intT<float> __fp_as_int<float>(float in) { return __float_as_int(in); }

template <typename T> __forceinline__ __device__ fpT<T> __int_as_fp(intT<T> in);
template <> __forceinline__ __device__ fpT<double> __int_as_fp<double>(intT<double> in) { return __longlong_as_double(in); }
template <> __forceinline__ __device__ fpT<float> __int_as_fp<float>(intT<float> in) { return __int_as_float(in); }

template <typename T> __forceinline__ __device__ intT<T> extract_sign(intT<T> in);
template <> __forceinline__ __device__ intT<double> extract_sign<double>(intT<double> in) { return in & 0x8000000000000000LL; }
template <> __forceinline__ __device__ intT<float> extract_sign<float>(intT<float> in) { return in & 0x80000000; }

template <typename T> __forceinline__ __device__ intT<T> extract_exp(intT<T> in);
template <> __forceinline__ __device__ intT<double> extract_exp<double>(intT<double> in) { return (in >> 52) & 0x7FF; }
template <> __forceinline__ __device__ intT<float> extract_exp<float>(intT<float> in) { return (in >> 23) & 0xFF; }

template <typename T> __forceinline__ __device__ intT<T> extract_significand(intT<T> in);
template <> __forceinline__ __device__ intT<double> extract_significand<double>(intT<double> in) { return in & 0x000FFFFFFFFFFFFFULL; }
template <> __forceinline__ __device__ intT<float> extract_significand<float>(intT<float> in) { return in & 0x007FFFFF; }

template <typename T> __forceinline__ __device__ int32_t countlz(intT<T> in);
template <> __forceinline__ __device__ int32_t countlz<double>(intT<double> in) { return __clzll(in); }
template <> __forceinline__ __device__ int32_t countlz<float>(intT<float> in) { return __clz(in); }

template <typename T> struct fp;
template <> struct fp<double> {
    static constexpr int32_t bias = 1023;
    static constexpr int32_t prec = 52;
    static constexpr int32_t bits = 64;
};
template <> struct fp<float> {
    static constexpr int32_t bias = 127;
    static constexpr int32_t prec = 23;
    static constexpr int32_t bits = 32;
};

template <typename T> struct Tzero {
    static constexpr T value = static_cast<T>(0);
};

template <typename T> struct Tone {
    static constexpr T value = static_cast<T>(1);
};

template <typename T> struct Tmone {
    static constexpr T value = static_cast<T>(-1);
};

// warp reduction
template <typename T, int width = 32> __forceinline__ __device__ T inner_warp_max(T amax) {
#pragma unroll
    for (unsigned i = width / 2; i > 0; i >>= 1) {
        amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, i, width));
    }
    return amax;
}

template <typename T, int width = 32> __forceinline__ __device__ T inner_warp_sum(T sum) {
#pragma unroll
    for (unsigned i = width / 2; i > 0; i >>= 1) {
        sum = __Tadd_ru<T>(sum, __shfl_down_sync(0xFFFFFFFFu, sum, i, width));
    }
    return sum;
}

// get value from constant memory
template <typename T> __device__ __forceinline__ oz2_table::tab_t<T> readtab(unsigned j);
template <> __device__ __forceinline__ oz2_table::tab_t<double> readtab<double>(unsigned j) { return oz2_table::moduli_dev[j]; };
template <> __device__ __forceinline__ oz2_table::tab_t<float> readtab<float>(unsigned j) { return oz2_table::modulif_dev[j]; };

// calculate mod: a - round(a/p(j))*p(j)
template <typename T, int MODE>
__device__ __forceinline__ int8_t mod_8i(T a, oz2_table::tab_t<T> val) //
{
    if constexpr (std::is_same_v<T, double> && MODE == 1) {
        double tmp = fma(rint(a * val.y), val.x, a);
        return static_cast<int8_t>(tmp);
    }

    float tmp;
    if constexpr (std::is_same_v<T, double>) {
        tmp = __double2float_rn(fma(rint(a * val.y), val.x, a));
#pragma unroll
        for (int i = 1; i < MODE; ++i) {
            tmp = __fmaf_rn(rintf(tmp * val.w), val.z, tmp);
        }
    } else {
        tmp = __fmaf_rn(rintf(a * val.y), val.x, a);
#pragma unroll
        for (int i = 1; i < MODE; ++i) {
            tmp = __fmaf_rn(rintf(tmp * val.y), val.x, tmp);
        }
    }
    return static_cast<int8_t>(tmp);
}

template <typename T> __device__ T T2int_fp(T in, const int sft) //
{
    intT<T> bits        = __fp_as_int<T>(in);
    const intT<T> sign  = extract_sign<T>(bits);
    int exp_biased      = (int)extract_exp<T>(bits);
    intT<T> significand = extract_significand<T>(bits);

    if (exp_biased != 0) {
        exp_biased += sft;
        if (exp_biased < fp<T>::bias) {
            return __int_as_fp<T>(sign);
        }
        if (exp_biased >= (fp<T>::bias + fp<T>::prec)) {
            bits = sign | ((intT<T>)exp_biased << fp<T>::prec) | significand;
            return __int_as_fp<T>(bits);
        }

        significand |= ((intT<T>)1 << fp<T>::prec);
        int chop_bits = (fp<T>::bias + fp<T>::prec) - exp_biased;
        intT<T> mask  = (intT<T>)(-1) << chop_bits;
        significand   = extract_significand<T>(significand & mask);
        bits          = sign | (intT<T>)exp_biased << fp<T>::prec | significand;
        return __int_as_fp<T>(bits);
    }

    if (significand == 0) {
        return in;
    }

    int lz = (fp<T>::bits - fp<T>::prec) - countlz<T>(significand);
    int e  = lz + sft;

    if (e < fp<T>::bias) {
        return __int_as_fp<T>(sign);
    }

    intT<T> frac_full = (significand << (2 - lz)) ^ ((intT<T>)1 << fp<T>::prec);
    intT<T> mask      = (intT<T>)(-1) << max((intT<T>)(fp<T>::bias + fp<T>::prec - e), (intT<T>)0);
    bits              = sign | (intT<T>)e << fp<T>::prec | (frac_full & mask);
    return __int_as_fp<T>(bits);
}

// int8_t(ceil(scalbn(fabs(in),sft)))
template <typename T> __device__ int8_t T2int8i(T in, const int sft) //
{
    intT<T> bits_full   = __fp_as_int<T>(in);
    int exp_biased      = (int)extract_exp<T>(bits_full);
    intT<T> significand = extract_significand<T>(bits_full);
    intT<T> result;

    if (exp_biased != 0) {
        significand |= ((intT<T>)1 << fp<T>::prec);
        int shift_amount = (fp<T>::bias + fp<T>::prec) - exp_biased - sft;

        intT<T> divisor = (intT<T>)1 << shift_amount;
        result          = (significand + divisor - 1) >> shift_amount;
        return static_cast<int8_t>(result);
    }

    if (significand == 0) {
        return Tzero<int8_t>::value;
    }

    int lz = (fp<T>::bits - fp<T>::prec) - countlz<T>(significand);
    significand <<= (2 - lz);
    int shift_amount = (fp<T>::bias + fp<T>::prec) - lz - sft;

    intT<T> divisor = (intT<T>)1 << shift_amount;
    result          = (significand + divisor - 1) >> shift_amount;
    return static_cast<int8_t>(result);
};

// column-wise amax
template <typename T> __device__ T find_amax(
    const T *const ptr,    //
    const unsigned length, //
    T *samax               // shared memory (workspace)
) {
    // max in thread
    T amax = Tzero<T>::value;
    for (unsigned i = threadIdx.x; i < length; i += blockDim.x) {
        T tmp = Tabs<T>(ptr[i]);
        amax  = max(amax, tmp);
    }

    // inner-warp reduction
    amax = inner_warp_max(amax);

    // inner-threadblock reduction
    if ((threadIdx.x & 0x1f) == 0) samax[threadIdx.x >> 5] = amax; // samax[warp-id] = max in warp

    __syncthreads();
    if (threadIdx.x < 32) {
        if (threadIdx.x < (blockDim.x >> 5)) amax = samax[threadIdx.x];
        amax = inner_warp_max(amax);
        if (threadIdx.x == 0) samax[0] = amax;
    }

    __syncthreads();
    return samax[0];
}

// column-wise amax & sum of squares
template <typename T> __device__ T find_amax_and_nrm(
    const T *const ptr,    //
    const unsigned length, //
    T *shm,                // shared memory (workspace)
    T &vecnrm              // 2-norm^2
) {
    T *samax = shm;
    T *ssum  = shm + 32;

    // max in thread
    T amax = Tzero<T>::value;
    T sum  = Tzero<T>::value;
    for (unsigned i = threadIdx.x; i < length; i += blockDim.x) {
        T tmp = Tabs<T>(ptr[i]);
        amax  = max(amax, tmp);
        sum   = __Tfma_ru<T>(tmp, tmp, sum); // round-up mode
    }

    // inner-warp reduction
    amax = inner_warp_max(amax);
    sum  = inner_warp_sum(sum);

    // inner-threadblock reduction
    if ((threadIdx.x & 31) == 0) {
        samax[threadIdx.x >> 5] = amax; // samax[warp-id] = max in warp
        ssum[threadIdx.x >> 5]  = sum;  // ssum[warp-id] = sum in warp
    }

    __syncthreads();
    sum = Tzero<T>::value;
    if (threadIdx.x < 32) {
        if (threadIdx.x < (blockDim.x >> 5)) {
            amax = samax[threadIdx.x];
            sum  = ssum[threadIdx.x];
        }
        amax = inner_warp_max(amax);
        sum  = inner_warp_sum(sum);
        if (threadIdx.x == 0) {
            samax[0] = amax;
            ssum[0]  = sum;
        }
    }

    __syncthreads();
    vecnrm = ssum[0];
    return samax[0];
}

void timing(std::chrono::system_clock::time_point &time_stamp) {
    cudaDeviceSynchronize();
    time_stamp = std::chrono::system_clock::now();
}

void timing(std::chrono::system_clock::time_point &time_stamp, double &timer) {
    cudaDeviceSynchronize();
    std::chrono::system_clock::time_point time_now = std::chrono::system_clock::now();
    timer += std::chrono::duration_cast<std::chrono::nanoseconds>(time_now - time_stamp).count();
    time_stamp = time_now;
}

} // namespace oz2
