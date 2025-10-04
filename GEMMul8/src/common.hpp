#pragma once
#include "cuda_impl.hpp"
#include "table.hpp"
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#define minus_half -0x1.0000060000000p-1F

namespace oz2_const {

size_t grids_invscaling;
size_t grids_conv32i8u;

size_t threads_scaling;
size_t threads_conv32i8u;
size_t threads_invscaling;

} // namespace oz2_const

namespace oz2_util {
template <typename T>
struct Vec4 {
    T x, y, z, w;
};

template <typename T> bool is_pow2(T n) {
    if (n == 0) return false;
    return (n & (n - 1)) == 0;
}

template <typename T> __forceinline__ __device__ T Tabs(T in);
template <> __forceinline__ __device__ double Tabs<double>(double in) { return fabs(in); };
template <> __forceinline__ __device__ float Tabs<float>(float in) { return fabsf(in); };
template <> __forceinline__ __device__ int32_t Tabs<int32_t>(int32_t in) { return abs(in); };

template <typename T> __forceinline__ __device__ int Tilogb(T in);
template <> __forceinline__ __device__ int Tilogb<double>(double in) { return (in == 0.0) ? 0 : ilogb(in); };
template <> __forceinline__ __device__ int Tilogb<float>(float in) { return (in == 0.0F) ? 0 : ilogbf(in); };

template <typename T> __forceinline__ __device__ T Tzero() { return 0; };
template <> __forceinline__ __device__ double Tzero<double>() { return 0.0; };
template <> __forceinline__ __device__ float Tzero<float>() { return 0.0F; };
template <> __forceinline__ __device__ int32_t Tzero<int32_t>() { return 0; };

template <typename T> __forceinline__ __device__ T __Tfma_ru(T in1, T in2, T in3);
template <> __forceinline__ __device__ double __Tfma_ru<double>(double in1, double in2, double in3) { return __fma_ru(in1, in2, in3); };
template <> __forceinline__ __device__ float __Tfma_ru<float>(float in1, float in2, float in3) { return __fmaf_ru(in1, in2, in3); };

template <typename T> __forceinline__ __device__ T __Tadd_ru(T in1, T in2);
template <> __forceinline__ __device__ double __Tadd_ru<double>(double in1, double in2) { return __dadd_ru(in1, in2); };
template <> __forceinline__ __device__ float __Tadd_ru<float>(float in1, float in2) { return __fadd_ru(in1, in2); };

template <typename T> __forceinline__ __device__ T Tcast(double in);
template <> __forceinline__ __device__ double Tcast<double>(double in) { return in; };
template <> __forceinline__ __device__ float Tcast<float>(double in) { return __double2float_rn(in); };

template <typename T> __forceinline__ __device__ T Tfma(const T in1, T in2, T in3);
template <> __forceinline__ __device__ double Tfma<double>(const double in1, double in2, double in3) {
    return fma(in1, in2, in3);
};
template <> __forceinline__ __device__ float Tfma<float>(const float in1, float in2, float in3) {
    return __fmaf_rn(in1, in2, in3);
};

template <typename T, int N> __forceinline__ __device__ void inner_warp_max(T &amax) {
#pragma unroll
    for (int offset = N; offset > 0; offset >>= 1) {
        amax = max(amax, __shfl_down_sync(FULL_MASK, amax, offset));
    }
}
template <int N> __forceinline__ __device__ void inner_warp_sum(double &sum) {
#pragma unroll
    for (int offset = N; offset > 0; offset >>= 1) {
        sum = __dadd_ru(sum, __shfl_down_sync(FULL_MASK, sum, offset));
    }
}
template <int N> __forceinline__ __device__ void inner_warp_sum(float &sum) {
#pragma unroll
    for (int offset = N; offset > 0; offset >>= 1) {
        sum = __fadd_ru(sum, __shfl_down_sync(FULL_MASK, sum, offset));
    }
}
template <int N> __forceinline__ __device__ void inner_warp_sum(int32_t &sum) {
#pragma unroll
    for (int offset = N; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }
}

template <typename T> __device__ __forceinline__ oz2_table::tab_t<T> readtab(unsigned j);
template <> __device__ __forceinline__ oz2_table::tab_t<double> readtab<double>(unsigned j) { return oz2_table::moduli_dev[j]; };
template <> __device__ __forceinline__ oz2_table::tab_t<float> readtab<float>(unsigned j) { return oz2_table::modulif_dev[j]; };

// calculate mod: a - round(a/p(j))*p(j)
template <typename T, int MODE> __device__ __forceinline__ int8_t mod_8i(T a, oz2_table::tab_t<T> val);
template <> __device__ __forceinline__ int8_t mod_8i<double, 1>(double a, oz2_table::tab_t<double> val) {
    double tmp1 = fma(rint(a * val.y), val.x, a);
    return static_cast<int8_t>(tmp1);
}
template <> __device__ __forceinline__ int8_t mod_8i<float, 1>(float a, oz2_table::tab_t<float> val) {
    float tmp1 = __fmaf_rn(rintf(a * val.y), val.x, a);
    return static_cast<int8_t>(tmp1);
}
template <> __device__ __forceinline__ int8_t mod_8i<double, 2>(double a, oz2_table::tab_t<double> val) {
    float tmp1 = __double2float_rn(fma(rint(a * val.y), val.x, a));
    float tmp2 = __fmaf_rn(rintf(tmp1 * val.w), val.z, tmp1);
    return static_cast<int8_t>(tmp2);
}
template <> __device__ __forceinline__ int8_t mod_8i<float, 2>(float a, oz2_table::tab_t<float> val) {
    float tmp1 = __fmaf_rn(rintf(a * val.y), val.x, a);
    float tmp2 = __fmaf_rn(rintf(tmp1 * val.y), val.x, tmp1);
    return static_cast<int8_t>(tmp2);
}
template <> __device__ __forceinline__ int8_t mod_8i<double, 3>(double a, oz2_table::tab_t<double> val) {
    float tmp1 = __double2float_rn(fma(rint(a * val.y), val.x, a));
    float tmp2 = __fmaf_rn(rintf(tmp1 * val.w), val.z, tmp1);
    float tmp3 = __fmaf_rn(rintf(tmp2 * val.w), val.z, tmp2);
    return static_cast<int8_t>(tmp3);
}
template <> __device__ __forceinline__ int8_t mod_8i<float, 3>(float a, oz2_table::tab_t<float> val) {
    float tmp1 = __fmaf_rn(rintf(a * val.y), val.x, a);
    float tmp2 = __fmaf_rn(rintf(tmp1 * val.y), val.x, tmp1);
    float tmp3 = __fmaf_rn(rintf(tmp2 * val.y), val.x, tmp2);
    return static_cast<int8_t>(tmp3);
}

//==========================
// trunc(scalbn(in,sft))
//==========================
template <typename T> __forceinline__ __device__ T T2int_fp(T in, const int sft);
template <>
__forceinline__ __device__ double T2int_fp<double>(double in, const int sft) {
    int64_t bits                = __double_as_longlong(in);
    const int64_t sign          = bits & 0x8000000000000000LL;
    const int exp_raw           = (int)((bits >> 52) & 0x7FF);
    const int64_t mantissa_bits = bits & 0x000FFFFFFFFFFFFFULL;

    if (exp_raw != 0) {
        int final_exp = exp_raw + sft;
        if (final_exp < 1023) {
            return __longlong_as_double(sign);
        }
        if (final_exp >= 1075) {
            bits = sign | (int64_t)final_exp << 52 | mantissa_bits;
            return __longlong_as_double(bits);
        }

        const int64_t mantissa_full  = mantissa_bits | (1LL << 52);
        const int chop_bits          = 1075 - final_exp;
        const int64_t mask           = -1LL << chop_bits;
        const int64_t final_mantissa = (mantissa_full & mask) & 0x000FFFFFFFFFFFFFULL;

        bits = sign | (int64_t)final_exp << 52 | final_mantissa;
        return __longlong_as_double(bits);
    }

    if (mantissa_bits == 0) {
        return in;
    }

    const int lz = 12 - __clzll(mantissa_bits);
    int e        = lz + sft;

    if (e < 1023) {
        return __longlong_as_double(sign);
    }

    const int64_t frac_full = (mantissa_bits << (2 - lz)) ^ (1LL << 52);
    const int64_t mask      = -1LL << max(1075 - e, 0);

    bits = sign | (int64_t)e << 52 | (frac_full & mask);
    return __longlong_as_double(bits);
}
template <>
__forceinline__ __device__ float T2int_fp<float>(float in, const int sft) {
    int32_t bits                = __float_as_int(in);
    const int32_t sign          = bits & 0x80000000;
    const int exp_raw           = (bits >> 23) & 0xFF;
    const int32_t mantissa_bits = bits & 0x007FFFFF;

    if (exp_raw != 0) {
        int final_exp = exp_raw + sft;
        if (final_exp < 127) {
            return __int_as_float(sign);
        }
        if (final_exp >= 150) {
            bits = sign | final_exp << 23 | mantissa_bits;
            return __int_as_float(bits);
        }

        const int32_t mantissa_full  = mantissa_bits | (1 << 23);
        const int chop_bits          = 150 - final_exp;
        const int32_t mask           = -1 << chop_bits;
        const int32_t final_mantissa = (mantissa_full & mask) & 0x007FFFFF;

        bits = sign | final_exp << 23 | final_mantissa;
        return __int_as_float(bits);
    }

    if (mantissa_bits == 0) {
        return in;
    }

    const int lz = 9 - __clz(mantissa_bits);
    int e        = lz + sft;

    if (e < 127) {
        return __int_as_float(sign);
    }

    const int32_t frac_full = (mantissa_bits << (2 - lz)) ^ (1 << 23);
    const int32_t mask      = -1 << max(150 - e, 0);

    bits = sign | e << 23 | (frac_full & mask);
    return __int_as_float(bits);
}

//==========================
// int8_t(ceil(scalbn(fabs(in),sft)))
//==========================
template <typename T> __forceinline__ __device__ int8_t T2int8i(T in, const int sft);
template <> __forceinline__ __device__ int8_t T2int8i<double>(double in, const int sft) {
    int64_t bits_full           = __double_as_longlong(in);
    const int exp_biased        = (int)((bits_full >> 52) & 0x7FF);
    const int64_t mantissa_bits = bits_full & 0x000FFFFFFFFFFFFFULL;
    int64_t result;

    if (exp_biased != 0) {
        const int64_t mantissa_full = mantissa_bits | (1LL << 52);
        const int shift_amount      = 1075 - exp_biased - sft;

        const int64_t divisor = 1LL << shift_amount;
        result                = (mantissa_full + divisor - 1) >> shift_amount;
        return static_cast<int8_t>(result);
    }

    if (mantissa_bits == 0) {
        return static_cast<int8_t>(0);
    }

    const int numzero           = 12 - __clzll(mantissa_bits);
    const int64_t mantissa_full = mantissa_bits << (2 - numzero);
    const int shift_amount      = 1075 - numzero - sft;

    const int64_t divisor = 1LL << shift_amount;
    result                = (mantissa_full + divisor - 1) >> shift_amount;
    return static_cast<int8_t>(result);
}
template <> __forceinline__ __device__ int8_t T2int8i<float>(float in, const int sft) {
    int32_t bits_full           = __float_as_int(in);
    const int exp_biased        = (bits_full >> 23) & 0xFF;
    const int32_t mantissa_bits = bits_full & 0x007FFFFFU;
    int32_t result;

    if (exp_biased != 0) {
        const int32_t mantissa_full = mantissa_bits | (1 << 23);
        const int shift_amount      = 150 - exp_biased - sft;

        const int32_t divisor = 1 << shift_amount;
        result                = (mantissa_full + divisor - 1) >> shift_amount;
        return static_cast<int8_t>(result);
    }

    if (mantissa_bits == 0) {
        return static_cast<int8_t>(0);
    }

    const int numzero           = 9 - __clz(mantissa_bits);
    const int32_t mantissa_full = mantissa_bits << (2 - numzero);
    const int shift_amount      = 150 - numzero - sft;

    const int32_t divisor = 1 << shift_amount;
    result                = (mantissa_full + divisor - 1) >> shift_amount;
    return static_cast<int8_t>(result);
}

// return max(abs(ptr[0:inc:length-1]))
template <typename T>
__device__ T find_amax(const T *const ptr,    //
                       const unsigned length, //
                       const unsigned inc,    // leading dimension
                       T *shm)                // shared memory (workspace)
{
    // max in thread
    T amax = Tzero<T>();
    for (unsigned i = threadIdx.x; i < length; i += blockDim.x) {
        T tmp = Tabs<T>(ptr[i * inc]);
        amax  = max(amax, tmp);
    }

    // inner-warp reduction
    inner_warp_max<T, 16>(amax);

    // inner-threadblock reduction
    if ((threadIdx.x & 0x1f) == 0) shm[threadIdx.x >> 5] = amax; // shm[warp-id] = max in warp

    __syncthreads();
    amax = Tzero<T>();
    if (threadIdx.x < 32) {
        if (threadIdx.x < (blockDim.x >> 5)) amax = shm[threadIdx.x];
        inner_warp_max<T, 16>(amax);
        if (threadIdx.x == 0) shm[0] = amax;
    }

    __syncthreads();
    return shm[0];
}

// return max(abs(ptr[0:inc:length-1])) and sum_{i=0}^{length-1}(ptr[i]^2)
template <typename T>
__device__ T find_amax_and_nrm(const T *const ptr,    //
                               const unsigned length, //
                               const unsigned inc,    // leading dimension
                               T *shm,                // shared memory (workspace)
                               T &vecnrm)             // 2-norm^2
{
    T *shm1 = shm;
    T *shm2 = shm + 32;

    // max in thread
    T amax = Tzero<T>();
    T sum  = Tzero<T>();
    for (unsigned i = threadIdx.x; i < length; i += blockDim.x) {
        T tmp = Tabs<T>(ptr[i * inc]);
        amax  = max(amax, tmp);
        sum   = __Tfma_ru<T>(tmp, tmp, sum); // round-up mode
    }

    // inner-warp reduction
    inner_warp_max<T, 16>(amax);
    inner_warp_sum<16>(sum);

    // inner-threadblock reduction
    const auto id = (threadIdx.x & 0x1f);
    if (id == 0) {
        shm1[threadIdx.x >> 5] = amax; // shm[warp-id] = max in warp
    } else if (id == 1) {
        shm2[(threadIdx.x - 1) >> 5] = sum; // shm[warp-id] = sum in warp
    }

    __syncthreads();
    amax = Tzero<T>();
    sum  = Tzero<T>();
    if (threadIdx.x < 32) {
        if (threadIdx.x < (blockDim.x >> 5)) amax = shm1[threadIdx.x];
        inner_warp_max<T, 16>(amax);
        if (threadIdx.x == 0) shm1[0] = amax;
    } else if (threadIdx.x < 64) {
        if ((threadIdx.x - 32) < (blockDim.x >> 5)) sum = shm[threadIdx.x];
        inner_warp_sum<16>(sum);
        if (threadIdx.x == 32) shm2[0] = sum;
    }

    __syncthreads();
    vecnrm = shm2[0];
    return shm[0];
}
} // namespace oz2_util
