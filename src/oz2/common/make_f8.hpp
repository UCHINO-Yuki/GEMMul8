#pragma once
#include "common.hpp"
#include "table.hpp"

namespace gemmul8::common {

// eror-free conversion a to {b.x, b.y} s.t. a = sqrt(p)*b.x + b.y for p = int32_t^2
template <unsigned IDX> __device__ __forceinline__ fp8x2_e4m3 make_fp8x2(int32_t a) {
    constexpr float inv_sqrtp = 1.0f / float(table::sqrt_moduli<IDX>); // 1/sqrt(p)
    const float a_f           = __int2float_rz(a);                     // without error
    const float q             = a_f * inv_sqrtp;                       // a/sqrt(p)
    const float bx_f          = rintf(q);                              // round(a/sqrt(p))
    constexpr float m_sqrtp   = -float(table::sqrt_moduli<IDX>);       // -sqrt(p)
    const float by_f          = __fmaf_rn(m_sqrtp, bx_f, a_f);         // a - sqrt(p) * round(a/sqrt(p)

    fp8x2_e4m3 b;
    b.x = __nv_fp8_e4m3(bx_f); // without error
    b.y = __nv_fp8_e4m3(by_f); // without error

    return b;
}

// eror-free conversion a to {b.x, b.y, b.z} s.t. a = 16*b.x + b.y & b.z = b.x + b.y
__device__ __forceinline__ fp8x3_e4m3 make_fp8x3(int32_t a) {
    const uint32_t ua   = uint32_t(a);
    const uint32_t sign = ua >> 31;                              // 0: a>=0, 1: a<0
    const uint32_t absu = (ua ^ (0u - sign)) + sign;             // |a| as uint32
    const uint32_t q    = (absu + 15u) >> 4;                     // ceil(|a|/16)
    const int32_t bx    = (sign ? (-int32_t(q)) : (int32_t(q))); // sign(a)*ceil(|a|/16)
    const int32_t by    = a - 16 * bx;

    fp8x3_e4m3 b;
    b.x = __nv_fp8_e4m3(bx);      // without error
    b.y = __nv_fp8_e4m3(by);      // without error
    b.z = __nv_fp8_e4m3(bx + by); // without error

    return b;
}

} // namespace gemmul8::common
