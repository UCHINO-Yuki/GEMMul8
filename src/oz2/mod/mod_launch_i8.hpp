#pragma once
#include "../common/common.hpp"
#include "mod_core.hpp"

namespace gemmul8::mod {

// launcher for V in {int32_t, int64_t, float, double}
template <unsigned IDX, typename V> __device__ __forceinline__ void mod_launch(
    int8_t *__restrict__ out,
    V v //
) {
    out[0] = static_cast<int8_t>(calc_mod<Backend::INT8, IDX>(v));
}

// launcher for V in {int32_t, int64_t, float, double}
template <unsigned IDX, typename V> __device__ __forceinline__ void mod_launch(
    char4 *__restrict__ out,
    V v0, V v1, V v2, V v3 //
) {
    char4 rem;
    rem.x  = static_cast<int8_t>(calc_mod<Backend::INT8, IDX>(v0));
    rem.y  = static_cast<int8_t>(calc_mod<Backend::INT8, IDX>(v1));
    rem.z  = static_cast<int8_t>(calc_mod<Backend::INT8, IDX>(v2));
    rem.w  = static_cast<int8_t>(calc_mod<Backend::INT8, IDX>(v3));
    out[0] = rem;
}

// launcher for V in {int2, int64x2_t, float2, double2}
template <unsigned IDX, typename V> __device__ __forceinline__ void mod_launch(
    int8_t *__restrict__ out_r,
    int8_t *__restrict__ out_i,
    int8_t *__restrict__ out_ri,
    V v //
) {
    const int8_t vr = static_cast<int8_t>(calc_mod<Backend::INT8, IDX>(v.x));
    const int8_t vi = static_cast<int8_t>(calc_mod<Backend::INT8, IDX>(v.y));
    out_r[0]        = vr;
    out_i[0]        = vi;
    out_ri[0]       = static_cast<int8_t>(wrapping<Backend::INT8, IDX>(int32_t(vr) + int32_t(vi)));
}

// launcher for V in {int2, int64x2_t, float2, double2}
template <unsigned IDX, typename V> __device__ __forceinline__ void mod_launch(
    char4 *__restrict__ out_r,
    char4 *__restrict__ out_i,
    char4 *__restrict__ out_ri,
    V v0, V v1, V v2, V v3 //
) {
    char4 rem_r;
    rem_r.x  = static_cast<int8_t>(calc_mod<Backend::INT8, IDX>(v0.x));
    rem_r.y  = static_cast<int8_t>(calc_mod<Backend::INT8, IDX>(v1.x));
    rem_r.z  = static_cast<int8_t>(calc_mod<Backend::INT8, IDX>(v2.x));
    rem_r.w  = static_cast<int8_t>(calc_mod<Backend::INT8, IDX>(v3.x));
    out_r[0] = rem_r;

    char4 rem_i;
    rem_i.x  = static_cast<int8_t>(calc_mod<Backend::INT8, IDX>(v0.y));
    rem_i.y  = static_cast<int8_t>(calc_mod<Backend::INT8, IDX>(v1.y));
    rem_i.z  = static_cast<int8_t>(calc_mod<Backend::INT8, IDX>(v2.y));
    rem_i.w  = static_cast<int8_t>(calc_mod<Backend::INT8, IDX>(v3.y));
    out_i[0] = rem_i;

    char4 rem_ri;
    rem_ri.x  = static_cast<int8_t>(wrapping<Backend::INT8, IDX>(int32_t(rem_r.x) + int32_t(rem_i.x)));
    rem_ri.y  = static_cast<int8_t>(wrapping<Backend::INT8, IDX>(int32_t(rem_r.y) + int32_t(rem_i.y)));
    rem_ri.z  = static_cast<int8_t>(wrapping<Backend::INT8, IDX>(int32_t(rem_r.z) + int32_t(rem_i.z)));
    rem_ri.w  = static_cast<int8_t>(wrapping<Backend::INT8, IDX>(int32_t(rem_r.w) + int32_t(rem_i.w)));
    out_ri[0] = rem_ri;
}

} // namespace gemmul8::mod
