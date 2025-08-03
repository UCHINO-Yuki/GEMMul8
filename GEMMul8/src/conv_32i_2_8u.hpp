#pragma once
#include "common.hpp"
#include "table.hpp"

namespace oz2_util {

__global__ void conv_32i_2_8u_256_kernel(const size_t sizeC,                     // ((m * n + 15) >> 4) << 4; // multiple of 16
                                         const int32_t *const __restrict__ C32i, // input
                                         uint8_t *const __restrict__ C8u)        // output
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;

    int4 in = reinterpret_cast<const int4 *>(C32i)[idx];

    uchar4 out{static_cast<unsigned char>(in.x),
               static_cast<unsigned char>(in.y),
               static_cast<unsigned char>(in.z),
               static_cast<unsigned char>(in.w)};

    reinterpret_cast<uchar4 *>(C8u)[idx] = out;
}

__global__ void conv_32i_2_8u_not256_kernel(const size_t sizeC,                     // ((m * n + 15) >> 4) << 4; // multiple of 16
                                            const int32_t *const __restrict__ C32i, // input
                                            const uint8_t modulus,                  // <= 256
                                            const int32_t invm,                     // 2^32 / modulus - 1
                                            uint8_t *const __restrict__ C8u)        // output
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;

    int4 in = reinterpret_cast<const int4 *>(C32i)[idx];

    // in.x -= __mulhi(in.x, invm) * modulus;
    // in.x -= (in.x >= modulus) * modulus;
    // in.x += (in.x < 0) * modulus;
    // in.y -= __mulhi(in.y, invm) * modulus;
    // in.y -= (in.y >= modulus) * modulus;
    // in.y += (in.y < 0) * modulus;
    // in.z -= __mulhi(in.z, invm) * modulus;
    // in.z -= (in.z >= modulus) * modulus;
    // in.z += (in.z < 0) * modulus;
    // in.w -= __mulhi(in.w, invm) * modulus;
    // in.w -= (in.w >= modulus) * modulus;
    // in.w += (in.w < 0) * modulus;

    const int32_t q0 = __mulhi(in.x, invm);
    const int32_t q1 = __mulhi(in.y, invm);
    const int32_t q2 = __mulhi(in.z, invm);
    const int32_t q3 = __mulhi(in.w, invm);

    in.x -= q0 * modulus;
    in.y -= q1 * modulus;
    in.z -= q2 * modulus;
    in.w -= q3 * modulus;

    in.x += (in.x >= modulus) ? -modulus : ((in.x < 0) ? modulus : 0);
    in.y += (in.y >= modulus) ? -modulus : ((in.y < 0) ? modulus : 0);
    in.z += (in.z >= modulus) ? -modulus : ((in.z < 0) ? modulus : 0);
    in.w += (in.w >= modulus) ? -modulus : ((in.w < 0) ? modulus : 0);

    uchar4 out{static_cast<unsigned char>(in.x),
               static_cast<unsigned char>(in.y),
               static_cast<unsigned char>(in.z),
               static_cast<unsigned char>(in.w)};

    reinterpret_cast<uchar4 *>(C8u)[idx] = out;
}

// interface!!
__inline__ void conv_32i_2_8u(const unsigned i,          //
                              const size_t sizeC,        // m*n/16*16
                              const int32_t *const C32i, // input
                              uint8_t *const C8u)        // output
{
    if (i == 0) {
        conv_32i_2_8u_256_kernel<<<oz2_const::grids_conv32i8u, oz2_const::threads_conv32i8u>>>(sizeC >> 2, C32i, C8u);
    } else {
        const uint8_t modulus = static_cast<uint8_t>(-oz2_table::moduli[i].z);
        const int32_t invm    = oz2_table::invm_32i[i - 1];
        conv_32i_2_8u_not256_kernel<<<oz2_const::grids_conv32i8u, oz2_const::threads_conv32i8u>>>(sizeC >> 2, C32i, modulus, invm, C8u);
    }
}

} // namespace oz2_util
