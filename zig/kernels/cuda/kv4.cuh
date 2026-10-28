#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tf_kv4 {
constexpr float inv_sqrt32 = 0.17677669529663688110f;

// Qwen27's H32 midpoint format stays independent of head geometry, RoPE and output gating.
__device__ __forceinline__ float turn(float x) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int bit = 1; bit < 32; bit *= 2) {
        const float other = __shfl_xor_sync(0xffffffffu, x, bit);
        x = (lane & bit) ? __fsub_rn(other, x) : __fadd_rn(x, other);
    }
    return __fmul_rn(x, inv_sqrt32);
}

__device__ __forceinline__ float value(const unsigned char* row, int d, int dim) {
    const unsigned b = row[d / 2];
    const int code = (b >> (4 * (d & 1))) & 15;
    const float scale = __half2float(*reinterpret_cast<const __half*>(row + dim / 2 + 2 * (d / 32)));
    return (code - 7.5f) * (scale * 0.125f);
}

__device__ __forceinline__ void quant(float x, unsigned char* row, int d, int dim) {
    const int lane = threadIdx.x & 31;
    const float y = turn(x);
    float scale = fabsf(y);
#pragma unroll
    for (int bit = 16; bit; bit /= 2) scale = fmaxf(scale, __shfl_xor_sync(0xffffffffu, scale, bit));
    const bool bad = __any_sync(0xffffffffu, !isfinite(y));
    const float divisor = __fadd_rn(scale, 1e-10f);
    const float inverse = __fdiv_rn(1.0f, divisor);
    const unsigned code = (unsigned)fminf(fmaxf(floorf(__fmul_rn(__fmul_rn(y, inverse), 8.0f)) + 8.0f, 0.0f), 15.0f);
    const unsigned next = __shfl_down_sync(0xffffffffu, code, 1);
    if (!(lane & 1)) row[d / 2] = code | (next << 4);
    if (lane == 0) {
        const __half s = __float2half_rn(scale);
        *reinterpret_cast<__half*>(row + dim / 2 + 2 * (d / 32)) =
            bad || !isfinite(__half2float(s)) ? __float2half_rn(nanf("")) : s;
    }
}
}
