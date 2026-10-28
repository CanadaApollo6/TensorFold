#include "kv4.cuh"
#include <stdint.h>

using namespace tf_kv4;
constexpr int chunk = 512;

extern "C" __global__ void tf_kv4_pack(const __nv_bfloat16* x, unsigned char* out, int rows, int dim) {
    const int group = blockIdx.x * 4 + threadIdx.x / 32;
    if (group >= rows * (dim / 32)) return;
    const int d = (group % (dim / 32)) * 32 + (threadIdx.x & 31);
    quant(__bfloat162float(x[group * 32 + (threadIdx.x & 31)]), out + (group / (dim / 32)) * (dim / 2 + dim / 16), d, dim);
}

extern "C" __global__ void tf_kv4_unpack(const unsigned char* x, __nv_bfloat16* out, int rows, int dim) {
    const int group = blockIdx.x * 4 + threadIdx.x / 32;
    if (group >= rows * (dim / 32)) return;
    const int d = (group % (dim / 32)) * 32 + (threadIdx.x & 31);
    const float v = value(x + (group / (dim / 32)) * (dim / 2 + dim / 16), d, dim);
    out[group * 32 + (threadIdx.x & 31)] = __float2bfloat16_rn(turn(v));
}

template <int D>
__device__ void partial(const __nv_bfloat16* q, const unsigned char* kc, const unsigned char* vc,
                       const __nv_bfloat16* first_k, const __nv_bfloat16* first_v,
                       const __nv_bfloat16* last_k, const __nv_bfloat16* last_v,
                       float* scratch, int heads, int kvheads, int keys, int groups) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const int head = blockIdx.y, kh = head / (heads / kvheads);
    const int begin = blockIdx.x * chunk + warp * (chunk / 4);
    const int end = min(begin + chunk / 4, keys);
    constexpr int row_bytes = D / 2 + D / 16;
    float query[D / 32], out[D / 32];
#pragma unroll
    for (int i = 0; i < D / 32; ++i) {
        query[i] = turn(__bfloat162float(q[head * D + 32 * i + lane]));
        out[i] = 0.0f;
    }
    float maximum = -INFINITY, sum = 0.0f;
    const float scale = rsqrtf(float(D));
    for (int t = begin; t < end; ++t) {
        float key[D / 32], val[D / 32], dot = 0.0f;
#pragma unroll
        for (int i = 0; i < D / 32; ++i) {
            const int d = 32 * i + lane;
            if ((t == 0 && first_k) || (t == keys - 1 && last_k)) {
                const __nv_bfloat16* k = t == 0 && first_k ? first_k : last_k;
                const __nv_bfloat16* v = t == 0 && first_v ? first_v : last_v;
                key[i] = turn(__bfloat162float(k[kh * D + d]));
                val[i] = turn(__bfloat162float(v[kh * D + d]));
            } else {
                const size_t row = ((size_t)t * kvheads + kh) * row_bytes;
                key[i] = value(kc + row, d, D);
                val[i] = value(vc + row, d, D);
            }
            dot = fmaf(query[i], key[i], dot);
        }
#pragma unroll
        for (int bit = 16; bit; bit /= 2) dot += __shfl_xor_sync(0xffffffffu, dot, bit);
        dot *= scale;
        const float next = fmaxf(maximum, dot);
        const float a = expf(maximum - next), b = expf(dot - next);
        sum = sum * a + b;
#pragma unroll
        for (int i = 0; i < D / 32; ++i) out[i] = out[i] * a + val[i] * b;
        maximum = next;
    }
    const size_t at = ((size_t)head * groups * 4 + blockIdx.x * 4 + warp) * (D + 2);
#pragma unroll
    for (int i = 0; i < D / 32; ++i) scratch[at + 32 * i + lane] = out[i];
    if (lane == 0) {
        scratch[at + D] = maximum;
        scratch[at + D + 1] = sum;
    }
}

#define ARGS const __nv_bfloat16* q, const unsigned char* kc, const unsigned char* vc, \
             const __nv_bfloat16* fk, const __nv_bfloat16* fv, const __nv_bfloat16* lk, \
             const __nv_bfloat16* lv, float* scratch, int heads, int kvheads, int keys, int groups
#define PASS q, kc, vc, fk, fv, lk, lv, scratch, heads, kvheads, keys, groups
extern "C" __global__ void tf_kv4_partial128(ARGS) { partial<128>(PASS); }
extern "C" __global__ void tf_kv4_partial256(ARGS) { partial<256>(PASS); }

extern "C" __global__ void tf_kv4_merge(const float* scratch, __nv_bfloat16* out, int dim, int groups) {
    const int d = threadIdx.x, head = blockIdx.x;
    const size_t base = (size_t)head * groups * 4 * (dim + 2);
    float maximum = -INFINITY, sum = 0.0f, value = 0.0f;
    for (int i = 0; i < groups * 4; ++i) {
        const float* p = scratch + base + (size_t)i * (dim + 2);
        if (p[dim + 1] == 0.0f) continue;
        const float next = fmaxf(maximum, p[dim]);
        const float a = expf(maximum - next), b = expf(p[dim] - next);
        value = value * a + p[d] * b;
        sum = sum * a + p[dim + 1] * b;
        maximum = next;
    }
    out[head * dim + d] = __float2bfloat16_rn(turn(value / sum));
}
