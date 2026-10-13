// The 27B's own small kernels: batched state copies, the drafter's taps, its fp32 RMSNorm, the MTP head's attention.
#include <cuda_bf16.h>
#include <stdint.h>

struct TfCopy {
    unsigned long long dst, src, bytes;  // 16-byte aligned pointers, bytes a multiple of 16
};

// One block an item, its threads striding through the item's 16-byte words.
extern "C" __global__ void __launch_bounds__(256) tf_copy_batch(const TfCopy* __restrict__ items) {
    const TfCopy it = items[blockIdx.x];
    const uint4* src = reinterpret_cast<const uint4*>(it.src);
    uint4* dst = reinterpret_cast<uint4*>(it.dst);
    const unsigned long long n = it.bytes / 16;
    for (unsigned long long i = threadIdx.x + static_cast<unsigned long long>(blockIdx.y) * blockDim.x; i < n;
         i += static_cast<unsigned long long>(blockDim.x) * gridDim.y)
        dst[i] = src[i];
}

// forward.tree_forward's taps: (x.float() + pending.float()).to(bfloat16), row r into columns [col, col + D) of OUT.
extern "C" __global__ void __launch_bounds__(256) tf_tap(const __nv_bfloat16* __restrict__ x,
                                                         const __nv_bfloat16* __restrict__ pending,
                                                         __nv_bfloat16* __restrict__ out, int D, int out_stride, int col) {
    const long long row = blockIdx.x;
    for (int i = threadIdx.x + blockIdx.y * blockDim.x; i < D; i += blockDim.x * gridDim.y)
        out[row * out_stride + col + i] =
            __float2bfloat16_rn(__bfloat162float(x[row * D + i]) + __bfloat162float(pending[row * D + i]));
}

// The drafter's RMSNorm in fp32: y = bf16(x * rsqrt(mean(x^2) + eps) * w), a 256-thread block a row.
extern "C" __global__ void __launch_bounds__(256) tf_rms_norm(const __nv_bfloat16* __restrict__ x, int x_stride,
                                                              const __nv_bfloat16* __restrict__ w,
                                                              __nv_bfloat16* __restrict__ y, int D, float eps) {
    __shared__ float part[8];
    const long long row = blockIdx.x;
    const __nv_bfloat16* xr = x + row * x_stride;
    float s = 0.0f;
    for (int i = threadIdx.x; i < D; i += 256) {
        const float v = __bfloat162float(xr[i]);
        s += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
    __syncthreads();
    float total = 0.0f;
    for (int k = 0; k < 8; ++k) total += part[k];
    const float inv = rsqrtf(total / D + eps);
    for (int i = threadIdx.x; i < D; i += 256)
        y[row * D + i] = __float2bfloat16_rn(__bfloat162float(xr[i]) * inv * __bfloat162float(w[i]));
}

// The drafter's inverse frequencies as dflash2 makes them on the GPU: 1 / theta ** (2i / dim), fp32 powf then a divide.
extern "C" __global__ void tf_inv_freq(float* __restrict__ inv, float theta, int half, int dim) {
    const int i = threadIdx.x;
    if (i < half) inv[i] = 1.0f / powf(theta, static_cast<float>(i) * 2.0f / static_cast<float>(dim));
}

// dflash2._rotary and add_taps_streams' tables: cos and sin of float(position) * inv_freq, fp32, a block a row.
extern "C" __global__ void tf_rotary(const int* __restrict__ pos, const float* __restrict__ inv,
                                     float* __restrict__ cos_out, float* __restrict__ sin_out, int half) {
    const int row = blockIdx.x;
    const float p = static_cast<float>(pos[row]);
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float phase = p * inv[i];
        cos_out[row * half + i] = cosf(phase);
        sin_out[row * half + i] = sinf(phase);
    }
}

// The MTP head's few-row attention, head dim 256: a block a (key head, SPLIT keys), a warp a query head, a lane 8 dims.
// Row r sees slots [base, at + r]; each split's (max, sum, unnormalized output) goes to part for tf_window_merge.
extern "C" __global__ void __launch_bounds__(256) tf_window_part(const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v, float* __restrict__ part_o,
    float* __restrict__ part_ml, int at, int base, int W, int H, int HK, int SPLIT, float scale) {
    constexpr int D = 256;
    const int lane = threadIdx.x & 31, hk = blockIdx.x, split = blockIdx.y, G = H / HK;
    const int k0 = base + split * SPLIT, kv_row = HK * D;
    for (int hh = threadIdx.x >> 5; hh < G; hh += 8) {
        const int h = hk * G + hh;
        for (int r = 0; r < W; ++r) {
            const uint4 qw = *reinterpret_cast<const uint4*>(q + ((size_t)r * H + h) * D + lane * 8);
            const uint32_t qv[4] = {qw.x, qw.y, qw.z, qw.w};
            float qf[8], o[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                qf[2 * j] = __uint_as_float(qv[j] << 16);
                qf[2 * j + 1] = __uint_as_float(qv[j] & 0xFFFF0000u);
                o[2 * j] = o[2 * j + 1] = 0.f;
            }
            float m = -INFINITY, l = 0.f;
            const int k1 = min(k0 + SPLIT, at + r + 1);
            for (int s = k0; s < k1; ++s) {
                const uint4 kw = *reinterpret_cast<const uint4*>(k + (size_t)s * kv_row + hk * D + lane * 8);
                const uint4 vw = *reinterpret_cast<const uint4*>(v + (size_t)s * kv_row + hk * D + lane * 8);
                const uint32_t kv[4] = {kw.x, kw.y, kw.z, kw.w};
                float dot = 0.f;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    dot = fmaf(qf[2 * j], __uint_as_float(kv[j] << 16), dot);
                    dot = fmaf(qf[2 * j + 1], __uint_as_float(kv[j] & 0xFFFF0000u), dot);
                }
#pragma unroll
                for (int x = 16; x > 0; x >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, x);
                const float sc = dot * scale, mn = fmaxf(m, sc), a = __expf(m - mn), p = __expf(sc - mn);
                l = l * a + p;
                m = mn;
                const uint32_t vv[4] = {vw.x, vw.y, vw.z, vw.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    o[2 * j] = fmaf(p, __uint_as_float(vv[j] << 16), o[2 * j] * a);
                    o[2 * j + 1] = fmaf(p, __uint_as_float(vv[j] & 0xFFFF0000u), o[2 * j + 1] * a);
                }
            }
            const size_t at_ = ((size_t)split * W + r) * H + h;
            float4* po = reinterpret_cast<float4*>(part_o + at_ * D + lane * 8);
            po[0] = make_float4(o[0], o[1], o[2], o[3]);
            po[1] = make_float4(o[4], o[5], o[6], o[7]);
            if (lane == 0) {
                part_ml[2 * at_] = m;
                part_ml[2 * at_ + 1] = l;
            }
        }
    }
}

// Folds tf_window_part's splits: a block a (row, query head), a thread a dim; out (W, H, 256) bf16.
extern "C" __global__ void __launch_bounds__(256) tf_window_merge(const float* __restrict__ part_o,
    const float* __restrict__ part_ml, __nv_bfloat16* __restrict__ out, int splits, int WH) {
    const int rh = blockIdx.x, d = threadIdx.x;
    float m = -INFINITY;
    for (int s = 0; s < splits; ++s) m = fmaxf(m, part_ml[2 * ((size_t)s * WH + rh)]);
    float l = 0.f, o = 0.f;
    for (int s = 0; s < splits; ++s) {
        const size_t i = (size_t)s * WH + rh;
        const float ms = part_ml[2 * i];
        if (ms == -INFINITY) continue;
        const float a = __expf(ms - m);
        l += part_ml[2 * i + 1] * a;
        o += part_o[i * 256 + d] * a;
    }
    out[(size_t)rh * 256 + d] = __float2bfloat16_rn(o / l);
}

// A chained MTP draft: the head's argmax column through the draft vocabulary (map null: every token), as the next
// step's id and the chain's draft `slot`.
extern "C" __global__ void tf_draft_pick(const int* __restrict__ pick, const unsigned* __restrict__ map,
                                         int* __restrict__ ids, unsigned* __restrict__ drafts, int slot) {
    if (threadIdx.x != 0) return;
    const unsigned t = map ? map[pick[0]] : (unsigned)pick[0];
    ids[0] = (int)t;
    drafts[slot] = t;
}
