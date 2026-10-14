// LFM2 (lfm2_moe) glue: the short convolution (kernel 3, depthwise, causal) and the sigmoid router with expert bias.

// transformers' Lfm2MoeShortConv in bf16: bx = bf16(B * x); y = bf16(conv3(bx)); out = bf16(C * y).
// A thread owns one channel and walks the rows in order, so rows of a verify window or a prompt chunk chain exactly.

#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_lfm2 {

constexpr int NT = 256;

__device__ __forceinline__ float ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }

}  // namespace tf_lfm2

using namespace tf_lfm2;

// bcx[r * ld + {0, h, 2h} + c] = B, C, x of row r. w[c * 3 + j] taps (j = 2 hits the newest row).
// state[{0, 1} * h + c]: the bx of the two rows before row 0 (1 is the newer). hist[r * h + c] gets row r's bx.
// out[r * h + c] = C * conv. commit >= 0 moves state past the first `commit` rows (commit <= rows).
extern "C" __global__ void __launch_bounds__(NT) tf_lfm2_conv_rows(const __nv_bfloat16* __restrict__ bcx, long long ld_,
                                                                   const __nv_bfloat16* __restrict__ w,
                                                                   __nv_bfloat16* __restrict__ state,
                                                                   __nv_bfloat16* __restrict__ hist,
                                                                   __nv_bfloat16* __restrict__ out, int rows, int h,
                                                                   int commit) {
    const int c = blockIdx.x * NT + threadIdx.x;
    if (c >= h) return;
    const float w0 = ld(w + 3 * c), w1 = ld(w + 3 * c + 1), w2 = ld(w + 3 * c + 2);
    float s0 = ld(state + c), s1 = ld(state + h + c);  // older, newer
    float k0 = s0, k1 = s1;
    for (int r = 0; r < rows; ++r) {
        const __nv_bfloat16* row = bcx + r * ld_;
        const float b = ld(row + c), cc = ld(row + h + c), x = ld(row + 2 * h + c);
        const __nv_bfloat16 bxh = __float2bfloat16_rn(__fmul_rn(b, x));
        const float bx = __bfloat162float(bxh);
        if (hist) hist[(long long)r * h + c] = bxh;
        float y = __fmul_rn(w0, s0);
        y = __fmaf_rn(w1, s1, y);
        y = __fmaf_rn(w2, bx, y);
        const float yb = __bfloat162float(__float2bfloat16_rn(y));
        out[(long long)r * h + c] = __float2bfloat16_rn(__fmul_rn(cc, yb));
        s0 = s1;
        s1 = bx;
        if (r + 1 == commit) k0 = s0, k1 = s1;
    }
    if (commit > 0) {
        state[c] = __float2bfloat16_rn(k0);
        state[h + c] = __float2bfloat16_rn(k1);
    }
}

// After a verify window: state moves past the first *accepted rows of hist (read on the device, 0 keeps state).
extern "C" __global__ void __launch_bounds__(NT) tf_lfm2_conv_commit(__nv_bfloat16* __restrict__ state,
                                                                     const __nv_bfloat16* __restrict__ hist,
                                                                     const int* __restrict__ accepted, int h) {
    const int c = blockIdx.x * NT + threadIdx.x;
    if (c >= h) return;
    const int n = *accepted;
    if (n <= 0) return;
    const __nv_bfloat16 newer = hist[(long long)(n - 1) * h + c];
    const __nv_bfloat16 older = n >= 2 ? hist[(long long)(n - 2) * h + c] : state[h + c];
    state[c] = older;
    state[h + c] = newer;
}

// Router: logits[r, e] = fp32 x[r] . w[e] (w fp32), score = sigmoid; the top k of score + bias[e] (lower index on a tie)
// with weights score / (sum + 1e-6). A block a row, a warp 8 experts (E <= 64, k <= 8): picks/wts [r, k].
extern "C" __global__ void __launch_bounds__(NT) tf_lfm2_route(const __nv_bfloat16* __restrict__ x, int x_ld,
                                                               const float* __restrict__ w,
                                                               const float* __restrict__ bias, int E, int D, int k,
                                                               int* __restrict__ picks, float* __restrict__ wts) {
    __shared__ float score[64];
    const int r = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const __nv_bfloat16* xr = x + (long long)r * x_ld;
    for (int j = 0; j < 8; ++j) {
        const int e = warp * 8 + j;
        if (e >= E) break;
        const float* wr = w + (long long)e * D;
        float acc = 0.f;
        for (int c = lane * 4; c < D; c += 128) {
            const float4 wv = *reinterpret_cast<const float4*>(wr + c);
            const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(xr + c);
            const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(xr + c + 2);
            acc = __fmaf_rn(__low2float(a), wv.x, acc);
            acc = __fmaf_rn(__high2float(a), wv.y, acc);
            acc = __fmaf_rn(__low2float(b), wv.z, acc);
            acc = __fmaf_rn(__high2float(b), wv.w, acc);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
        if (lane == 0) score[e] = __fdiv_rn(1.f, __fadd_rn(1.f, expf(-acc)));
    }
    __syncthreads();
    if (warp != 0) return;
    float s0 = lane < E ? score[lane] : 0.f, s1 = lane + 32 < E ? score[lane + 32] : 0.f;
    float v0 = lane < E ? __fadd_rn(s0, bias[lane]) : -INFINITY;
    float v1 = lane + 32 < E ? __fadd_rn(s1, bias[lane + 32]) : -INFINITY;
    float ps[8];
    int ids[8];
    float total = 0.f;
    for (int t = 0; t < k; ++t) {
        float bv = v0;
        int bi = lane;
        if (v1 > bv) bv = v1, bi = lane + 32;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
        }
        if (bi == lane) v0 = -INFINITY;
        if (bi == lane + 32) v1 = -INFINITY;
        ids[t] = bi;
        ps[t] = score[bi];
        total = __fadd_rn(total, ps[t]);
    }
    if (lane != 0) return;
    const float den = __fadd_rn(total, 1e-6f);
    for (int t = 0; t < k; ++t) {
        picks[r * k + t] = ids[t];
        wts[r * k + t] = __fdiv_rn(ps[t], den);
    }
}
