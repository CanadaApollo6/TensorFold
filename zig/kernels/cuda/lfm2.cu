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

// Router: score[r, e] = sigmoid(fp32 x[r] . w[e]) (w fp32 [E, D]), then each row's top k of score + bias[e] (lower
// index on a tie) with weights score / (sum + 1e-6): picks/wts [r, k]. A block an (expert, run of RR rows): its threads
// hold the expert's row (D <= 1024 * DQ) and dot it with each row; the run's last block to finish picks for the run
// (counters [runs], zero between launches; scores [rows, E] fp32 scratch). E <= 64, k <= 8.
template <int DQ, int RR>
__device__ __forceinline__ void route_run(const __nv_bfloat16* __restrict__ x, int x_ld, const float* __restrict__ w,
                                          const float* __restrict__ bias, int E, int D, int k, int rows,
                                          float* __restrict__ scores, int* __restrict__ counters,
                                          int* __restrict__ picks, float* __restrict__ wts) {
    __shared__ float part[NT / 32][RR];
    __shared__ int last;
    const int e = blockIdx.x, r0 = blockIdx.y * RR, n = min(RR, rows - r0);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float4 wv[DQ];
#pragma unroll
    for (int j = 0; j < DQ; ++j) {
        const int c = 4 * threadIdx.x + 1024 * j;
        wv[j] = c < D ? __ldg(reinterpret_cast<const float4*>(w + (long long)e * D + c)) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    for (int r = 0; r < n; ++r) {
        const __nv_bfloat16* xr = x + (long long)(r0 + r) * x_ld;
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < DQ; ++j) {
            const int c = 4 * threadIdx.x + 1024 * j;
            if (c < D) {
                const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(xr + c);
                const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(xr + c + 2);
                acc = __fmaf_rn(__low2float(a), wv[j].x, acc);
                acc = __fmaf_rn(__high2float(a), wv[j].y, acc);
                acc = __fmaf_rn(__low2float(b), wv[j].z, acc);
                acc = __fmaf_rn(__high2float(b), wv[j].w, acc);
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
        if (lane == 0) part[warp][r] = acc;
    }
    __syncthreads();
    if (threadIdx.x < n) {
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < NT / 32; ++i) acc = __fadd_rn(acc, part[i][threadIdx.x]);
        scores[(long long)(r0 + threadIdx.x) * E + e] = __fdiv_rn(1.f, __fadd_rn(1.f, expf(-acc)));
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(counters + blockIdx.y, 1) == E - 1;
        if (last) counters[blockIdx.y] = 0;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int r = warp; r < n; r += NT / 32) {
        const float* sr = scores + (long long)(r0 + r) * E;
        const float s0 = lane < E ? __ldcg(sr + lane) : 0.f, s1 = lane + 32 < E ? __ldcg(sr + lane + 32) : 0.f;
        float v0 = lane < E ? __fadd_rn(s0, bias[lane]) : -INFINITY;
        float v1 = lane + 32 < E ? __fadd_rn(s1, bias[lane + 32]) : -INFINITY;
        float total = 0.f, ps = 0.f;
        int mine = 0;
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
            const float p = __shfl_sync(0xffffffffu, bi < 32 ? s0 : s1, bi & 31);
            total = __fadd_rn(total, p);
            if (lane == t) ps = p, mine = bi;
        }
        if (lane < k) {
            picks[(r0 + r) * k + lane] = mine;
            wts[(r0 + r) * k + lane] = __fdiv_rn(ps, __fadd_rn(total, 1e-6f));
        }
    }
}

extern "C" __global__ void __launch_bounds__(NT) tf_lfm2_route(const __nv_bfloat16* __restrict__ x, int x_ld,
                                                               const float* __restrict__ w,
                                                               const float* __restrict__ bias, int E, int D, int k,
                                                               int rows, float* __restrict__ scores,
                                                               int* __restrict__ counters, int* __restrict__ picks,
                                                               float* __restrict__ wts) {
    route_run<2, 16>(x, x_ld, w, bias, E, D, k, rows, scores, counters, picks, wts);
}

// A graphed decode step's tail: the pick becomes the next input id at the next position, and log[pos] keeps it.
extern "C" __global__ void tf_lfm2_next(const int* __restrict__ picks, int* __restrict__ ids, int* __restrict__ pos,
                                        int* __restrict__ log) {
    const int p = pos[0], t = picks[0];
    log[p] = t;
    ids[0] = t;
    pos[0] = p + 1;
}

// Decode attention, head dim 64 and 4 query heads a key head: a block a (row, key head, split). Split s of S takes
// keys [s c, s c + c) of the row's pos + 1 (c a multiple of 32, from the device's pos, so a graph replays it at any
// position) in 32-key tiles staged in shared memory, a warp a query head with an online softmax; the row's last
// block to finish folds the S partials in split order. Keys and values [capacity, hk, 64] bf16, q and out
// [rows, 4 hk, 64] bf16; po [rows, 4 hk, S, 64], pm and pl [rows, 4 hk, S] fp32; counters [rows * hk] zero between
// launches.
constexpr int AD = 64, AG = 4, AK = 32, AKS = AD + 8;

__device__ __forceinline__ void bf8(const uint4 u, float* f) {
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) f[2 * i] = __low2float(b[i]), f[2 * i + 1] = __high2float(b[i]);
}

extern "C" __global__ void __launch_bounds__(AG * 32) tf_lfm2_attn(const __nv_bfloat16* __restrict__ q,
                                                                   const __nv_bfloat16* __restrict__ kc,
                                                                   const __nv_bfloat16* __restrict__ vc,
                                                                   const int* __restrict__ pos, int hk, float scale,
                                                                   float* __restrict__ po, float* __restrict__ pm,
                                                                   float* __restrict__ pl, int* __restrict__ counters,
                                                                   __nv_bfloat16* __restrict__ out) {
    __shared__ __align__(16) __nv_bfloat16 ks[AK * AKS];
    __shared__ __align__(16) __nv_bfloat16 vs[AK * AD];
    __shared__ int last;
    const int r = blockIdx.x, kh = blockIdx.y, s = blockIdx.z, S = gridDim.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, h = AG * hk, head = kh * AG + warp;
    const int n = pos[r] + 1;
    const int c = ((n + S - 1) / S + AK - 1) / AK * AK;
    const int k0 = s * c, k1 = min(n, k0 + c);
    float qv[AD];
    const __nv_bfloat16* qr = q + ((long long)r * h + head) * AD;
#pragma unroll
    for (int e = 0; e < AD; e += 8) bf8(*reinterpret_cast<const uint4*>(qr + e), qv + e);
    float m = -INFINITY, l = 0.f, a0 = 0.f, a1 = 0.f;
    const int lk = threadIdx.x >> 3, lc = threadIdx.x & 7;
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    uint4 pk[2], pv[2];
    auto fetch = [&](int t0) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int key = t0 + lk + 16 * i;
            const long long at = ((long long)key * hk + kh) * AD + lc * 8;
            pk[i] = key < k1 ? __ldg(reinterpret_cast<const uint4*>(kc + at)) : zero;
            pv[i] = key < k1 ? __ldg(reinterpret_cast<const uint4*>(vc + at)) : zero;
        }
    };
    if (k0 < k1) fetch(k0);
    for (int t0 = k0; t0 < k1; t0 += AK) {
        __syncthreads(); // the last tile's readers are done
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            *reinterpret_cast<uint4*>(ks + (lk + 16 * i) * AKS + lc * 8) = pk[i];
            *reinterpret_cast<uint4*>(vs + (lk + 16 * i) * AD + lc * 8) = pv[i];
        }
        __syncthreads();
        if (t0 + AK < k1) fetch(t0 + AK);
        float sc = 0.f;
#pragma unroll
        for (int cc = 0; cc < AD / 8; ++cc) {
            float kf[8];
            bf8(*reinterpret_cast<const uint4*>(ks + lane * AKS + 8 * cc), kf);
#pragma unroll
            for (int i = 0; i < 8; ++i) sc = __fmaf_rn(qv[8 * cc + i], kf[i], sc);
        }
        const bool in = t0 + lane < k1;
        sc = in ? __fmul_rn(sc, scale) : -INFINITY;
        float mt = sc;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, o));
        const float mn = fmaxf(m, mt);
        const float corr = m == -INFINITY ? 0.f : expf(__fsub_rn(m, mn));
        const float p = in ? expf(__fsub_rn(sc, mn)) : 0.f;
        float ps = p;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ps = __fadd_rn(ps, __shfl_xor_sync(0xffffffffu, ps, o));
        l = __fmaf_rn(l, corr, ps);
        a0 = __fmul_rn(a0, corr);
        a1 = __fmul_rn(a1, corr);
#pragma unroll 8
        for (int j = 0; j < AK; ++j) {
            const float pj = __shfl_sync(0xffffffffu, p, j);
            const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(vs + j * AD + 2 * lane);
            a0 = __fmaf_rn(pj, __low2float(v), a0);
            a1 = __fmaf_rn(pj, __high2float(v), a1);
        }
        m = mn;
    }
    const long long p0 = ((long long)r * h + head) * S;
    *reinterpret_cast<float2*>(po + (p0 + s) * AD + 2 * lane) = make_float2(a0, a1);
    if (lane == 0) pm[p0 + s] = m, pl[p0 + s] = l;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(counters + r * hk + kh, 1) == S - 1;
        if (last) counters[r * hk + kh] = 0;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    float M = -INFINITY;
    for (int j = 0; j < S; ++j) M = fmaxf(M, __ldcg(pm + p0 + j));
    float b0 = 0.f, b1 = 0.f, L = 0.f;
    for (int j = 0; j < S; ++j) {
        const float mj = __ldcg(pm + p0 + j);
        if (mj == -INFINITY) continue;
        const float w = expf(__fsub_rn(mj, M));
        const float2 o = __ldcg(reinterpret_cast<const float2*>(po + (p0 + j) * AD + 2 * lane));
        b0 = __fmaf_rn(w, o.x, b0);
        b1 = __fmaf_rn(w, o.y, b1);
        L = __fmaf_rn(w, __ldcg(pl + p0 + j), L);
    }
    *reinterpret_cast<__nv_bfloat162*>(out + ((long long)r * h + head) * AD + 2 * lane) =
        __floats2bfloat162_rn(__fdiv_rn(b0, L), __fdiv_rn(b1, L));
}
