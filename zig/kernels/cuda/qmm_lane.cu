// The 3090's 4-bit decode matmul for the Zig engine: the 3090 stack's qmm.cu device code (TensorFold 0.3.6.3 with
// agent-training's qmm-lean.patch and qmm-inblock.patch), ATen dropped, with C entry points. Before sm_90 a tile's K
// slices meet in memory or in one block, added in slice order: 0.6.6's qmm.cu bits on a 3090 (no clusters).

#include <algorithm>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "qmm_lane_frag.cuh"

namespace {

using namespace qmm_frag;

// Tile shapes: BM rows by BN columns a block, WM x WN warps, each warp (BM / WM) x (BN / WN).
template <int GS, int BM, int BN, int WM, int WN, int STAGES>
struct Tile {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;               // m16 tiles a warp
    static constexpr int NT = BN / WN / 8;                // n8 tiles a warp
    static constexpr int ROW = GS * 2;                    // bytes of one input row a group
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int X = BM * ROW;                    // stage bytes: inputs,
    static constexpr int W = BN * GS / 2;                 // weights,
    static constexpr int S = BN * 2;                      // scales, biases (bf16),
    static constexpr int XS = BM * 4;                     // and input sums (fp32)
    static constexpr int STAGE = X + W + 2 * S + XS;
    static constexpr int PARTIALS = MT * NT * 4 * THREADS * 4;   // a K slice's partial, parked for the cluster sum
    static constexpr int SMEM = STAGES * STAGE > PARTIALS ? STAGES * STAGE : PARTIALS;
};

// A tile's K slice: acc = fma(xs, b, fma(p, s, acc)) per group of [g0, g0 + per) in order, from zero. Every launch form
// runs these steps, so a column's bits depend only on its K slices.
template <int GS, int BM, int BN, int WM, int WN, int STAGES, bool PIPE>
__device__ __forceinline__ void tile_slice(float (&acc)[Tile<GS, BM, BN, WM, WN, STAGES>::MT]
                                                      [Tile<GS, BM, BN, WM, WN, STAGES>::NT][4],
        unsigned char* buf, const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,
        const uint32_t* __restrict__ w, const __nv_bfloat16* __restrict__ scales,
        const __nv_bfloat16* __restrict__ biases, int M, int KG, int per, int g0, int m0, int n0, int npad, int ldx) {
    using T = Tile<GS, BM, BN, WM, WN, STAGES>;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    // one group's inputs, weights, scales, biases and input sums into stage s
    auto load = [&](int s, int g, int m0) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int row = min(m0 + r, M - 1);
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8,
                  m0 + r < M);
        }
        unsigned char* pw = p + T::X;
        constexpr int TILE_BYTES = 64 * GS / 2;           // one stored 64-column tile's group block
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
            const size_t tile = static_cast<size_t>(n0 / 64 + t) * KG + g;
            cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) + tile * TILE_BYTES + off * 16);
        }
        unsigned char* ps = pw + T::W;
        for (int c = tid; c < 2 * (T::S / 16); c += T::THREADS) {
            const int which = c / (T::S / 16), off = c % (T::S / 16);
            const __nv_bfloat16* src = (which ? biases : scales) + static_cast<size_t>(g) * npad + n0 + off * 8;
            cp16(ps + which * T::S + off * 16, src);
        }
        float* px = reinterpret_cast<float*>(ps + 2 * T::S);
        for (int r = tid; r < BM; r += T::THREADS)
            cp4(px + r, xs + static_cast<size_t>(min(m0 + r, M - 1)) * KG + g);
    };

    {
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
        // acc = fma(xs, b, fma(p, s, acc)) per output, the order every row gets
        auto epilogue = [&](float (&acc_)[T::MT][T::NT][4], const float (&d_)[T::MT][T::NT][4],
                            const float (&s_)[T::NT][2], const float (&b_)[T::NT][2], const float (&x_)[T::MT][2]) {
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int i = 0; i < T::MT; ++i)
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        acc_[i][j][e] = __fmaf_rn(x_[i][e >> 1], b_[j][e & 1], __fmaf_rn(d_[i][j][e], s_[j][e & 1],
                                                                                       acc_[i][j][e]));
        };
        float dp[T::MT][T::NT][4], sp[T::NT][2], bp[T::NT][2], xp[T::MT][2];
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < per) load(s, g0 + s, m0);
            commit();
        }
        for (int it = 0; it < per; ++it) {
            wait<STAGES - 2>();
            __syncthreads();
            const int next = it + STAGES - 1;
            if (next < per) load(next % STAGES, g0 + next, m0);
            commit();
            const unsigned char* p = stage(it % STAGES);
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
            const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + T::X + T::W);
            const float* px = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S);
            uint32_t words[T::NT][GS / 32];
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int v = 0; v < GS / 32; ++v)
                    words[j][v] = pw[((wn * T::NT + j) * 32 + lane) * (GS / 32) + v];
            float d[T::MT][T::NT][4];
#pragma unroll
            for (int kt = 0; kt < GS / 16; ++kt) {
                uint32_t a[T::MT][4];
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int ch = kt * 2 + (lane >> 4);
                    ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
                }
#pragma unroll
                for (int j = 0; j < T::NT; ++j) {
                    const uint32_t b0 = pair(words[j][kt / 2], (kt & 1) * 8);
                    const uint32_t b1 = pair(words[j][kt / 2], (kt & 1) * 8 + 4);
#pragma unroll
                    for (int i = 0; i < T::MT; ++i) {
                        if (kt == 0) mma0(d[i][j], a[i], b0, b1);
                        else mma(d[i][j], a[i], b0, b1);
                    }
                }
                if (PIPE && kt == 0 && it > 0) epilogue(acc, dp, sp, bp, xp);      // the previous group's, between MMAs
            }
            float sv[T::NT][2], bv[T::NT][2], xv[T::MT][2];
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const int col = wn * (BN / WN) + j * 8 + (lane & 3) * 2;
                const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(ps + col);
                const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(ps + BN + col);
                sv[j][0] = __low2float(s2); sv[j][1] = __high2float(s2);
                bv[j][0] = __low2float(b2); bv[j][1] = __high2float(b2);
            }
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int row = wm * (BM / WM) + i * 16 + (lane >> 2);
                xv[i][0] = px[row];
                xv[i][1] = px[row + 8];
            }
            if constexpr (PIPE) {
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
#pragma unroll
                    for (int j = 0; j < T::NT; ++j)
#pragma unroll
                        for (int e = 0; e < 4; ++e) dp[i][j][e] = d[i][j][e];
                    xp[i][0] = xv[i][0]; xp[i][1] = xv[i][1];
                }
#pragma unroll
                for (int j = 0; j < T::NT; ++j) {
                    sp[j][0] = sv[j][0]; sp[j][1] = sv[j][1]; bp[j][0] = bv[j][0]; bp[j][1] = bv[j][1];
                }
            } else {
                epilogue(acc, d, sv, bv, xv);
            }
        }
        if (PIPE && per > 0) epilogue(acc, dp, sp, bp, xp);
        wait<0>();
        __syncthreads();
    }
}

template <int GS, int BM, int BN, int WM, int WN, int STAGES, bool F32, bool CLUSTER, bool PIPE = false>
__device__ __forceinline__ void qmm_kernel(
        const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const uint32_t* __restrict__ w,
        const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases,
        void* __restrict__ out, float* __restrict__ part, int M, int N, int K, int SK, int npad, int ldx, int group) {
    using T = Tile<GS, BM, BN, WM, WN, STAGES>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, per = KG / SK;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y, slice = blockIdx.z, g0 = slice * per;
    {
        float acc[T::MT][T::NT][4];
        tile_slice<GS, BM, BN, WM, WN, STAGES, PIPE>(acc, buf, x, xs, w, scales, biases, M, KG, per, g0, m0, n0, npad,
                                                     ldx);
        if constexpr (CLUSTER) {
#if __CUDA_ARCH__ >= 900
            // K slices of a tile form a cluster: slice 0 adds peers' partials in slice order, as reduce_kernel does
            constexpr int E = T::MT * T::NT * 4;
            auto cluster = cooperative_groups::this_cluster();
            float* mine = reinterpret_cast<float*>(buf);
            if (slice != 0) {
#pragma unroll
                for (int i = 0; i < T::MT; ++i)
#pragma unroll
                    for (int j = 0; j < T::NT; ++j)
#pragma unroll
                        for (int e = 0; e < 4; ++e) mine[((i * T::NT + j) * 4 + e) * T::THREADS + tid] = acc[i][j][e];
            }
            cluster.sync();
            if (slice == 0) {
                for (int peer = 1; peer < SK; ++peer) {
                    const float* theirs = cluster.map_shared_rank(mine, peer);
#pragma unroll
                    for (int i = 0; i < T::MT; ++i)
#pragma unroll
                        for (int j = 0; j < T::NT; ++j)
#pragma unroll
                            for (int e = 0; e < 4; ++e)
                                acc[i][j][e] = acc[i][j][e] + theirs[((i * T::NT + j) * 4 + e) * T::THREADS + tid];
                }
            }
            cluster.sync();                               // peers keep their memory until slice 0 has read it
            if (slice != 0) return;
#else
            __trap();                                     // clusters are 9.0+; qmm_cuda sends slices to reduce_kernel
#endif
        }
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                    if (row >= M) continue;
                    const float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
                    if (SK > 1 && !CLUSTER) {
                        float* dst = part + (static_cast<size_t>(slice) * M + row) * N + col;
                        if (col < N) dst[0] = v0;
                        if (col + 1 < N) dst[1] = v1;
                    } else if (F32) {
                        float* dst = reinterpret_cast<float*>(out) + static_cast<size_t>(row) * N + col;
                        if (col < N) dst[0] = v0;
                        if (col + 1 < N) dst[1] = v1;
                    } else {
                        __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;
                        if (col + 1 < N && (N & 1) == 0)
                            *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                        else {
                            if (col < N) dst[0] = __float2bfloat16_rn(v0);
                            if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                        }
                    }
                }
            }
    }
}

// The K slices' partial sums added in slice order, then rounded once.
template <bool F32>
__device__ __forceinline__ void reduce_kernel(const float* __restrict__ part, void* __restrict__ out, long long total, int SK) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float acc = part[i];
    for (int s = 1; s < SK; ++s) acc = acc + part[s * total + i];
    if (F32) reinterpret_cast<float*>(out)[i] = acc;
    else reinterpret_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(acc);
}

// tile_slice with the weights as the mma's A (16 rows: the warp's columns of n8 tiles 0 and 1, from the same words) and
// the inputs as its B (8 columns: tokens 8 tt .. 8 tt + 7, the halves of the same ldmatrix), so TT = 1 (at most 8 rows)
// takes half the mmas. A lane ends with its columns wn 16 + gq (acc 0, 1) and wn 16 + 8 + gq (acc 2, 3) for tokens
// 8 tt + 2 t and + 1. The tensor core's dot product doesn't depend on which operand is which, and the steps run on the
// same values in the same order, so every output has tile_slice's bits.
// P (timing probes only, wrong outputs): 1 skips the mmas, 2 the input rows' loads, 3 both.
template <int GS, int BN, int WN, int STAGES, int TT, int G = 1, int P = 0>
__device__ __forceinline__ void tile_slice_swap(float (&acc)[TT][4], unsigned char* buf,
        const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const uint32_t* __restrict__ w,
        const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases, int M, int KG, int per,
        int g0, int n0, int npad, int ldx) {
    constexpr int BM = 16;
    using T = Tile<GS, BM, BN, 1, WN, STAGES>;
    static_assert(T::NT == 2 && T::MT == 1, "a warp: one m16 tile of 16 columns");
    const int tid = threadIdx.x % T::THREADS, lane = tid & 31, wn = tid >> 5, gq = lane >> 2, t = lane & 3;
    // G > 1: the block holds G slices of a tile, a group of T::THREADS each with its own stages, synced alone
    auto sync = [] {
        if constexpr (G == 1) __syncthreads();
        else asm volatile("bar.sync %0, %1;" :: "r"(1 + static_cast<int>(threadIdx.x) / T::THREADS), "n"(T::THREADS));
    };

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    // a group's stage is one 16-byte copy of inputs and one of weights a thread, a scale or bias piece for 16 threads
    // and an input sum for 16; each thread finds its sources once and steps them a group at a time (general copy loops
    // made the loop several times longer than its mmas, so a block streamed ~5 GB/s)
    static_assert(T::THREADS == BM * T::CHUNKS && T::W == T::THREADS * 16 && 2 * (T::S / 16) <= T::THREADS,
                  "one copy of each kind a thread");
    const int xr = tid / T::CHUNKS, xc = tid % T::CHUNKS;
    const bool xread = xr < M && !(P & 2);
    const int xoff = xr * T::ROW + swz<T::CHUNKS>(xr, xc) * 16;
    const __nv_bfloat16* xsrc = x + static_cast<size_t>(min(xr, M - 1)) * ldx + static_cast<size_t>(g0) * GS + xc * 8;
    const unsigned char* wsrc = reinterpret_cast<const unsigned char*>(w)
        + (static_cast<size_t>(n0 / 64) * KG + g0) * (64 * GS / 2) + tid * 16;
    const int sw = tid / (T::S / 16), so = tid % (T::S / 16);
    const bool sread = tid < 2 * (T::S / 16);
    const __nv_bfloat16* ssrc = (sw ? biases : scales) + static_cast<size_t>(g0) * npad + n0 + so * 8;
    const int soff = T::X + T::W + sw * T::S + so * 16;
    const bool xsread = tid < BM;
    const float* xssrc = xs + static_cast<size_t>(min(tid, M - 1)) * KG + g0;
    const int xsoff = T::X + T::W + 2 * T::S + tid * 4;
    auto load = [&](int s, int i) {                       // group g0 + i into stage s
        unsigned char* p = stage(s);
        cp16z(p + xoff, xsrc + i * GS, xread);
        cp16(p + T::X + tid * 16, wsrc + static_cast<size_t>(i) * (64 * GS / 2));
        if (sread) cp16(p + soff, ssrc + static_cast<size_t>(i) * npad);
        if (xsread) cp4(p + xsoff, xssrc + i);
    };

#pragma unroll
    for (int u = 0; u < TT; ++u) acc[u][0] = acc[u][1] = acc[u][2] = acc[u][3] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < per) load(s, s);
        commit();
    }
    const int c0 = wn * 16 + gq, c1 = c0 + 8;             // the lane's columns in the block
    for (int it = 0; it < per; ++it) {
        wait<STAGES - 2>();
        sync();
        const int next = it + STAGES - 1;
        if (next < per) load(next % STAGES, next);
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + T::X + T::W);
        const float* px = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S);
        uint32_t words[2][GS / 32];
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[((wn * 2 + j) * 32 + lane) * (GS / 32) + v];
        float d[TT][4];
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            uint32_t a[4];
            const int r = (lane & 7) + ((lane >> 3) & 1) * 8;
            ldmatrix4(a, p + r * T::ROW + swz<T::CHUNKS>(r, kt * 2 + (lane >> 4)) * 16);
            const int sh = (kt & 1) * 8;
            const uint32_t wa[4] = {pair(words[0][kt / 2], sh), pair(words[1][kt / 2], sh),
                                    pair(words[0][kt / 2], sh + 4), pair(words[1][kt / 2], sh + 4)};
#pragma unroll
            for (int u = 0; u < TT; ++u) {
                if constexpr (P & 1) {
                    if (kt == 0) d[u][0] = d[u][1] = d[u][2] = d[u][3] = 0.f;
                    d[u][0] += __uint_as_float(wa[0] ^ a[u]); // keeps the words and fragments live
                } else if (kt == 0) mma0(d[u], wa, a[u], a[u + 2]);
                else mma(d[u], wa, a[u], a[u + 2]);
            }
        }
        const float s0 = __bfloat162float(ps[c0]), s1 = __bfloat162float(ps[c1]);
        const float b0 = __bfloat162float(ps[BN + c0]), b1 = __bfloat162float(ps[BN + c1]);
#pragma unroll
        for (int u = 0; u < TT; ++u) {
            const float x0 = px[u * 8 + 2 * t], x1 = px[u * 8 + 2 * t + 1];
            acc[u][0] = __fmaf_rn(x0, b0, __fmaf_rn(d[u][0], s0, acc[u][0]));
            acc[u][1] = __fmaf_rn(x1, b0, __fmaf_rn(d[u][1], s0, acc[u][1]));
            acc[u][2] = __fmaf_rn(x0, b1, __fmaf_rn(d[u][2], s1, acc[u][2]));
            acc[u][3] = __fmaf_rn(x1, b1, __fmaf_rn(d[u][3], s1, acc[u][3]));
        }
    }
    wait<0>();
    sync();
}

// Several weights of the same input in one launch (decode, one row tile): segment i's blocks are its column tiles
// times its own K slices, the slices it would get alone. Slices park in part (SK, M, N) and the tile's last block to
// finish adds them in slice order and rounds once, as reduce_kernel does; sync counts a tile's finished slices and is
// zero before and after a launch.
constexpr int SEGS = 4;
struct Seg {
    const uint32_t* w;
    const __nv_bfloat16* scales;
    const __nv_bfloat16* biases;
    void* out;
    float* part;
    int N, npad, SK, blocks, sync;                        // blocks: tiles x SK; sync: its tiles' first counter
};
struct Segs {
    Seg s[SEGS];
    int count;
};

template <bool F32>
__device__ __forceinline__ void put(void* out, size_t i, float v) {
    if (F32) reinterpret_cast<float*>(out)[i] = v;
    else reinterpret_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(v);
}

// TT 0: tile_slice; 1 or 2: tile_slice_swap with that many 8-token tiles. G > 1 (swapped form, every segment G K
// slices): a block is a tile's G slices side by side, and slice 0 adds the others' partials from shared memory in slice
// order, the sum the slice buffer gives, without parking them in global memory or a block waiting on the others.
template <int GS, int BM, int BN, int WM, int WN, int STAGES, bool F32, int TT, int G = 1, int P = 0>
__device__ __forceinline__ void qmm_seg_kernel(
        const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const Segs segs, int* __restrict__ sync,
        int M, int K, int ldx) {
    using T = Tile<GS, BM, BN, WM, WN, STAGES>;
    extern __shared__ __align__(128) unsigned char buf[];
    __shared__ int last;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    int b = blockIdx.x;
    Seg sg = segs.s[0];
#pragma unroll
    for (int i = 1; i < SEGS; ++i)
        if (i < segs.count && b >= sg.blocks / G) {           // G > 1: a block a tile
            b -= sg.blocks / G;
            sg = segs.s[i];
        }
    const int KG = K / GS, per = KG / sg.SK, tiles = sg.blocks / sg.SK;
    const int slice = G > 1 ? static_cast<int>(threadIdx.x) / T::THREADS : b / tiles;
    const int tile = G > 1 ? b : b - slice * tiles, n0 = tile * BN;
    constexpr int E = TT ? TT * 4 : T::MT * T::NT * 4;  // the lane's outputs
    float v[E];
    int rows[E], cols[E];
    if constexpr (TT == 0) {
        float acc[T::MT][T::NT][4];
        tile_slice<GS, BM, BN, WM, WN, STAGES, false>(acc, buf, x, xs, sg.w, sg.scales, sg.biases, M, KG, per,
                                                      slice * per, 0, n0, sg.npad, ldx);
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int q = (i * T::NT + j) * 4 + e;
                    v[q] = acc[i][j][e];
                    rows[q] = wm * (BM / WM) + i * 16 + (lane >> 2) + (e >> 1) * 8;
                    cols[q] = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2 + (e & 1);
                }
    } else {
        static_assert(WM == 1 && BM == 16, "the swapped form: one 16-row tile, a warp's 16 columns");
        float acc[TT][4];
        tile_slice_swap<GS, BN, WN, STAGES, TT, G, P>(acc, buf + (G > 1 ? slice * STAGES * T::STAGE : 0), x, xs, sg.w,
                                                   sg.scales, sg.biases, M, KG, per, slice * per, n0, sg.npad, ldx);
#pragma unroll
        for (int u = 0; u < TT; ++u)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                v[u * 4 + e] = acc[u][e];
                rows[u * 4 + e] = u * 8 + 2 * (lane & 3) + (e & 1);
                cols[u * 4 + e] = n0 + wn * 16 + (lane >> 2) + (e >> 1) * 8;
            }
    }
    const int N = sg.N;
    const size_t MN = static_cast<size_t>(M) * N;
    if constexpr (G > 1) {
        float* red = reinterpret_cast<float*>(buf);       // slice s's lane values at [s - 1][q][thread]
        const int lt = threadIdx.x % T::THREADS;
        __syncthreads();                                  // every slice's stages done
        if (slice > 0)
#pragma unroll
            for (int q = 0; q < E; ++q) red[((slice - 1) * E + q) * T::THREADS + lt] = v[q];
        __syncthreads();
        if (slice > 0) return;
#pragma unroll
        for (int q = 0; q < E; ++q) {
            float a = v[q];
#pragma unroll
            for (int s2 = 1; s2 < G; ++s2) a = a + red[((s2 - 1) * E + q) * T::THREADS + lt];
            if (rows[q] < M && cols[q] < N) put<F32>(sg.out, static_cast<size_t>(rows[q]) * N + cols[q], a);
        }
        return;
    }
#pragma unroll
    for (int q = 0; q < E; ++q) {
        if (rows[q] >= M || cols[q] >= N) continue;
        const size_t at = static_cast<size_t>(rows[q]) * N + cols[q];
        if (sg.SK == 1) put<F32>(sg.out, at, v[q]);
        else sg.part[slice * MN + at] = v[q];
    }
    if (sg.SK == 1) return;
    __threadfence();                                      // this slice's partials, then its count
    __syncthreads();
    if (tid == 0) last = atomicAdd(sync + sg.sync + tile, 1) == sg.SK - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
#pragma unroll
    for (int q = 0; q < E; ++q) {
        if (rows[q] >= M || cols[q] >= N) continue;
        const size_t at = static_cast<size_t>(rows[q]) * N + cols[q];
        float p[8];                                       // every slice's load in flight, then added in order
#pragma unroll
        for (int s = 0; s < 8; ++s) p[s] = s < sg.SK ? __ldcg(sg.part + s * MN + at) : 0.f;
        float a = p[0];
#pragma unroll
        for (int s = 1; s < 8; ++s)
            if (s < sg.SK) a = a + p[s];
        put<F32>(sg.out, at, a);
    }
    if (tid == 0) sync[sg.sync + tile] = 0;
}
} // namespace

// Segmented decode launch (rows <= 16): the weights as the mma's A, TT 8-token tiles (1: <= 8 rows, 2: <= 16), G K
// slices side by side in a block (1: slices park in part and the tile's last block adds them), STAGES 3 or 4.
#define TF_SEG(ST, TT, G)                                                                                              \
    extern "C" __global__ void __launch_bounds__(128 * G) tf_qmm_seg_s##ST##_t##TT##_g##G(                         \
            const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const Segs segs, int* __restrict__ sync, \
            int M, int K, int ldx) {                                                                                   \
        qmm_seg_kernel<64, 16, 64, 1, 4, ST, false, TT, G>(x, xs, segs, sync, M, K, ldx);                              \
    }
TF_SEG(3, 1, 1) TF_SEG(3, 2, 1) TF_SEG(3, 1, 2) TF_SEG(3, 2, 2) TF_SEG(3, 1, 4) TF_SEG(3, 2, 4)
TF_SEG(4, 1, 1) TF_SEG(4, 2, 1) TF_SEG(4, 1, 2) TF_SEG(4, 2, 2) TF_SEG(4, 1, 4) TF_SEG(4, 2, 4)
#undef TF_SEG

// Timing probes of the 16-row, one-slice-a-block launch (TF_QWEN27_TUNE probe=P): wrong outputs.
#define TF_SEG_PROBE(ST, P)                                                                                            \
    extern "C" __global__ void __launch_bounds__(128) tf_qmm_seg_probe##P##_s##ST(                                   \
            const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const Segs segs, int* __restrict__ sync, \
            int M, int K, int ldx) {                                                                                   \
        qmm_seg_kernel<64, 16, 64, 1, 4, ST, false, 2, 1, P>(x, xs, segs, sync, M, K, ldx);                            \
    }
TF_SEG_PROBE(3, 1) TF_SEG_PROBE(3, 2) TF_SEG_PROBE(3, 3) TF_SEG_PROBE(4, 1) TF_SEG_PROBE(4, 2) TF_SEG_PROBE(4, 3)
#undef TF_SEG_PROBE

// Wider rounds: BM-row tiles, K slices into part (SK, M, N) for tf_qmm_reduce, or straight to bf16 with one slice.
#define TF_LANE(BM)                                                                                                    \
    extern "C" __global__ void __launch_bounds__(128) tf_qmm_lane_m##BM(                                             \
            const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const uint32_t* __restrict__ w,        \
            const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases,                      \
            void* __restrict__ out, float* __restrict__ part, int M, int N, int K, int SK, int npad, int ldx,         \
            int group) {                                                                                               \
        qmm_kernel<64, BM, 64, 1, 4, 4, false, false>(x, xs, w, scales, biases, out, part, M, N, K, SK, npad, ldx, group); \
    }
TF_LANE(16) TF_LANE(32) TF_LANE(64)
#undef TF_LANE

extern "C" __global__ void tf_qmm_reduce(const float* __restrict__ part, void* __restrict__ out, long long total, int SK) {
    reduce_kernel<false>(part, out, total, SK);
}

