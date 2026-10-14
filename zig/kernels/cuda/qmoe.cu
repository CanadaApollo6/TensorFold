// Decode form: acc = fma(xs, b, fma(p, s, acc)) per group in order, one pair an mma row, so no pair affects another.

#include <cuda_bf16.h>
#include <stdint.h>

#include "qmoe.cuh"

namespace qmoe {

__device__ __forceinline__ uint4 ld_w(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ uint2 ld_w2(const uint2* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];\n" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}

__device__ __forceinline__ uint4 ld_x(const __nv_bfloat16* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }

// a row's group sum: the lane's inputs in order, then the quad (every lane of the quad gets the same bits)
template <int XV>
__device__ __forceinline__ float group_sum(const uint4 (&v)[XV]) {
  float s = 0.f;
#pragma unroll
  for (int c = 0; c < XV; ++c)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t u = comp(v[c], i);
      s += lo_f(u);
      s += hi_f(u);
    }
  s += __shfl_xor_sync(0xffffffffu, s, 1);
  s += __shfl_xor_sync(0xffffffffu, s, 2);
  return s;
}

template <int GS, int M, int BITS>
struct Stage {
  uint32_t w[M][Geo<GS, BITS>::WORDS];
  uint4 sb[M][Geo<GS, BITS>::SBV];
  uint4 xa[Geo<GS>::XV];       // row gq of the tile: the lane's inputs of the group
  uint4 xb[Geo<GS>::XV];       // row gq + 8
};

template <int GS, int M, int BITS>
__device__ __forceinline__ void load_stage(Stage<GS, M, BITS>& st, const uint4* blk, int g, int lane, int t,
                                           const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
  using G = Geo<GS, BITS>;
  const uint4* b = blk + (size_t)g * (M * G::BLOCK);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    lane_words<GS, BITS>(st.w[m], b + m * G::BLOCK, lane, [](const uint4* q) { return ld_w(q); },
                         [](const uint2* q) { return ld_w2(q); });
#pragma unroll
    for (int c = 0; c < G::SBV; ++c) st.sb[m][c] = ld_w(b + m * G::BLOCK + G::WU + t * G::SBV + c);
  }
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
  const int k0 = g * GS;
#pragma unroll
  for (int c = 0; c < G::XV; ++c) {
    st.xa[c] = v0 ? ld_x(x0 + k0 + 8 * c) : zero;
    st.xb[c] = v1 ? ld_x(x1 + k0 + 8 * c) : zero;
  }
}

// one matrix's products of a group: p[j] = the levels times the inputs, from zero
template <int GS, int BITS>
__device__ __forceinline__ void products(float (&p)[NTW][4], const uint32_t (&w)[Geo<GS, BITS>::WORDS],
                                         const uint4 (&xa)[Geo<GS>::XV], const uint4 (&xb)[Geo<GS>::XV]) {
#pragma unroll
  for (int j = 0; j < NTW; ++j) p[j][0] = p[j][1] = p[j][2] = p[j][3] = 0.f;
#pragma unroll
  for (int ks = 0; ks < Geo<GS>::KS; ++ks) {
    // k-step ks: the lane's inputs 4 ks .. 4 ks + 3 (pairs at the mma's k positions 2t and 2t + 8)
    const uint32_t a0 = comp(xa[ks >> 1], 2 * (ks & 1));
    const uint32_t a2 = comp(xa[ks >> 1], 2 * (ks & 1) + 1);
    const uint32_t a1 = comp(xb[ks >> 1], 2 * (ks & 1));
    const uint32_t a3 = comp(xb[ks >> 1], 2 * (ks & 1) + 1);
    const int sh = (ks & 1) ? 8 : 0;
#pragma unroll
    for (int j = 0; j < NTW; ++j) {
      const int wi = j * (GS / 32) + (ks >> 1);
      const uint32_t word = nibword<BITS>(w, wi);
      mma(p[j], a0, a1, a2, a3, nib2(word, sh), nib2(word, sh + 4));
    }
  }
}

// tile j's step of the decode form: sb holds the quad's scales (uint4 0 .. NTW/4 - 1) then biases
__device__ __forceinline__ void accumulate(float (&a)[4], const float (&p)[4], const uint4* sb, int j, float xsa,
                                           float xsb, bool hi) {
  const uint32_t sp = comp(sb[j >> 2], j & 3);
  const uint32_t bp = comp(sb[(NTW + j) >> 2], (NTW + j) & 3);
  const float s0 = lo_f(sp), s1 = hi_f(sp), b0 = lo_f(bp), b1 = hi_f(bp);
  a[0] = fmaf(xsa, b0, fmaf(p[0], s0, a[0]));
  a[1] = fmaf(xsa, b1, fmaf(p[1], s1, a[1]));
  if (hi) {
    a[2] = fmaf(xsb, b0, fmaf(p[2], s0, a[2]));
    a[3] = fmaf(xsb, b1, fmaf(p[3], s1, a[3]));
  }
}

template <int GS, int M, int BITS>
__device__ __forceinline__ void compute_stage(float (&acc)[M][1][NTW][4], const Stage<GS, M, BITS>& st, bool hi) {
  using G = Geo<GS, BITS>;
  const float xsa = group_sum<G::XV>(st.xa);
  const float xsb = group_sum<G::XV>(st.xb);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    float p[NTW][4];
    products<GS, BITS>(p, st.w[m], st.xa, st.xb);
#pragma unroll
    for (int j = 0; j < NTW; ++j) accumulate(acc[m][0][j], p[j], st.sb[m], j, xsa, xsb, hi);
  }
}

template <int GS, int M, int D, int BITS>
__device__ __forceinline__ void k_loop(float (&acc)[M][1][NTW][4], const uint4* blk, int KG, int lane, int t,
                                       const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
  Stage<GS, M, BITS> st[D];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < KG) load_stage<GS, M, BITS>(st[d], blk, d, lane, t, x0, x1, v0, v1);
  for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int g = g0 + d;
      if (g < KG) {
        compute_stage<GS, M, BITS>(acc, st[d], v1);
        if (g + D < KG) load_stage<GS, M, BITS>(st[d], blk, g + D, lane, t, x0, x1, v0, v1);
      }
    }
  }
}

// Pair p reads X row p / slots when slots > 0 (X holds tokens), else X row p (X holds a row a pair).
// Experts are BITS wide, except expert e4 (the shared expert, -1: none), which is 4-bit in W4.
template <int GS, int M, int EPI, int D, int WARPS, int BITS>
__global__ void __launch_bounds__(WARPS * 32)
    expert_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W, int KG,
                  int NB, const int* __restrict__ items, const int* __restrict__ counts,
                  const int* __restrict__ members, void* __restrict__ out, int N, float limit,
                  const uint4* __restrict__ W4, int e4) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    const bool v0 = gq < cnt, v1 = gq + 8 < cnt;
    const int pr0 = v0 ? __ldg(members + first + gq) : 0;
    const int pr1 = v1 ? __ldg(members + first + gq + 8) : 0;
    const int r0 = slots ? pr0 / slots : pr0, r1 = slots ? pr1 / slots : pr1;
    const __nv_bfloat16* x0 = X + (size_t)r0 * x_stride + t * (GS / 4);
    const __nv_bfloat16* x1 = X + (size_t)r1 * x_stride + t * (GS / 4);
    float acc[M][1][NTW][4];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int j = 0; j < NTW; ++j) acc[m][0][j][0] = acc[m][0][j][1] = acc[m][0][j][2] = acc[m][0][j][3] = 0.f;
    if (BITS != 4 && e == e4) {                    // warp-uniform: a unit is one expert's tile
      const uint4* blk = W4 + (size_t)cb * KG * (M * Geo<GS, 4>::BLOCK);
      k_loop<GS, M, D, 4>(acc, blk, KG, lane, t, x0, x1, v0, v1);
    } else {
      const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * Geo<GS, BITS>::BLOCK);
      k_loop<GS, M, D, BITS>(acc, blk, KG, lane, t, x0, x1, v0, v1);
    }
    epilogue<EPI, M, 1>(acc, 0, out, N, cb * COLS + 2 * t, pr0, pr1, v0, v1, limit);
  }
}

// Swapped form, the decode kernel: the weights are the mma's A (16 rows: column gq of tiles j and j + 1) and a token's
// inputs its B (8 columns: tokens h8 + gq), so a unit of at most 8 pairs takes half the mmas; an item of more takes
// two passes. A lane ends with column gq of each tile for tokens h8 + 2t and h8 + 2t + 1. The tensor core's dot
// product doesn't depend on which operand is which, and the steps above run on the same values in the same order,
// so every output has the decode form's bits (test_experts_split.py).
// S warps can share a unit: warp w takes groups w, w + S, ..., writes its products, group sums and scales to shared
// memory, and warp j folds tile j over each round's groups in order.

template <int GS, int BITS>
__device__ __forceinline__ void swapped_products(float (&q)[NTW][2], const uint32_t (&w)[Geo<GS, BITS>::WORDS],
                                                 const uint4 (&xa)[Geo<GS>::XV]) {
  float c[NTW / 2][4];
#pragma unroll
  for (int P = 0; P < NTW / 2; ++P) c[P][0] = c[P][1] = c[P][2] = c[P][3] = 0.f;
#pragma unroll
  for (int ks = 0; ks < Geo<GS>::KS; ++ks) {
    const uint32_t b0 = comp(xa[ks >> 1], 2 * (ks & 1));
    const uint32_t b1 = comp(xa[ks >> 1], 2 * (ks & 1) + 1);
    const int sh = (ks & 1) ? 8 : 0;
#pragma unroll
    for (int P = 0; P < NTW / 2; ++P) {
      const uint32_t w0 = nibword<BITS>(w, 2 * P * (GS / 32) + (ks >> 1));
      const uint32_t w1 = nibword<BITS>(w, (2 * P + 1) * (GS / 32) + (ks >> 1));
      mma(c[P], nib2(w0, sh), nib2(w1, sh), nib2(w0, sh + 4), nib2(w1, sh + 4), b0, b1);
    }
  }
#pragma unroll
  for (int P = 0; P < NTW / 2; ++P) {
    q[2 * P][0] = c[P][0];
    q[2 * P][1] = c[P][1];
    q[2 * P + 1][0] = c[P][2];
    q[2 * P + 1][1] = c[P][3];
  }
}

// tile j's step for column gq (odd: gq & 1) and two tokens; sb holds scales of columns 2 (gq >> 1) and 2 (gq >> 1) + 1
__device__ __forceinline__ void accumulate_t(float (&a)[2], const float (&p)[2], const uint4* sb, int j, bool odd,
                                             const float (&xt)[2]) {
  const uint32_t sp = comp(sb[j >> 2], j & 3);
  const uint32_t bp = comp(sb[(NTW + j) >> 2], (NTW + j) & 3);
  const float sc = odd ? hi_f(sp) : lo_f(sp), bi = odd ? hi_f(bp) : lo_f(bp);
  a[0] = fmaf(xt[0], bi, fmaf(p[0], sc, a[0]));
  a[1] = fmaf(xt[1], bi, fmaf(p[1], sc, a[1]));
}

template <int GS, int M>
struct Part {
  float p[M][NTW][2][32];          // [.., lane]
  float xs[32];                    // token gq's group sum, at lane 4 gq
  uint4 sb[M][4][NTW / 2];         // [.., gq >> 1, ..]
};

template <int GS, int M, int D, int S, int BITS>
__device__ __forceinline__ void swap_loop(float (&acc)[M][NTW][2], Part<GS, M>* part, const uint4* blk, int KG,
                                          int warp, int lane, const __nv_bfloat16* x0, bool v0) {
  using G = Geo<GS, BITS>;
  const int gq = lane >> 2, t = lane & 3, sq = gq >> 1;
  const bool odd = gq & 1;
  Stage<GS, M, BITS> st[D];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (warp + d * S < KG) load_stage<GS, M, BITS>(st[d], blk, warp + d * S, lane, sq, x0, x0, v0, false);
  const int rounds = (KG + S - 1) / S;
  for (int r0 = 0; r0 < rounds; r0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int r = r0 + d;
      if (r >= rounds) break;                                   // block-uniform
      const int g = r * S + warp;
      if constexpr (S == 1) {
        const float xs = group_sum<G::XV>(st[d].xa);
        const float xt[2] = {__shfl_sync(0xffffffffu, xs, 8 * t), __shfl_sync(0xffffffffu, xs, 8 * t + 4)};
#pragma unroll
        for (int m = 0; m < M; ++m) {
          float q[NTW][2];
          swapped_products<GS, BITS>(q, st[d].w[m], st[d].xa);
#pragma unroll
          for (int j = 0; j < NTW; ++j) accumulate_t(acc[m][j], q[j], st[d].sb[m], j, odd, xt);
        }
        if (g + D < KG) load_stage<GS, M, BITS>(st[d], blk, g + D, lane, sq, x0, x0, v0, false);
      } else {
        Part<GS, M>* buf = part + (r & 1) * S;
        if (g < KG) {
          Part<GS, M>& me = buf[warp];
          me.xs[lane] = group_sum<G::XV>(st[d].xa);
#pragma unroll
          for (int m = 0; m < M; ++m) {
            float q[NTW][2];
            swapped_products<GS, BITS>(q, st[d].w[m], st[d].xa);
#pragma unroll
            for (int j = 0; j < NTW; ++j) {
              me.p[m][j][0][lane] = q[j][0];
              me.p[m][j][1][lane] = q[j][1];
            }
            if (t == 0 && !odd)
#pragma unroll
              for (int c = 0; c < G::SBV; ++c) me.sb[m][sq][c] = st[d].sb[m][c];
          }
          if (g + D * S < KG) load_stage<GS, M, BITS>(st[d], blk, g + D * S, lane, sq, x0, x0, v0, false);
        }
        __syncthreads();
        const int n = min(S, KG - r * S);
        for (int s = 0; s < n; ++s) {
          const Part<GS, M>& b = buf[s];
          const float xt[2] = {b.xs[8 * t], b.xs[8 * t + 4]};
#pragma unroll
          for (int m = 0; m < M; ++m)
#pragma unroll
            for (int j = 0; j < NTW; ++j) {
              if (j % S != warp) continue;
              const float q[2] = {b.p[m][j][0][lane], b.p[m][j][1][lane]};
              accumulate_t(acc[m][j], q, b.sb[m][sq], j, odd, xt);
            }
        }
      }
    }
  }
}

// tokens tok0 and tok0 + 1 (those below cnt) of column col of each tile j in js
template <int EPI, int M>
__device__ __forceinline__ void epilogue_t(const float (&acc)[M][NTW][2], void* out, int N, int col,
                                           const int* members, int first, int cnt, int tok0, float limit,
                                           unsigned js) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (tok0 + h >= cnt) continue;
    const size_t row = (size_t)members[first + tok0 + h] * N;          // members: global or shared
#pragma unroll
    for (int j = 0; j < NTW; ++j) {
      if (!((js >> j) & 1u)) continue;
      const float a = acc[0][j][h];
      if constexpr (EPI == 0) {
        reinterpret_cast<float*>(out)[row + col + 8 * j] = a;
      } else {
        const float o = EPI == 1 ? relu2(a) : swiglu(a, acc[M - 1][j][h], limit);
        reinterpret_cast<__nv_bfloat16*>(out)[row + col + 8 * j] = __float2bfloat16_rn(o);
      }
    }
  }
}

// EPI 4: down (fp32 out, as 0), then moe.combine: the last unit to finish a (row, tile) sums the row's slots in slot
// order, fp32 fma(w, y, acc) from 0, rounded once to bf16 (Triton's steps, so its bits). sync [rows x tiles] is zero
// before a launch and after it.
struct Combine {
  const float* w;                  // [rows, slots]
  __nv_bfloat16* out;              // [rows, N]
  int* sync;
  int slots;
};

__device__ __forceinline__ void combine_tiles(const Combine& c, const float* y, const int* members, int first,
                                              int cnt, int cb, int NB, int N, int lane) {
  for (int h = 0; h < cnt; ++h) {
    const int r = members[first + h] / c.slots;
    int last = 0;
    if (lane == 0) last = atomicAdd(c.sync + r * NB + cb, 1) == c.slots - 1;
    if (__shfl_sync(0xffffffffu, last, 0)) {
      __threadfence();
      const int col = cb * COLS + lane;
      float a = 0.f;
      for (int k = 0; k < c.slots; ++k)
        a = fmaf(__ldg(c.w + r * c.slots + k), __ldcg(y + ((size_t)r * c.slots + k) * N + col), a);
      c.out[(size_t)r * N + col] = __float2bfloat16_rn(a);
      if (lane == 0) c.sync[r * NB + cb] = 0;
    }
  }
}

// S == 1: four warps a block, a unit each; else S warps a unit
template <int GS, int M, int EPI, int D, int S, int BITS>
__global__ void __launch_bounds__((S == 1 ? 4 : S) * 32)
    swap_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W, int KG,
                int NB, const int* __restrict__ items, const int* __restrict__ counts,
                const int* __restrict__ members, void* __restrict__ out, int N, float limit,
                const uint4* __restrict__ W4, int e4, Combine comb) {
  constexpr int U = S == 1 ? 4 : 1;                // units a block
  extern __shared__ uint4 smem[];
  Part<GS, M>* part = reinterpret_cast<Part<GS, M>*>(smem);    // [2 rounds][S]
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int warp = S == 1 ? 0 : threadIdx.x >> 5;
  unsigned js = 0;
#pragma unroll
  for (int j = 0; j < NTW; ++j) js |= (j % S == warp) << j;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x * U + (S == 1 ? threadIdx.x >> 5 : 0); unit < units; unit += gridDim.x * U) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    for (int h8 = 0; h8 < cnt; h8 += 8) {
      const bool v = h8 + gq < cnt;
      const int pr = v ? __ldg(members + first + h8 + gq) : 0;
      const int r = slots ? pr / slots : pr;
      const __nv_bfloat16* x0 = X + (size_t)r * x_stride + t * (GS / 4);
      float acc[M][NTW][2];
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int j = 0; j < NTW; ++j) acc[m][j][0] = acc[m][j][1] = 0.f;
      if (BITS != 4 && e == e4) {                  // uniform: a unit is one expert's tile
        const uint4* blk = W4 + (size_t)cb * KG * (M * Geo<GS, 4>::BLOCK);
        swap_loop<GS, M, D, S, 4>(acc, part, blk, KG, warp, lane, x0, v);
      } else {
        const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * Geo<GS, BITS>::BLOCK);
        swap_loop<GS, M, D, S, BITS>(acc, part, blk, KG, warp, lane, x0, v);
      }
      if (js) epilogue_t<EPI == 4 ? 0 : EPI, M>(acc, out, N, cb * COLS + gq, members, first, cnt, h8 + 2 * t, limit,
                                                js);
      if constexpr (S > 1) __syncthreads();        // the next pass's first rounds reuse the parts
    }
    if constexpr (EPI == 4) {
      __threadfence();                             // this unit's y, then the tiles' slot counts
      if constexpr (S == 1) {
        __syncwarp();
        combine_tiles(comb, reinterpret_cast<const float*>(out), members, first, cnt, cb, NB, N, lane);
      } else {
        __syncthreads();
        if (warp == 0) combine_tiles(comb, reinterpret_cast<const float*>(out), members, first, cnt, cb, NB, N, lane);
      }
    }
  }
}

// Split-K decode (not swap_kernel's bits): S warps a unit, warp w takes groups w, w + S, ... with the S == 1 steps in
// registers, and warp 0 adds the other warps' sums in warp order once, at the end (one barrier a pass, not a round).
template <int GS, int M, int D, int S, int BITS>
__device__ __forceinline__ void splitk_loop(float (&acc)[M][NTW][2], const uint4* blk, int KG, int warp, int lane,
                                            const __nv_bfloat16* x0, bool v0) {
  using G = Geo<GS, BITS>;
  const int gq = lane >> 2, t = lane & 3, sq = gq >> 1;
  const bool odd = gq & 1;
  Stage<GS, M, BITS> st[D];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (warp + d * S < KG) load_stage<GS, M, BITS>(st[d], blk, warp + d * S, lane, sq, x0, x0, v0, false);
  for (int g0 = warp; g0 < KG; g0 += D * S) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int g = g0 + d * S;
      if (g >= KG) break;                                       // warp-uniform
      const float xs = group_sum<G::XV>(st[d].xa);
      const float xt[2] = {__shfl_sync(0xffffffffu, xs, 8 * t), __shfl_sync(0xffffffffu, xs, 8 * t + 4)};
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float q[NTW][2];
        swapped_products<GS, BITS>(q, st[d].w[m], st[d].xa);
#pragma unroll
        for (int j = 0; j < NTW; ++j) accumulate_t(acc[m][j], q[j], st[d].sb[m], j, odd, xt);
      }
      if (g + D * S < KG) load_stage<GS, M, BITS>(st[d], blk, g + D * S, lane, sq, x0, x0, v0, false);
    }
  }
}

template <int GS, int M, int EPI, int D, int S, int BITS>
__device__ __forceinline__ void splitk_body(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                            const uint4* __restrict__ W, int KG, int NB, const int* __restrict__ items,
                                            const int* __restrict__ counts, const int* __restrict__ members,
                                            void* __restrict__ out, int N, float limit, Combine comb) {
  __shared__ float red[S - 1][M][NTW][2][32];
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3, warp = threadIdx.x >> 5;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    for (int h8 = 0; h8 < cnt; h8 += 8) {
      const bool v = h8 + gq < cnt;
      const int pr = v ? __ldg(members + first + h8 + gq) : 0;
      const int r = slots ? pr / slots : pr;
      const __nv_bfloat16* x0 = X + (size_t)r * x_stride + t * (GS / 4);
      float acc[M][NTW][2];
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int j = 0; j < NTW; ++j) acc[m][j][0] = acc[m][j][1] = 0.f;
      const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * Geo<GS, BITS>::BLOCK);
      splitk_loop<GS, M, D, S, BITS>(acc, blk, KG, warp, lane, x0, v);
      if (warp > 0)
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
          for (int j = 0; j < NTW; ++j) red[warp - 1][m][j][0][lane] = acc[m][j][0], red[warp - 1][m][j][1][lane] = acc[m][j][1];
      __syncthreads();
      if (warp == 0) {
#pragma unroll
        for (int w = 0; w < S - 1; ++w)
#pragma unroll
          for (int m = 0; m < M; ++m)
#pragma unroll
            for (int j = 0; j < NTW; ++j) {
              acc[m][j][0] = __fadd_rn(acc[m][j][0], red[w][m][j][0][lane]);
              acc[m][j][1] = __fadd_rn(acc[m][j][1], red[w][m][j][1][lane]);
            }
        epilogue_t<EPI == 4 ? 0 : EPI, M>(acc, out, N, cb * COLS + gq, members, first, cnt, h8 + 2 * t, limit,
                                          (1u << NTW) - 1u);
      }
      __syncthreads();                             // red is free for the next pass
    }
    if constexpr (EPI == 4) {
      if (warp == 0) {
        __threadfence();                           // this unit's y, then the tiles' slot counts
        combine_tiles(comb, reinterpret_cast<const float*>(out), members, first, cnt, cb, NB, N, lane);
      }
    }
  }
}

}  // namespace qmoe

// The forms the qwen27 MoE engine launches (qwen27/cuda/moe.zig): groups of 64, gate/up SwiGLU, down fp32 or combined.
#define QMOE_SWAP(M, EPI, S, B)                                                                     \
  template __global__ void qmoe::swap_kernel<64, M, EPI, 2, S, B>(                                          \
      const __nv_bfloat16*, int, int, const uint4*, int, int, const int*, const int*, const int*, void*, int, \
      float, const uint4*, int, qmoe::Combine);
#define QMOE_SWAPS(M, EPI, B) \
  QMOE_SWAP(M, EPI, 1, B) QMOE_SWAP(M, EPI, 2, B) QMOE_SWAP(M, EPI, 4, B) QMOE_SWAP(M, EPI, 8, B)
QMOE_SWAPS(2, 2, 3)
QMOE_SWAPS(2, 2, 4)
QMOE_SWAPS(1, 0, 3)
QMOE_SWAPS(1, 0, 4)
QMOE_SWAPS(1, 4, 3)
QMOE_SWAPS(1, 4, 4)

// Split-K decode forms by name (lfm2/moe.zig): gate/up SwiGLU (gu), down fp32 (down), down combined (dc); S warps a unit.
#define QMOE_SPLITK(NAME, M, EPI, S, B)                                                                            \
  extern "C" __global__ void __launch_bounds__(S * 32) tf_qmoe_splitk_##NAME##_b##B##_s##S(                     \
      const __nv_bfloat16* X, int xs, int slots, const uint4* W, int KG, int NB, const int* items, const int* counts, \
      const int* members, void* out, int N, float limit, qmoe::Combine comb) {                                  \
    qmoe::splitk_body<64, M, EPI, 2, S, B>(X, xs, slots, W, KG, NB, items, counts, members, out, N, limit, comb); \
  }
#define QMOE_SPLITKS(NAME, M, EPI, B) QMOE_SPLITK(NAME, M, EPI, 2, B) QMOE_SPLITK(NAME, M, EPI, 4, B) QMOE_SPLITK(NAME, M, EPI, 8, B)
QMOE_SPLITKS(gu, 2, 2, 4)
QMOE_SPLITKS(down, 1, 0, 4)
QMOE_SPLITKS(dc, 1, 4, 4)
