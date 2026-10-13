// The grouped experts' plans: pairs grouped by expert, and decode routing (top-k, then the plan) in one launch.

#include <cuda_bf16.h>
#include <stdint.h>

#include "qmoe.cuh"

namespace qmoe {

constexpr int PLAN_THREADS = 1024;
constexpr int EMAX = 1024;
constexpr int PSMALL = 1024;       // pairs the one-block plan takes; wider calls rank in blocks of PLAN_THREADS

// Whole block: thread e gives expert e's pair count, gets its first member; items of <= T pairs land in expert order.
__device__ int place_items(int c, int E, int T, int* __restrict__ items, int* __restrict__ counts) {
  __shared__ int wsum[3][32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int tiles = (c + T - 1) / T;
  int a = c, b = tiles, u = c > 0;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int ua = __shfl_up_sync(0xffffffffu, a, o), ub = __shfl_up_sync(0xffffffffu, b, o);
    const int uu = __shfl_up_sync(0xffffffffu, u, o);
    if (lane >= o) {
      a += ua;
      b += ub;
      u += uu;
    }
  }
  if (lane == 31) {
    wsum[0][warp] = a;
    wsum[1][warp] = b;
    wsum[2][warp] = u;
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int s = 0; s < 3; ++s) {
      int v = wsum[s][lane];
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int w = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += w;
      }
      wsum[s][lane] = v;
    }
  }
  __syncthreads();
  const int off = (warp ? wsum[0][warp - 1] : 0) + a - c;
  const int ioff = (warp ? wsum[1][warp - 1] : 0) + b - tiles;
  if (tid == 0) {
    counts[0] = wsum[1][31];
    counts[1] = wsum[2][31];
  }
  if (tid < E)
    for (int j = 0; j < tiles; ++j) {
      int* it = items + 3 * (ioff + j);
      it[0] = tid;
      it[1] = off + T * j;
      it[2] = min(T, c - T * j);
    }
  return off;
}

// One block: pairs p = row * slots + slot grouped by expert (pair order within an expert).
__global__ void __launch_bounds__(PLAN_THREADS)
    plan_kernel(const int* __restrict__ picks, int P, int E, int T, int* __restrict__ members,
                int* __restrict__ items, int* __restrict__ counts) {
  __shared__ int pk[PSMALL];
  __shared__ int cnt[EMAX];
  __shared__ int off[EMAX];
  const int tid = threadIdx.x;
  for (int e = tid; e < EMAX; e += PLAN_THREADS) cnt[e] = 0;
  __syncthreads();
  for (int p = tid; p < P; p += PLAN_THREADS) {
    const int e = picks[p];
    pk[p] = e;
    atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  const int o = place_items(tid < E ? cnt[tid] : 0, E, T, items, counts);
  if (tid < E) off[tid] = o;
  __syncthreads();
  for (int p = tid; p < P; p += PLAN_THREADS) {
    const int e = pk[p];
    int rank = 0;
    for (int q = 0; q < p; ++q) rank += pk[q] == e;
    members[off[e] + rank] = p;
  }
}

// Wide calls, pass 1: block b ranks its pairs within each expert in pair order and writes its counts to hist[b][e].
__global__ void __launch_bounds__(PLAN_THREADS)
    plan_rank(const int* __restrict__ picks, int P, int E, int* __restrict__ rank, int* __restrict__ hist) {
  __shared__ int cnt[EMAX];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int e = tid; e < E; e += PLAN_THREADS) cnt[e] = 0;
  __syncthreads();
  const int p = blockIdx.x * PLAN_THREADS + tid;
  const bool ok = p < P;
  const int e = ok ? picks[p] : -1 - lane;
  const unsigned same = __match_any_sync(0xffffffffu, e);
  const int below = __popc(same & ((1u << lane) - 1u));
  for (int w = 0; w < PLAN_THREADS / 32; ++w) {
    if (warp == w) {
      const int base = ok ? cnt[e] : 0;
      __syncwarp();
      if (ok && below == 0) cnt[e] = base + __popc(same);
      if (ok) rank[p] = base + below;
    }
    __syncthreads();
  }
  for (int x = tid; x < E; x += PLAN_THREADS) hist[(size_t)blockIdx.x * E + x] = cnt[x];
}

// Pass 2, one block: each block's first member for each expert in place of its count, and the items.
__global__ void __launch_bounds__(PLAN_THREADS)
    plan_offsets(int nblk, int E, int T, int* __restrict__ hist, int* __restrict__ items, int* __restrict__ counts) {
  const int tid = threadIdx.x;
  int c = 0;
  if (tid < E)
    for (int b = 0; b < nblk; ++b) {
      int* h = hist + (size_t)b * E + tid;
      const int v = *h;
      *h = c;
      c += v;
    }
  const int off = place_items(c, E, T, items, counts);
  if (tid < E)
    for (int b = 0; b < nblk; ++b) hist[(size_t)b * E + tid] += off;
}

// Pass 3: every pair to its place.
__global__ void plan_scatter(const int* __restrict__ picks, int P, int E, const int* __restrict__ rank,
                             const int* __restrict__ hist, int* __restrict__ members) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < P) members[hist[(size_t)(p / PLAN_THREADS) * E + picks[p]] + rank[p]] = p;
}

// Decode routing in one launch: moe._topk_rows (the same instructions, so the same bits) then plan_kernel's plan.
// A warp a row: its top_k experts by fp32 logit (largest first, the lower id among equal logits), weights
// exp(l_k - l_0) / sum rounded to bf16, then the shared expert (id NE) with bf16(sigmoid(bf16(logit NE)))).
constexpr float LOG2E = 1.44269502162933349609375f;             // 0x3FB8AA3B, as Triton's exp

__device__ __forceinline__ float ex2_approx(float x) {
  float r;
  asm("ex2.approx.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

constexpr int SEL_THREADS = 256;                                // and at most as many pairs: one a thread

// float -> key whose unsigned order is the float order (-0 counts as +0)
__device__ __forceinline__ unsigned order_key(float f) {
  const unsigned u = __float_as_uint(__fadd_rn(f, 0.f));
  return (u & 0x80000000u) ? ~u : u | 0x80000000u;
}

__device__ __forceinline__ float key_value(unsigned k) {
  return __uint_as_float((k & 0x80000000u) ? k ^ 0x80000000u : ~k);
}

template <int PL>                                                // logits a lane: 32 PL >= NE
__global__ void __launch_bounds__(SEL_THREADS)
    select_kernel(const float* __restrict__ logits, int R, int NL, int NE, int K, int T, int* __restrict__ picks,
                  float* __restrict__ wts, int* __restrict__ members, int* __restrict__ items,
                  int* __restrict__ counts) {
  __shared__ int pk[SEL_THREADS];
  __shared__ int hist[EMAX + 1];                                 // pairs an expert
  __shared__ int part[SEL_THREADS / 32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, S = K + 1, P = R * S;
  for (int e = tid; e <= NE; e += SEL_THREADS) hist[e] = 0;
  for (int r = warp; r < R; r += SEL_THREADS / 32) {
    const float* row = logits + (size_t)r * NL;
    unsigned v[PL];
#pragma unroll
    for (int c = 0; c < PL; ++c) {
      const int i = lane + 32 * c;
      v[c] = i < NE ? order_key(row[i]) : 0u;
    }
    float top = 0.f, total = 0.f, ex_mine = 0.f;
    int idx_mine = 0;
    for (int k = 0; k < K; ++k) {
      unsigned bk = 0u, bi = 0xffffffffu;                        // the lane's best, lowest index on ties
#pragma unroll
      for (int c = 0; c < PL; ++c)
        if (v[c] > bk) {
          bk = v[c];
          bi = lane + 32 * c;
        }
      const unsigned m = __reduce_max_sync(0xffffffffu, bk);
      bi = __reduce_min_sync(0xffffffffu, bk == m ? bi : 0xffffffffu);
      const float bv = key_value(m);
      if (k == 0) top = bv;
      const float ex = ex2_approx(__fmul_rn(__fsub_rn(bv, top), LOG2E));
      total = __fadd_rn(total, ex);
      if (lane == k) {
        ex_mine = ex;
        idx_mine = bi;
      }
#pragma unroll
      for (int c = 0; c < PL; ++c)
        if (lane + 32 * c == (int)bi) v[c] = 0u;
    }
    if (lane < K) {
      picks[r * S + lane] = idx_mine;
      wts[r * S + lane] = bf(div_full(ex_mine, total));
      pk[r * S + lane] = idx_mine;
    } else if (lane == K) {
      const float sg = bf(row[NE]);
      picks[r * S + K] = NE;
      wts[r * S + K] = bf(div_full(1.f, __fadd_rn(ex2_approx(__fmul_rn(sg, -LOG2E)), 1.f)));
      pk[r * S + K] = NE;
    }
  }
  __syncthreads();
  // plan_kernel's plan: pair p's place is the pairs of lower experts, then the lower pairs of its own; items of <= T
  // pairs in expert order
  int e = 0, before = 0, below = 0;
  if (tid < P) {
    e = pk[tid];
    int c = 0;
    for (int q = 0; q < P; ++q) {
      const int f = pk[q];
      below += f < e;
      before += (f == e) & (q < tid);
      c += f == e;
    }
    if (before == 0) hist[e] = c;
  }
  __syncthreads();
  if (tid < P) members[below + before] = tid;
  // one scan over experts, a run of them a thread, of (items, pairs, 1 if used) packed 10 bits each (<= 256 pairs)
  const int per = (NE + SEL_THREADS) / SEL_THREADS, e0 = tid * per;
  int sum = 0;
  for (int f = e0; f < e0 + per && f <= NE; ++f) {
    const int c = hist[f];
    if (c) sum += (((c + T - 1) / T) << 20) | (c << 10) | 1;
  }
  int inc = sum;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, inc, o);
    if (lane >= o) inc += y;
  }
  if (lane == 31) part[warp] = inc;
  __syncthreads();
  int at = inc - sum;
  for (int w = 0; w < warp; ++w) at += part[w];
  for (int f = e0; f < e0 + per && f <= NE; ++f) {
    const int c = hist[f];
    if (!c) continue;
    const int it = at >> 20, pairs = (at >> 10) & 1023, n = (c + T - 1) / T;
    for (int j = 0; j < n; ++j) {
      int* x = items + 3 * (it + j);
      x[0] = f;
      x[1] = pairs + T * j;
      x[2] = min(T, c - T * j);
    }
    at += (n << 20) | (c << 10) | 1;
  }
  if (tid == SEL_THREADS - 1) {                                  // the last thread's running total is everything
    counts[0] = at >> 20;
    counts[1] = at & 1023;
  }
}

}  // namespace

template __global__ void qmoe::select_kernel<8>(const float*, int, int, int, int, int, int*, float*, int*, int*, int*);
