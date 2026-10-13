// The qwen27 MoE engine's own glue: router logits, prompt routing, the slots' weighted sum, and the load-time repacks.

#include <cuda_bf16.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

constexpr float LOG2E = 1.44269502162933349609375f;  // 0x3FB8AA3B, as Triton's exp

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

// float -> key whose unsigned order is the float order (-0 counts as +0)
__device__ __forceinline__ unsigned order_key(float f) {
  const unsigned u = __float_as_uint(__fadd_rn(f, 0.f));
  return (u & 0x80000000u) ? ~u : u | 0x80000000u;
}

__device__ __forceinline__ float key_value(unsigned k) {
  return __uint_as_float((k & 0x80000000u) ? k ^ 0x80000000u : ~k);
}

__device__ __forceinline__ float wide(float v) { return v; }
__device__ __forceinline__ float wide(__nv_bfloat16 v) { return __bfloat162float(v); }

}  // namespace

// out[r, e] = fp32 x[r] . w[e]: a warp an expert, 16 rows a block, fixed lane order and shuffle tree (rows independent)
extern "C" __global__ void __launch_bounds__(256)
    tf_qmoe_router(const __nv_bfloat16* __restrict__ x, int x_stride, const __nv_bfloat16* __restrict__ w, int R,
                   int NE, int D, float* __restrict__ out) {
  const int lane = threadIdx.x & 31, e = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (e >= NE) return;
  const uint4* wr = reinterpret_cast<const uint4*>(w + (size_t)e * D);
  const int r1 = min(R, (int)blockIdx.y * 16 + 16);
  for (int r = blockIdx.y * 16; r < r1; ++r) {
    const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)r * x_stride);
    float acc = 0.f;
    for (int c = lane; c < D / 8; c += 32) {
      const uint4 a = __ldg(xr + c), b = __ldg(wr + c);
      const uint32_t av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        acc = fmaf(__uint_as_float(av[j] << 16), __uint_as_float(bv[j] << 16), acc);
        acc = fmaf(__uint_as_float(av[j] & 0xFFFF0000u), __uint_as_float(bv[j] & 0xFFFF0000u), acc);
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
    if (lane == 0) out[(size_t)r * NE + e] = acc;
  }
}

// Prompt routing, a warp a row: select_kernel's top-k and weights (the same instructions), without the plan.
extern "C" __global__ void __launch_bounds__(256)
    tf_qmoe_topk_rows(const float* __restrict__ logits, int R, int NL, int NE, int K, int* __restrict__ picks,
                      float* __restrict__ wts) {
  constexpr int PL = 8;  // logits a lane: 256 experts at most
  const int lane = threadIdx.x & 31, r = blockIdx.x * 8 + (threadIdx.x >> 5), S = K + 1;
  if (r >= R) return;
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
    unsigned bk = 0u, bi = 0xffffffffu;
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
  } else if (lane == K) {
    const float sg = bf(row[NE]);
    picks[r * S + K] = NE;
    wts[r * S + K] = bf(div_full(1.f, __fadd_rn(ex2_approx(__fmul_rn(sg, -LOG2E)), 1.f)));
  }
}

// out[r] = bf16(sum over slots in order of fma(w, y, acc)): y fp32 (decode) or bf16 (prompt), [R, S, N]
template <class T>
__device__ __forceinline__ void combine_rows(const T* __restrict__ y, const float* __restrict__ w, int S, int N,
                                             __nv_bfloat16* __restrict__ out) {
  const int r = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < S; ++k) {
    acc = fmaf(__ldg(w + r * S + k), wide(y[((size_t)r * S + k) * N + col]), acc);
  }
  out[(size_t)r * N + col] = __float2bfloat16_rn(acc);
}

extern "C" __global__ void tf_qmoe_combine_f32(const float* __restrict__ y, const float* __restrict__ w, int S, int N,
                                               __nv_bfloat16* __restrict__ out) {
  combine_rows(y, w, S, N, out);
}

extern "C" __global__ void tf_qmoe_combine_bf16(const __nv_bfloat16* __restrict__ y, const float* __restrict__ w, int S,
                                                int N, __nv_bfloat16* __restrict__ out) {
  combine_rows(y, w, S, N, out);
}

// MLX 3-bit words (an LSB-first bitstream, 3 bytes a run of 8) -> MLX 4-bit words: a thread an output word.
extern "C" __global__ void tf_qmoe_widen3(const uint8_t* __restrict__ in, uint32_t* __restrict__ out, long words) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < words; i += (long)gridDim.x * blockDim.x) {
    const uint32_t v = in[3 * i] | (in[3 * i + 1] << 8) | (in[3 * i + 2] << 16);
    uint32_t o = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) o |= ((v >> (3 * j)) & 7u) << (4 * j);
    out[i] = o;
  }
}

// MLX 8-bit rows with their scales and biases (groups of 64) -> bf16 rows, bf16(fp32(s) * q + fp32(b)).
extern "C" __global__ void tf_qmoe_dequant8(const uint32_t* __restrict__ words, const __nv_bfloat16* __restrict__ s,
                                            const __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out,
                                            long values) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < values; i += (long)gridDim.x * blockDim.x) {
    const float q = (float)((words[i / 4] >> (8 * (i % 4))) & 0xFFu);
    const float v = __fadd_rn(__fmul_rn(__bfloat162float(s[i / 64]), q), __bfloat162float(b[i / 64]));
    out[i] = __float2bfloat16_rn(v);
  }
}

// 4-bit blocks [n, 288] -> _planes' 3-bit [n, 224]: 2-bit plane, 1-bit plane, scales; at i * stride + offset
extern "C" __global__ void tf_qmoe_planes3(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, long blocks,
                                           int stride, int offset) {
  for (long t = blockIdx.x * (long)blockDim.x + threadIdx.x; t < blocks * 32; t += (long)gridDim.x * blockDim.x) {
    const long i = t / 32;
    const int lane = t % 32;
    const uint32_t* b = in + i * 288;
    uint32_t* o = out + i * stride + offset;
    uint32_t w[8];
#pragma unroll
    for (int c = 0; c < 2; ++c)
#pragma unroll
      for (int j = 0; j < 4; ++j) w[c * 4 + j] = b[(c * 32 + lane) * 4 + j];
#pragma unroll
    for (int k = 0; k < 4; ++k) o[lane * 4 + k] = (w[2 * k] & 0x33333333u) | ((w[2 * k + 1] & 0x33333333u) << 2);
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      uint32_t p = 0;
#pragma unroll
      for (int c = 0; c < 4; ++c) p |= ((w[4 * k + c] >> 2) & 0x11111111u) << c;
      o[128 + lane * 2 + k] = p;
    }
    o[192 + lane] = b[256 + lane];
  }
}

// Blocks of `block` words, block i to out + i * stride + offset: gate and up interleaved as the kernels read them.
extern "C" __global__ void tf_qmoe_place(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, long blocks,
                                         int block, int stride, int offset) {
  for (long t = blockIdx.x * (long)blockDim.x + threadIdx.x; t < blocks * block; t += (long)gridDim.x * blockDim.x)
    out[(t / block) * stride + offset + t % block] = in[t];
}
