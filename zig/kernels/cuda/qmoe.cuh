// The 3090 stack's grouped experts (2-, 3- and 4-bit MLX affine), shared by qmoe.cu and qmoe_prefill.cu.
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

namespace qmoe {

constexpr int NTW = 4;                       // n8 tiles a warp: 32 output columns
constexpr int COLS = 8 * NTW;
constexpr uint32_t LOW = 0x000F000Fu;
constexpr uint32_t K128 = 0x43004300u;       // the bf16 pair (128, 128)

// A block is one group of one 32-column tile: each lane's weight words, then scales and biases.
// 4-bit: WORDS nibble words a lane, word wi at uint4 wi / 4 (lane-major). 2- and 3-bit (groups of 64): bit planes
// that rebuild the same nibble words (see ``nibword``): 2-bit is one uint4 a lane of 2-bit fields; 3-bit adds a
// 1-bit plane, one uint2 a lane after the 32 uint4.
template <int GS, int BITS = 4>
struct Geo {
  static_assert(BITS == 4 || (GS == 64 && (BITS == 2 || BITS == 3)), "2- and 3-bit experts take groups of 64");
  static constexpr int KS = GS / 16;             // mma k-steps a group
  static constexpr int WORDS = NTW * GS / 32 * BITS / 4;   // weight uint32 a lane a group (8 for 4-bit, groups of 64)
  static constexpr int WU = 8 * WORDS;           // weight uint4 a block
  static constexpr int XV = GS / 32;             // input uint4 a row a lane a group
  static constexpr int SBV = NTW / 2;            // scale and bias uint4 a quad a group
  static constexpr int BLOCK = WU + 4 * SBV;
};

// A lane's raw words of one block (b: the block's first uint4); L loads a uint4, L2 a uint2.
template <int GS, int BITS, class L, class L2>
__device__ __forceinline__ void lane_words(uint32_t (&r)[Geo<GS, BITS>::WORDS], const uint4* b, int lane, L ld4,
                                           L2 ld2) {
  if constexpr (BITS == 3) {
    const uint4 v = ld4(b + lane);
    const uint2 u = ld2(reinterpret_cast<const uint2*>(b + 32) + lane);
    r[0] = v.x; r[1] = v.y; r[2] = v.z; r[3] = v.w; r[4] = u.x; r[5] = u.y;
  } else {
#pragma unroll
    for (int c = 0; c < Geo<GS, BITS>::WORDS / 4; ++c) {
      const uint4 v = ld4(b + c * 32 + lane);
      r[4 * c] = v.x; r[4 * c + 1] = v.y; r[4 * c + 2] = v.z; r[4 * c + 3] = v.w;
    }
  }
}

// Nibble word wi of a lane (the 4-bit layout's word): 2-bit fields sit in the low half of each nibble, two words a
// uint32 (word 2i at bits 4s, word 2i + 1 at bits 4s + 2); 3-bit adds each nibble's bit 2 from a plane of four words
// a uint32 (word 4i + c at bits 4s + c).
template <int BITS, int N>
__device__ __forceinline__ uint32_t nibword(const uint32_t (&r)[N], int wi) {
  if constexpr (BITS == 4) {
    return r[wi];
  } else {
    const uint32_t lo = (r[wi >> 1] >> (2 * (wi & 1))) & 0x33333333u;
    if constexpr (BITS == 2) return lo;
    else return lo | (((r[4 + (wi >> 2)] >> (wi & 3)) & 0x11111111u) << 2);
  }
}

__device__ __forceinline__ void cp16(void* dst, const void* src) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ uint32_t comp(const uint4& v, int c) {
  return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w;
}

// nibbles at bits sh and sh + 16 -> the exact bf16 pair (q0, q1): (128 + q) - 128
__device__ __forceinline__ uint32_t nib2(uint32_t w, int sh) {
  uint32_t v = ((w >> sh) & LOW) | K128;
  const uint32_t k = K128;
  __nv_bfloat162 r = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&v), *reinterpret_cast<const __nv_bfloat162*>(&k));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ void mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float lo_f(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi_f(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }
__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// SwiGLU as the families define it: bf16(bf16(silu(g)) * u) on bf16 g and u, clipped first when limit > 0
__device__ __forceinline__ float swiglu(float g, float u, float limit) {
  float gv = bf(g), uv = bf(u);
  if (limit > 0.f) {
    gv = fminf(gv, limit);
    uv = fminf(fmaxf(uv, -limit), limit);
  }
  return bf(gv / (1.f + expf(-gv))) * uv;
}

__device__ __forceinline__ float relu2(float a) {
  const float u = fmaxf(bf(a), 0.f);
  return u * u;
}

// EPI 0: fp32 out (down); 1: bf16 relu(bf16(acc))^2; 2: bf16 SwiGLU(matrix 0, matrix 1); 3: bf16 out (prefill down)
// Writes the n8 tiles j with bit j of js set.
template <int EPI, int M, int RT>
__device__ __forceinline__ void epilogue(const float (&acc)[M][RT][NTW][4], int r, void* out, int N, int col0, int pr0,
                                         int pr1, bool v0, bool v1, float limit, unsigned js = (1u << NTW) - 1u) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!(h ? v1 : v0)) continue;
    const size_t row = (size_t)(h ? pr1 : pr0) * N;
#pragma unroll
    for (int j = 0; j < NTW; ++j) {
      if (!((js >> j) & 1u)) continue;
      const int col = col0 + 8 * j;
      const float a0 = acc[0][r][j][2 * h], a1 = acc[0][r][j][2 * h + 1];
      if constexpr (EPI == 0) {
        *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + row + col) = make_float2(a0, a1);
      } else {
        float o0, o1;
        if constexpr (EPI == 1) {
          o0 = relu2(a0);
          o1 = relu2(a1);
        } else if constexpr (EPI == 2) {
          o0 = swiglu(a0, acc[M - 1][r][j][2 * h], limit);
          o1 = swiglu(a1, acc[M - 1][r][j][2 * h + 1], limit);
        } else {
          o0 = a0;
          o1 = a1;
        }
        *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + row + col) =
            __floats2bfloat162_rn(o0, o1);
      }
    }
  }
}

}  // namespace
