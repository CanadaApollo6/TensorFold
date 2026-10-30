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

__device__ void merge(const float* scratch, __nv_bfloat16* out, int dim, int groups) {
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

extern "C" __global__ void tf_kv4_merge(const float* scratch, __nv_bfloat16* out, int dim, int groups) {
    merge(scratch, out, dim, groups);
}

// Prompt tiles: 64 decoded keys serve 16 rows of a GQA group; absolute 2048-key segments keep sums chunk-free.
namespace tiles {
constexpr int D = 128, BN = 64, SEG = 2048, RAW = D / 2 + D / 16;

__device__ __forceinline__ uint32_t at(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void cp8(void* dst, const void* src, bool in) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(at(dst)), "l"(src), "r"(in ? 8 : 0));
}

__device__ __forceinline__ void ldm(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(at(p)));
}

__device__ __forceinline__ void ldmt(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(at(p)));
}

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t bits(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }

__device__ __forceinline__ void split(float a, float b, uint32_t& hi, uint32_t& lo) {
    const __half2 h = __floats2half2_rn(a, b);
    const float2 f = __half22float2(h);
    hi = bits(h);
    lo = bits(__floats2half2_rn(__fsub_rn(a, f.x), __fsub_rn(b, f.y)));
}

// A masked key (later in this prompt) may hold a non-finite scale; visible keys keep their NaN as decode does
__device__ __forceinline__ float weigh(float p, float scale) { return p > 0.0f ? __fmul_rn(p, scale) : 0.0f; }

__device__ __forceinline__ float texp(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(__fmul_rn(x, 1.4426950408889634f)));
    return y;
}

__device__ __forceinline__ float rowsum(const float (&c)[8]) {
    float w[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float x = __fadd_rn(c[j], __shfl_xor_sync(0xffffffffu, c[j], 2));
        w[j] = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 1));
    }
    return __fadd_rn(__fadd_rn(__fadd_rn(w[0], w[4]), __fadd_rn(w[2], w[6])),
                     __fadd_rn(__fadd_rn(w[1], w[5]), __fadd_rn(w[3], w[7])));
}

__device__ __forceinline__ int sw(int r, int c) { return r * (D / 8) + (c ^ (r & 7)); }
}

// grid (16-row blocks, key heads, segments), a warp a query head of the group; partials [row][head][segment][D + 2]
extern "C" __global__ void __launch_bounds__(256, 1)
tf_kv4_tiles128(const __nv_bfloat16* q, const unsigned char* kc, const unsigned char* vc, float* scratch, int heads,
                int kvheads, int p0, int rows, int segs) {
    using namespace tiles;
    extern __shared__ uint4 smem[];
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31, g = lane / 4, qd = lane % 4, mt = lane / 8;
    const int kh = blockIdx.y, head = kh * (heads / kvheads) + warp, s = blockIdx.z, wrow = blockIdx.x * 16;
    const int kb = max(1, s * SEG), ke = min((s + 1) * SEG, p0 + min(wrow + 15, rows - 1));
    if (kb >= ke) return;
    unsigned char* raw = reinterpret_cast<unsigned char*>(smem);  // two slots of [K, V][BN][RAW] cache bytes
    uint4* kt = smem + 4 * BN * RAW / 16;                         // [BN][D] key codes 2c - 15
    uint4* vt = kt + BN * D / 8;                                  // [BN][D] value codes 2c - 15
    float* ks = reinterpret_cast<float*>(vt + BN * D / 8);        // [BN][4] key scales / 16
    float* vs = ks + BN * 4;                                      // [BN][4] value scales; sums carry a factor of 16
    float* stage = vs + BN * 4 + warp * 16 * 32;                  // a warp's 16 turned query rows, 32 dims at a time
    const int r0 = wrow + g, r1 = r0 + 8;
    const int pos0 = r0 < rows ? p0 + r0 : 0, pos1 = r1 < rows ? p0 + r1 : 0;
    uint32_t qh[D / 16][4], ql[D / 16][4];
    // Turned query rows past the f16 split's range scale by 2^-e (exact; e = 0 when small) and scores take 2^e back.
    float* down = ks + warp * 16;  // free until the first decode, which follows a block barrier
    float up0 = 1.0f, up1 = 1.0f;
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int row = wrow + r;
        float big = 0.0f;
#pragma unroll
        for (int gi = 0; gi < D / 32; ++gi) {
            const float x = row < rows ? __bfloat162float(q[((size_t)row * heads + head) * D + 32 * gi + lane]) : 0.0f;
            big = fmaxf(big, fabsf(turn(x)));
        }
        for (int x = 16; x; x /= 2) big = fmaxf(big, __shfl_xor_sync(0xffffffffu, big, x));
        const int e = big >= 16384.0f && big <= 3.0e38f ? ilogbf(big) - 13 : 0;
        if (lane == 0) down[r] = ldexpf(1.0f, -e);
        if (r == g) up0 = ldexpf(1.0f, e);
        if (r == g + 8) up1 = ldexpf(1.0f, e);
    }
    __syncwarp();
#pragma unroll
    for (int gi = 0; gi < D / 32; ++gi) {
        for (int r = 0; r < 16; ++r) {
            const int row = wrow + r;
            const float x = row < rows ? __bfloat162float(q[((size_t)row * heads + head) * D + 32 * gi + lane]) : 0.0f;
            stage[r * 32 + lane] = __fmul_rn(turn(x), down[r]);
        }
        __syncwarp();
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int k = 2 * gi + h, c = 16 * h + 2 * qd;
            split(stage[g * 32 + c], stage[g * 32 + c + 1], qh[k][0], ql[k][0]);
            split(stage[(g + 8) * 32 + c], stage[(g + 8) * 32 + c + 1], qh[k][1], ql[k][1]);
            split(stage[g * 32 + c + 8], stage[g * 32 + c + 9], qh[k][2], ql[k][2]);
            split(stage[(g + 8) * 32 + c + 8], stage[(g + 8) * 32 + c + 9], qh[k][3], ql[k][3]);
        }
        __syncwarp();
    }
    auto fetch = [&](int t) {
        unsigned char* dst = raw + (t & 1) * 2 * BN * RAW;
        for (int i = threadIdx.x; i < 2 * BN * (RAW / 8); i += blockDim.x) {
            const int kv = i / (BN * (RAW / 8)), key = (i / (RAW / 8)) % BN, j = i % (RAW / 8), pos = t * BN + key;
            // BOS comes from its bf16 copy in the merge; its packed row may be non-finite and 0 * NaN survives masking.
            const bool in = pos >= kb && pos < ke;
            const unsigned char* from = (kv ? vc : kc) + ((size_t)(in ? pos : 0) * kvheads + kh) * RAW + 8 * j;
            cp8(dst + (kv * BN + key) * RAW + 8 * j, from, in);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    auto decode = [&](int t) {
        const unsigned char* src = raw + (t & 1) * 2 * BN * RAW;
        for (int i = threadIdx.x; i < 2 * BN * (D / 8); i += blockDim.x) {
            const int kv = i / (BN * (D / 8)), key = (i / (D / 8)) % BN, c = i % (D / 8);
            const unsigned char* row = src + (kv * BN + key) * RAW;
            const uint32_t w = *reinterpret_cast<const uint32_t*>(row + 4 * c);
            const float scale = __half2float(*reinterpret_cast<const __half*>(row + D / 2 + 2 * (c / 4)));
            uint32_t h[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float a = float((w >> (8 * j)) & 15), b = float((w >> (8 * j + 4)) & 15);
                h[j] = bits(__floats2half2_rn(2.0f * a - 15.0f, 2.0f * b - 15.0f));
            }
            (kv ? vt : kt)[sw(key, c)] = make_uint4(h[0], h[1], h[2], h[3]);
            if (c % 4 == 0) (kv ? vs : ks)[key * 4 + c / 4] = kv ? scale : scale * 0.0625f;
        }
    };
    float o[D / 8][4];
#pragma unroll
    for (int i = 0; i < D / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const float scale = rsqrtf(float(D)), scale0 = scale * up0, scale1 = scale * up1;
    const int first = kb / BN, last = (ke - 1) / BN;
    fetch(first);
    for (int t = first; t <= last; ++t) {
        if (t < last) {
            fetch(t + 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        decode(t);
        __syncthreads();
        float sc[8][4];
#pragma unroll
        for (int n = 0; n < 8; n += 2) {
#pragma unroll
            for (int gi = 0; gi < D / 32; ++gi) {
                float x0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, x1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int k = 2 * gi + h;
                    uint32_t b[4];
                    ldm(b, kt + sw(8 * (n + mt / 2) + lane % 8, 2 * k + mt % 2));
                    mma(x0, qh[k], b[0], b[1]);
                    mma(x0, ql[k], b[0], b[1]);
                    mma(x1, qh[k], b[2], b[3]);
                    mma(x1, ql[k], b[2], b[3]);
                }
                const float* k0 = ks + (8 * n + 2 * qd) * 4 + gi;
                const float* k1 = k0 + 8 * 4;
                const float f[4] = {k0[0], k0[4], k1[0], k1[4]};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float y0 = __fmul_rn(x0[e], f[e % 2]), y1 = __fmul_rn(x1[e], f[2 + e % 2]);
                    sc[n][e] = gi ? __fadd_rn(sc[n][e], y0) : y0;
                    sc[n + 1][e] = gi ? __fadd_rn(sc[n + 1][e], y1) : y1;
                }
            }
        }
        {
            const int kb0 = t * BN + 2 * qd;
            float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int c = kb0 + 8 * n;
                sc[n][0] = c >= 1 && c < pos0 ? __fmul_rn(sc[n][0], scale0) : -INFINITY;
                sc[n][1] = c + 1 < pos0 ? __fmul_rn(sc[n][1], scale0) : -INFINITY;
                sc[n][2] = c >= 1 && c < pos1 ? __fmul_rn(sc[n][2], scale1) : -INFINITY;
                sc[n][3] = c + 1 < pos1 ? __fmul_rn(sc[n][3], scale1) : -INFINITY;
                tm0 = fmaxf(tm0, fmaxf(sc[n][0], sc[n][1]));
                tm1 = fmaxf(tm1, fmaxf(sc[n][2], sc[n][3]));
            }
            for (int x = 1; x <= 2; x *= 2) {
                tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
                tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
            }
            const bool act0 = tm0 != -INFINITY, act1 = tm1 != -INFINITY;
            const float n0 = act0 ? fmaxf(m0, tm0) : m0, n1 = act1 ? fmaxf(m1, tm1) : m1;
            const float a0 = act0 ? (m0 == -INFINITY ? 0.0f : texp(__fsub_rn(m0, n0))) : 1.0f;
            const float a1 = act1 ? (m1 == -INFINITY ? 0.0f : texp(__fsub_rn(m1, n1))) : 1.0f;
            float c0[8], c1[8];
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                sc[n][0] = act0 && sc[n][0] != -INFINITY ? texp(__fsub_rn(sc[n][0], n0)) : 0.0f;
                sc[n][1] = act0 && sc[n][1] != -INFINITY ? texp(__fsub_rn(sc[n][1], n0)) : 0.0f;
                sc[n][2] = act1 && sc[n][2] != -INFINITY ? texp(__fsub_rn(sc[n][2], n1)) : 0.0f;
                sc[n][3] = act1 && sc[n][3] != -INFINITY ? texp(__fsub_rn(sc[n][3], n1)) : 0.0f;
                c0[n] = __fadd_rn(sc[n][0], sc[n][1]);
                c1[n] = __fadd_rn(sc[n][2], sc[n][3]);
            }
            l0 = __fmaf_rn(l0, a0, rowsum(c0));
            l1 = __fmaf_rn(l1, a1, rowsum(c1));
            m0 = n0;
            m1 = n1;
#pragma unroll
            for (int i = 0; i < D / 8; ++i) {
                o[i][0] = __fmul_rn(o[i][0], a0);
                o[i][1] = __fmul_rn(o[i][1], a0);
                o[i][2] = __fmul_rn(o[i][2], a1);
                o[i][3] = __fmul_rn(o[i][3], a1);
            }
        }
        // A value group's scale rides on P, split hi + lo, so value codes stay exact and P keeps about 22 bits.
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int gi = 0; gi < D / 32; ++gi) {
                const float* v = vs + (16 * j + 2 * qd) * 4 + gi;
                const float f0 = v[0], f1 = v[4], f2 = v[32], f3 = v[36];
                uint32_t ah[4], al[4];
                split(weigh(sc[2 * j][0], f0), weigh(sc[2 * j][1], f1), ah[0], al[0]);
                split(weigh(sc[2 * j][2], f0), weigh(sc[2 * j][3], f1), ah[1], al[1]);
                split(weigh(sc[2 * j + 1][0], f2), weigh(sc[2 * j + 1][1], f3), ah[2], al[2]);
                split(weigh(sc[2 * j + 1][2], f2), weigh(sc[2 * j + 1][3], f3), ah[3], al[3]);
#pragma unroll
                for (int i = 4 * gi; i < 4 * gi + 4; i += 2) {
                    uint32_t b[4];
                    ldmt(b, vt + sw(16 * j + 8 * (mt % 2) + lane % 8, i + mt / 2));
                    mma(o[i], ah, b[0], b[1]);
                    mma(o[i], al, b[0], b[1]);
                    mma(o[i + 1], ah, b[2], b[3]);
                    mma(o[i + 1], al, b[2], b[3]);
                }
            }
        }
    }
    auto put = [&](int row, int h, float m, float l) {
        if (row >= rows) return;
        float* p = scratch + (((size_t)row * heads + head) * segs + s) * (D + 2);
#pragma unroll
        for (int i = 0; i < D / 8; ++i)
            *reinterpret_cast<float2*>(p + 8 * i + 2 * qd) =
                make_float2(o[i][2 * h] * 0.0625f, o[i][2 * h + 1] * 0.0625f);
        if (qd == 0) {
            p[D] = m;
            p[D + 1] = l;
        }
    };
    put(r0, 0, m0, l0);
    put(r1, 1, m1, l1);
}

// grid (heads, rows), D threads: BOS, the row's segments in order, then its own key; turned back to bf16.
extern "C" __global__ void tf_kv4_tile_merge(const __nv_bfloat16* q, const __nv_bfloat16* first_k,
                                             const __nv_bfloat16* first_v, const __nv_bfloat16* last_k,
                                             const __nv_bfloat16* last_v, const float* scratch, __nv_bfloat16* out,
                                             int heads, int kvheads, int p0, int segs) {
    constexpr int D = tiles::D;
    __shared__ float qs[D], keys[2][D], dots[2];
    const int d = threadIdx.x, head = blockIdx.x, row = blockIdx.y, kh = head / (heads / kvheads), pos = p0 + row;
    const size_t at = ((size_t)row * heads + head) * D + d, own = ((size_t)row * kvheads + kh) * D + d;
    qs[d] = turn(__bfloat162float(q[at]));
    keys[0][d] = turn(__bfloat162float(first_k[kh * D + d]));
    keys[1][d] = turn(__bfloat162float(last_k[own]));
    const float bos = turn(__bfloat162float(first_v[kh * D + d])), self = turn(__bfloat162float(last_v[own]));
    __syncthreads();
    if (d < 64) {
        const int lane = d & 31;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < D / 32; ++i) dot = fmaf(qs[32 * i + lane], keys[d / 32][32 * i + lane], dot);
#pragma unroll
        for (int bit = 16; bit; bit /= 2) dot += __shfl_xor_sync(0xffffffffu, dot, bit);
        if (lane == 0) dots[d / 32] = dot * rsqrtf(float(D));
    }
    __syncthreads();
    float maximum = dots[0], sum = 1.0f, value = bos;
    auto fold = [&](float m, float l, float v) {
        const float next = fmaxf(maximum, m);
        const float a = expf(maximum - next), b = expf(m - next);
        value = value * a + v * b;
        sum = sum * a + l * b;
        maximum = next;
    };
    const float* p = scratch + ((size_t)row * heads + head) * segs * (D + 2);
    for (int s = 0; pos >= 2 && s <= (pos - 1) / tiles::SEG; ++s, p += D + 2) fold(p[D], p[D + 1], p[d]);
    if (pos > 0) fold(dots[1], 1.0f, self);
    out[at] = __float2bfloat16_rn(turn(value / sum));
}
