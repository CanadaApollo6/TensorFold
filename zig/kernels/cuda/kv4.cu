// A 4-bit attention cache for the 27B (TF_QWEN27_KV_BITS=4; the default bf16 cache is untouched): our Python
// fork's kvq format and attn4.cu, ported for the 3090. A row-head of 256 values is stored as 32-value groups rotated by
// H32 (its own inverse), then 128 bytes of 4-bit codes on the midpoint grid (low nibble first) and 8 fp16 scales (the
// rotated absmax): 144 bytes, not 512. The cache stays rotated: queries go in rotated and outputs come out rotated
// back, which keeps q.k. A window's own keys and values are packed before attention reads them, so a drafted row sees
// its path exactly as serial decoding later reads those rows from the cache.
//
// attn4 (from the fork's attn4.cu, unchanged below but for C entry points):
// Tree attention over a 4-bit cache (kvq BITS=4, head dim 256): a block of four warps per 512-key chunk and up to
// 48 (row, head) pairs, MT tiles of 16.
//
// The 3090 runs int->float and float->bf16 conversions at 1/8 of its float rate and Triton's kernel stages every
// dequantized tile through shared memory between barriers. Here each lane reads packed codes straight from the
// cache into its mma.sync fragments: a code nibble becomes bf16 by bit placement (0x3F80 | c << 3 = 1 + c/16), two
// packed bf16 FMAs make (c - 7.5)/16 exactly and then times bf16(2 * scale). The head's 256 values are assigned to
// MMA slots so that a lane's share is one contiguous run of a row: 32 bytes of a key row for q.k (the queries
// follow the same order) and 4 bytes of one scale group of four value rows for p.v (the output is written back in
// order). A step is 32 keys: each warp scores 8 of them for every tile, the block agrees on each row's maximum and
// sum through shared memory and each warp accumulates a quarter of the values. A key or value is loaded and
// decoded once for all the block's rows, and each row's arithmetic is the same whatever the tiling. Shared chunks
// and the tail (the last committed keys plus each row's own path) run the same code per step, so a full chunk gives
// identical bits in either and drafted rows equal serial ones.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <cuda_runtime.h>

#ifndef ABL
#define ABL 0     // timing ablations: 1 no bf16 FMAs, 2 no nibble math either, 3 no MMAs, 4 no cache loads
#endif

namespace {

constexpr int D = 256, R = 144, CH = 512, MAXD = 128;

// dequant constants, made once by opaque movs: otherwise the compiler rematerializes them before every use (about
// a hundred extra instructions a step)
struct Consts {
    uint32_t mask, one, off, nz;
};
__device__ __forceinline__ Consts consts() {
    Consts k;
    asm volatile("mov.b32 %0, 0x00780078;" : "=r"(k.mask));
    asm volatile("mov.b32 %0, 0x3F803F80;" : "=r"(k.one));
    asm volatile("mov.b32 %0, 0xBFBCBFBC;" : "=r"(k.off));
    asm volatile("mov.b32 %0, 0x80008000;" : "=r"(k.nz));
    return k;
}

__device__ __forceinline__ uint32_t nib_lo(uint32_t x, const Consts& k) {
    if (ABL == 2) return x;
    return ((x << 3) & k.mask) | k.one;
}
__device__ __forceinline__ uint32_t nib_hi(uint32_t x, const Consts& k) {
    if (ABL == 2) return x ^ 0x11u;
    return ((x >> 1) & k.mask) | k.one;
}

__device__ __forceinline__ uint32_t deq(uint32_t raw, uint32_t s, const Consts& k) {
    if (ABL == 1 || ABL == 2) return raw ^ s;
    uint32_t t, o;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(t) : "r"(raw), "r"(k.one), "r"(k.off));
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(o) : "r"(t), "r"(s), "r"(k.nz));
    return o;
}

__device__ __forceinline__ uint32_t scale_bf16(uint32_t h16) {          // bf16(2 * fp16 scale) in the low half
    float f = __half2float(__ushort_as_half((unsigned short)(h16 & 0xFFFFu))) * 2.0f;
    return (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(f));
}

__device__ __forceinline__ uint32_t cvt2(float lo, float hi) {
    uint32_t d;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}

__device__ __forceinline__ void mma(float* c, const uint4& a, uint32_t b0, uint32_t b1) {
    if (ABL == 3) {
        c[0] = __int_as_float(__float_as_int(c[0]) ^ b0 ^ b1 ^ a.x);
        return;
    }
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint4 ld16(const int8_t* p) {
    if (ABL == 4) {
        const uint32_t v = (uint32_t)(uintptr_t)p;
        return make_uint4(v, v * 3u, v * 5u, v * 7u);
    }
    return *reinterpret_cast<const uint4*>(p);
}
__device__ __forceinline__ uint32_t ld4(const int8_t* p) {
    if (ABL == 4) return ((uint32_t)(uintptr_t)p & 0x03FF03FFu) | 0x3C003C00u;
    return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ uint32_t ld2(const int8_t* p) {
    if (ABL == 4) return ((uint32_t)(uintptr_t)p & 0x03FFu) | 0x3C00u;
    return *reinterpret_cast<const unsigned short*>(p);
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// shared memory (4-byte words): q fragments (MT tiles x 16 k-steps x 32 lanes x 4), row maxima and sums of each
// warp's keys (2 x 4 warps x 16 MT), probabilities as p.v A fragment halves (MT x 4 warps x 32 lanes x 2)
template <int MT>
constexpr int smem_words() { return MT * 16 * 32 * 4 + 2 * 4 * 16 * MT + MT * 4 * 32 * 2; }

template <bool TAIL, int MT>
__device__ __forceinline__ void attn4_kernel(const __nv_bfloat16* __restrict__ Q,
        const int8_t* __restrict__ base, const int64_t* __restrict__ OFF, const int* __restrict__ STREAM,
        const int* __restrict__ ITEMS, int n_items, const int* __restrict__ ROWS, const int* __restrict__ PATHS,
        const int* __restrict__ DEPTHS, const int8_t* __restrict__ KN, const int8_t* __restrict__ VN,
        float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL, int W, int H, int HK, int G,
        float scale) {
    extern __shared__ uint4 smem[];
    uint4* qf = smem;
    float* rmax = reinterpret_cast<float*>(smem + MT * 16 * 32);            // [warp][16 MT]
    float* rsum = rmax + 4 * 16 * MT;
    uint2* pbuf = reinterpret_cast<uint2*>(rsum + 4 * 16 * MT);           // [MT][warp][lane]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, j = lane & 3;
    const int hk = TAIL ? blockIdx.y : blockIdx.x, bx = TAIL ? blockIdx.x : blockIdx.y;
    int s, chunk, p, steps, depth = 0, node_t = 0, first = 0, start = 0, rows = 0;
    if (!TAIL) {
        // the plan's 16-row tiles: a block takes MT of them from a multiple of MT. Blocks go to items 0, MT, 2 MT ..
        // first (the ones that work when a chunk's tiles are a multiple of MT), so the scheduler spreads them
        const int nrow = (n_items + MT - 1) / MT, item = (bx % nrow) * MT + bx / nrow;
        if (item >= n_items) return;
        s = ITEMS[item * 3];
        first = ITEMS[item * 3 + 1];
        if (first % (16 * MT)) return;
        chunk = ITEMS[item * 3 + 2];
        start = STREAM[s * 4];
        rows = STREAM[s * 4 + 1];
        p = STREAM[s * 4 + 2];
        if ((chunk + 1) * CH > p) return;                       // a plan padded for a longer context (a graph's)
        steps = CH / 32;
    } else {
        node_t = bx;
        s = ROWS[node_t];
        p = STREAM[s * 4 + 2];
        const int nch = STREAM[s * 4 + 3];
        chunk = p / CH + blockIdx.z;
        if (chunk >= nch) return;
        depth = DEPTHS[node_t];
        const int last = min(p + depth - chunk * CH, CH);      // past it every key is masked: steps there are no-ops
        steps = (last + 31) / 32;
    }
    const int64_t koff = OFF[s * 2], voff = OFF[s * 2 + 1];

    // row r of the block (0 .. 16 MT): its (node, head) and whether real
    auto row_info = [&](int r, int& node, int& head) -> bool {
        if (!TAIL) {
            const int pr = first + r;
            node = start + pr / G;
            head = hk * G + pr % G;
            return pr < rows * G;
        }
        node = node_t;
        head = hk * G + r;
        return r < G;
    };

    // queries as A fragments in the q.k slot order
    for (int idx = threadIdx.x; idx < MT * 16 * 32; idx += 128) {
        const int mt = idx >> 9, st = (idx >> 5) & 15, ln = idx & 31, g = ln >> 2, jj = ln & 3;
        const int d0 = 64 * jj + 8 * (st >> 1) + 4 * (st & 1);
        uint32_t lo[2], hi[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int node, head;
            uint2 v = make_uint2(0u, 0u);
            if (row_info(16 * mt + g + 8 * i, node, head))
                v = *reinterpret_cast<const uint2*>(Q + ((int64_t)node * H + head) * D + d0);
            lo[i] = __byte_perm(v.x, v.y, 0x5410);               // elements d0, d0 + 2
            hi[i] = __byte_perm(v.x, v.y, 0x7632);               // d0 + 1, d0 + 3
        }
        qf[idx] = make_uint4(lo[0], lo[1], hi[0], hi[1]);
    }
    __syncthreads();

    float o[MT][8][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int n = 0; n < 8; ++n) o[mt][n][0] = o[mt][n][1] = o[mt][n][2] = o[mt][n][3] = 0.f;
    float m[MT][2], l[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) m[mt][0] = m[mt][1] = -INFINITY, l[mt][0] = l[mt][1] = 0.f;
    const float LOG2E = 1.4426950408889634f;

    // a key's cache row, or its path row in the window (the tail), or none: then a safe row and zeros
    auto row_of = [&](int key, int64_t off, const int8_t* nodes, bool& valid) -> const int8_t* {
        if (TAIL && key >= p) {
            const int slot = min(key - p, depth - 1);           // path slots past the depth are unset
            valid = key - p < depth;
            return nodes + ((int64_t)PATHS[node_t * MAXD + slot] * HK + hk) * R;
        }
        valid = !TAIL || key < p;                               // shared chunks are full
        if (!TAIL) return base + off + ((int64_t)key * HK + hk) * R;
        return base + off + ((int64_t)(valid ? key : 0) * HK + hk) * R;
    };
    // q.k: this warp's key row is key0 + 8 warp + gid, bytes [32 j, 32 j + 32) and 2 scales
    uint4 kw[2];
    uint32_t ks;
    auto load_k = [&](int key0) {
        bool valid;
        const int8_t* row = row_of(key0 + 8 * warp + gid, koff, KN, valid);
        const uint4 a = ld16(row + 32 * j), b = ld16(row + 32 * j + 16);
        const uint32_t c = ld4(row + 128 + 4 * j);
        const uint4 z = make_uint4(0u, 0u, 0u, 0u);
        kw[0] = valid ? a : z;
        kw[1] = valid ? b : z;
        ks = valid ? c : 0u;
    };
    load_k(chunk * CH);
    const Consts kc = consts();

    for (int step = 0; step < steps; ++step) {
        const int key0 = chunk * CH + step * 32;
        // p.v rows, loaded now and used after q.k: of each 16 keys, 2j, 2j+1, 2j+8, 2j+9, bytes 16 gid + 4 warp ..
        uint32_t vw[2][4], vraw[2][4];
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                bool valid;
                const int8_t* row = row_of(key0 + 16 * h + 2 * j + (c & 1) + 8 * (c >> 1), voff, VN, valid);
                const uint32_t a = ld4(row + 16 * gid + 4 * warp);
                const uint32_t sc = ld2(row + 128 + 2 * gid);
                vw[h][c] = valid ? a : 0u;
                vraw[h][c] = valid ? sc : 0u;
            }
        float acc[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.f;
        {
            const uint32_t a = scale_bf16(ks), b = scale_bf16(ks >> 16);
            const uint32_t sk[2] = {a | (a << 16), b | (b << 16)};
#pragma unroll
            for (int u = 0; u < 8; ++u)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint32_t x = __byte_perm(word(kw[u >> 2], u & 3), 0u, h ? 0x3322 : 0x1100);
                    const uint32_t b0 = deq(nib_lo(x, kc), sk[u >> 2], kc), b1 = deq(nib_hi(x, kc), sk[u >> 2], kc);
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) mma(acc[mt], qf[(mt * 16 + 2 * u + h) * 32 + lane], b0, b1);
                }
        }
        load_k(chunk * CH + min(step + 1, steps - 1) * 32);     // the next step's key, during softmax and p.v
        // this warp's keys: key0 + 8 warp + 2j + e (C columns 2j, 2j+1)
        bool cvalid[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int key = key0 + 8 * warp + 2 * j + e;
            cvalid[e] = key < p || (TAIL && key - p < depth);
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                float mx = -INFINITY;
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    acc[mt][2 * i + e] = cvalid[e] ? acc[mt][2 * i + e] * scale : -INFINITY;
                    mx = fmaxf(mx, acc[mt][2 * i + e]);
                }
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                if (j == 0) rmax[warp * 16 * MT + 16 * mt + gid + 8 * i] = mx;
            }
        __syncthreads();
        // every warp forms the same new maximum; probabilities of its own keys, their sums
        float alpha[MT][2];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            uint32_t pw[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int r = 16 * mt + gid + 8 * i;
                const float mx = fmaxf(fmaxf(rmax[r], rmax[16 * MT + r]),
                                       fmaxf(rmax[2 * 16 * MT + r], rmax[3 * 16 * MT + r]));
                const bool active = mx != -INFINITY;
                const float next = active ? fmaxf(m[mt][i], mx) : m[mt][i];
                alpha[mt][i] = active ? (m[mt][i] == -INFINITY ? 0.f : ex2((m[mt][i] - next) * LOG2E)) : 1.f;
                m[mt][i] = next;
                float pv[2];
#pragma unroll
                for (int e = 0; e < 2; ++e)
                    pv[e] = (active && cvalid[e]) ? ex2((acc[mt][2 * i + e] - next) * LOG2E) : 0.f;
                float sum = pv[0] + pv[1];
                sum += __shfl_xor_sync(0xffffffffu, sum, 1);
                sum += __shfl_xor_sync(0xffffffffu, sum, 2);
                if (j == 0) rsum[warp * 16 * MT + r] = sum;
                pw[i] = cvt2(pv[0], pv[1]);
            }
            pbuf[(mt * 4 + warp) * 32 + lane] = make_uint2(pw[0], pw[1]);
        }
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int r = 16 * mt + gid + 8 * i;
                const float sum = (rsum[r] + rsum[16 * MT + r]) + (rsum[2 * 16 * MT + r] + rsum[3 * 16 * MT + r]);
                l[mt][i] = l[mt][i] * alpha[mt][i] + sum;
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    o[mt][n][2 * i] *= alpha[mt][i];
                    o[mt][n][2 * i + 1] *= alpha[mt][i];
                }
            }
        // p.v over this warp's 8 value tiles, the step's 32 keys as two k-steps
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint4 pa[MT];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const uint2 x = pbuf[(mt * 4 + 2 * h) * 32 + lane], y = pbuf[(mt * 4 + 2 * h + 1) * 32 + lane];
                pa[mt] = make_uint4(x.x, x.y, y.x, y.y);
            }
            const uint32_t s01 = scale_bf16(vraw[h][0]) | (scale_bf16(vraw[h][1]) << 16);
            const uint32_t s89 = scale_bf16(vraw[h][2]) | (scale_bf16(vraw[h][3]) << 16);
#pragma unroll
            for (int bb = 0; bb < 4; ++bb) {
                const uint32_t sel = bb | ((4 + bb) << 8);
                const uint32_t x01 = __byte_perm(vw[h][0], vw[h][1], sel), x89 = __byte_perm(vw[h][2], vw[h][3], sel);
                const uint32_t l0 = deq(nib_lo(x01, kc), s01, kc), l1 = deq(nib_lo(x89, kc), s89, kc);
                const uint32_t h0 = deq(nib_hi(x01, kc), s01, kc), h1 = deq(nib_hi(x89, kc), s89, kc);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    mma(o[mt][2 * bb], pa[mt], l0, l1);
                    mma(o[mt][2 * bb + 1], pa[mt], h0, h1);
                }
            }
        }
    }

    // value tile t (of 32) column n is value 32 n + t; this warp has tiles 8 warp .. 8 warp + 7, so lane j holds
    // values 64 j + 32 e + 8 warp + [0, 8) of rows gid, gid + 8 of each tile (e: the column's parity)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int node, head;
            if (!row_info(16 * mt + gid + 8 * i, node, head)) continue;
            const int64_t row = ((int64_t)chunk * W + node) * H + head;
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                float* out = PO + row * D + 64 * j + 32 * e + 8 * warp;
                const int c = 2 * i + e;
                *reinterpret_cast<float4*>(out) = make_float4(o[mt][0][c], o[mt][1][c], o[mt][2][c], o[mt][3][c]);
                *reinterpret_cast<float4*>(out + 4) =
                    make_float4(o[mt][4][c], o[mt][5][c], o[mt][6][c], o[mt][7][c]);
            }
            if (j == 0 && warp == 0) {
                PM[row] = m[mt][i];
                PL[row] = l[mt][i];
            }
        }
}

constexpr float INV_SQRT32 = 0.17677669529663688110f;

// H32 over 32 fp32 values in a thread's registers: kvq.h32's butterfly stages (stride 1 first), then 1/sqrt(32)
__device__ __forceinline__ void h32(float (&x)[32]) {
#pragma unroll
    for (int lo = 1; lo < 32; lo *= 2)
#pragma unroll
        for (int i = 0; i < 32; ++i)
            if (!(i & lo)) {
                const float a = x[i], b = x[i + lo];
                x[i] = __fadd_rn(a, b);
                x[i + lo] = __fsub_rn(a, b);
            }
#pragma unroll
    for (int i = 0; i < 32; ++i) x[i] = __fmul_rn(x[i], INV_SQRT32);
}

__device__ __forceinline__ void load32(const __nv_bfloat16* p, float (&x)[32]) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const uint4 v = reinterpret_cast<const uint4*>(p)[c];
        const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            x[8 * c + 2 * k] = __uint_as_float(w[k] << 16);
            x[8 * c + 2 * k + 1] = __uint_as_float(w[k] & 0xFFFF0000u);
        }
    }
}

}  // namespace

// x (groups x 32) bf16 -> y, each group by H32 (y may be x): one thread a group, so a row's bits never depend on how
// many rows there are
extern "C" __global__ void __launch_bounds__(128) tf_kv4_rotate(const __nv_bfloat16* __restrict__ x,
                                                                 __nv_bfloat16* y, long long groups) {
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    float v[32];
    load32(x + g * 32, v);
    h32(v);
    uint4* out = reinterpret_cast<uint4*>(y + g * 32);
#pragma unroll
    for (int c = 0; c < 4; ++c)
        out[c] = make_uint4(cvt2(v[8 * c], v[8 * c + 1]), cvt2(v[8 * c + 2], v[8 * c + 3]),
                            cvt2(v[8 * c + 4], v[8 * c + 5]), cvt2(v[8 * c + 6], v[8 * c + 7]));
}

// kvq._quant4: rows (row-heads x 256) bf16 -> packed row-heads (x 144 bytes), a thread a 32-value group
extern "C" __global__ void __launch_bounds__(128) tf_kv4_quant(const __nv_bfloat16* __restrict__ x,
                                                                int8_t* __restrict__ out, long long groups) {
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    float v[32];
    load32(x + g * 32, v);
    h32(v);
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) s = fmaxf(s, fabsf(v[i]));
    s = __fadd_rn(s, 1e-10f);
    const float r = __fdiv_rn(1.0f, s);
    uint32_t w[4] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        const float q = fminf(fmaxf(__fadd_rn(floorf(__fmul_rn(__fmul_rn(v[i], r), 8.0f)), 8.0f), 0.0f), 15.0f);
        w[i / 8] |= (uint32_t)q << (4 * (i % 8));
    }
    int8_t* row = out + (g / 8) * R + (g % 8) * 16;
    *reinterpret_cast<uint4*>(row) = make_uint4(w[0], w[1], w[2], w[3]);
    *reinterpret_cast<__half*>(out + (g / 8) * R + D / 2 + (g % 8) * 2) = __float2half_rn(s);
}

// attention._merge_head: a (row, head)'s chunk partials folded in order, a thread a value
extern "C" __global__ void __launch_bounds__(256) tf_kv4_merge(const float* __restrict__ PO,
        const float* __restrict__ PM, const float* __restrict__ PL, __nv_bfloat16* __restrict__ OUT,
        const int* __restrict__ STREAM, const int* __restrict__ ROWS, int W, int H) {
    const int node = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const int nch = STREAM[ROWS[node] * 4 + 3];
    const float LOG2E = 1.4426950408889634f;
    float m = -INFINITY, l = 0.0f, o = 0.0f;
    for (int c = 0; c < nch; ++c) {
        const int64_t at = ((int64_t)c * W + node) * H + head;
        const float cm = PM[at], cl = PL[at], co = PO[at * D + d];
        const bool active = cl > 0.0f;
        const float nm = active ? fmaxf(m, cm) : m;
        const float a = active ? (m == -INFINITY ? 0.0f : ex2(__fmul_rn(__fsub_rn(m, nm), LOG2E))) : 1.0f;
        const float b = active ? ex2(__fmul_rn(__fsub_rn(cm, nm), LOG2E)) : 0.0f;
        o = __fadd_rn(__fmul_rn(o, a), __fmul_rn(co, b));
        l = __fadd_rn(__fmul_rn(l, a), __fmul_rn(cl, b));
        m = nm;
    }
    OUT[((int64_t)node * H + head) * D + d] = __float2bfloat16_rn(__fdiv_rn(o, l));
}

#define TF_ATTN4_ARGS const __nv_bfloat16* __restrict__ Q, const int8_t* __restrict__ base,                   \
        const int64_t* __restrict__ OFF, const int* __restrict__ STREAM, const int* __restrict__ ITEMS, int n_items, \
        const int* __restrict__ ROWS, const int* __restrict__ PATHS, const int* __restrict__ DEPTHS,            \
        const int8_t* __restrict__ KN, const int8_t* __restrict__ VN, float* __restrict__ PO,                  \
        float* __restrict__ PM, float* __restrict__ PL, int W, int H, int HK, int G, float scale
#define TF_ATTN4_PASS Q, base, OFF, STREAM, ITEMS, n_items, ROWS, PATHS, DEPTHS, KN, VN, PO, PM, PL, W, H, HK, G, scale

// shared chunks: grid (kv heads, items rounded up to MT), MT 16-row tiles a block; the tail: grid (rows, kv heads, 2)
#define TF_SHARED(MT)                                                                                            \
    extern "C" __global__ void __launch_bounds__(128, MT == 3 ? 3 : 4) tf_attn4_shared_m##MT(TF_ATTN4_ARGS) {   \
        attn4_kernel<false, MT>(TF_ATTN4_PASS);                                                                  \
    }
TF_SHARED(1)
TF_SHARED(2)
TF_SHARED(3)
extern "C" __global__ void __launch_bounds__(128, 4) tf_attn4_tail(TF_ATTN4_ARGS) {
    attn4_kernel<true, 1>(TF_ATTN4_PASS);
}

// prefill_attention's kernel over a 4-bit cache: the same tiles, folds and bits as the bf16 kernel but each staged
// slot decoded from packed rows (queries in rotated, output rotated back by the caller)
#include "qmm_frag.cuh"

namespace tf_p4 {

using qmm_frag::commit;
using qmm_frag::cp16z;
using qmm_frag::ldmatrix4;
using qmm_frag::mma;
using qmm_frag::mma0;

constexpr int BN = 64, HALF = 32;             // keys a tile (the fold's unit), keys a staging slot
constexpr float LOG2E = 1.4426950408889634f;
constexpr int R4 = 144;                       // a 4-bit row-head: 128 bytes of codes, 8 fp16 scales

template <int N>
__device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(qmm_frag::smem(p)));
}

__device__ __forceinline__ float texp(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(__fmul_rn(x, LOG2E)));
    return y;
}

__device__ __forceinline__ float tdiv(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;\n" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

template <int D>
__device__ __forceinline__ int sw(int r, int c) { return r * (D / 8) + (c ^ (r & 7)); }

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

template <int D, int WARPS, int HPC, int NS>
__device__ __forceinline__ void pattn4(const __nv_bfloat16* __restrict__ q, const int8_t* __restrict__ kc,
             const int8_t* __restrict__ vc, __nv_bfloat16* __restrict__ out, int p0, int W, int H, int HK,
             int G, float scale) {
    constexpr int C = D / 8, K16 = D / 16, NB = D / 8, RB = WARPS / HPC, THREADS = 32 * WARPS;
    constexpr bool QREG = D <= 128;               // the queries' fragments stay in registers, else in shared memory
    extern __shared__ uint4 smem[];
    uint4* slots = smem;
    uint4* qs = smem + NS * HALF * C;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, qd = lane % 4, mt = lane / 8;
    const int groups = G / HPC, hk = blockIdx.y / groups, hg = blockIdx.y % groups;
    const int row0 = blockIdx.x * 16 * RB, last = min(row0 + 16 * RB, W) - 1;
    const int head = hk * G + hg * HPC + warp % HPC, wrow = row0 + (warp / HPC) * 16;
    const int tiles = (p0 + last) / BN + 1, items = 4 * tiles, limit = p0 + W;
    // a slot's 32 keys decoded from 4-bit rows by plain loads (the barrier in take() publishes them): eight values
    // a store, one packed word and its group's scale, the decode attention's bits (bf16((c - 7.5) / 16 * bf16(2 s)))
    auto stage = [&](int item) {
        const int8_t* src = item % 4 < 2 ? kc : vc;
        const int key0 = (item / 4) * BN + (item % 2) * HALF;
        uint4* dst = slots + (item % NS) * HALF * C;
        for (int i = threadIdx.x; i < HALF * C; i += THREADS) {
            const int key = key0 + i / C, cc = i % C;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (key < limit) {                       // past the cache's filled keys: zeros, never read
                const int8_t* row = src + ((int64_t)key * HK + hk) * R4;
                const uint32_t w = *reinterpret_cast<const uint32_t*>(row + 4 * cc);
                const float s = __bfloat162float(__float2bfloat16_rn(
                    2.0f * __half2float(*reinterpret_cast<const __half*>(row + D / 2 + 2 * (cc / 4)))));
                auto at = [&](int k) { return __fmul_rn(__fmul_rn(__int2float_rn((int)((w >> (4 * k)) & 15u)) - 7.5f, 0.0625f), s); };
                v = make_uint4(pack(at(0), at(1)), pack(at(2), at(3)), pack(at(4), at(5)), pack(at(6), at(7)));
            }
            dst[sw<D>(i / C, cc)] = v;
        }
    };
    uint32_t qa[QREG ? K16 : 1][4];
    if constexpr (QREG) {
        const int r0 = wrow + g, r1 = r0 + 8;
        const uint32_t* q0 = reinterpret_cast<const uint32_t*>(q + ((int64_t)min(r0, W - 1) * H + head) * D);
        const uint32_t* q1 = reinterpret_cast<const uint32_t*>(q + ((int64_t)min(r1, W - 1) * H + head) * D);
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            qa[k][0] = r0 < W ? q0[8 * k + qd] : 0u;
            qa[k][1] = r1 < W ? q1[8 * k + qd] : 0u;
            qa[k][2] = r0 < W ? q0[8 * k + qd + 4] : 0u;
            qa[k][3] = r1 < W ? q1[8 * k + qd + 4] : 0u;
        }
    } else {
        for (int i = threadIdx.x; i < WARPS * 16 * C; i += THREADS) {
            const int w = i / (16 * C), r = (i / C) % 16, row = row0 + (w / HPC) * 16 + r;
            const int hd = hk * G + hg * HPC + w % HPC;
            cp16z(qs + sw<D>(i / C, i % C), q + ((int64_t)min(row, W - 1) * H + hd) * D + (i % C) * 8, row < W);
        }
        commit();
    }
#pragma unroll
    for (int i = 0; i < NS; ++i) {
        if (i < items) stage(i);
        commit();
    }
    float o[NB][4];
#pragma unroll
    for (int i = 0; i < NB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float sc[8][4];
    uint32_t pa[4][4];
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const int pos0 = p0 + wrow + g, pos1 = pos0 + 8;
    auto take = [&](int it) -> const uint4* {           // item it's slot, once every thread's copy has landed
        wait_group<NS - 1>();
        __syncthreads();
        return slots + (it % NS) * HALF * C;
    };
    auto done = [&](int it) {                            // the slot is free: stage the item NS ahead into it
        __syncthreads();
        if (it + NS < items) stage(it + NS);
        commit();
    };
    auto scores = [&](float (&s)[4][4], const uint4* sl) {
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            uint32_t a[4];
            if constexpr (QREG) {
                a[0] = qa[k][0]; a[1] = qa[k][1]; a[2] = qa[k][2]; a[3] = qa[k][3];
            } else {
                ldmatrix4(a, qs + sw<D>(warp * 16 + lane % 16, 2 * k + lane / 16));
            }
#pragma unroll
            for (int n = 0; n < 4; n += 2) {
                uint32_t b[4];
                ldmatrix4(b, sl + sw<D>(8 * (n + mt / 2) + lane % 8, 2 * k + mt % 2));
                if (k == 0) {
                    mma0(s[n], a, b[0], b[1]);
                    mma0(s[n + 1], a, b[2], b[3]);
                } else {
                    mma(s[n], a, b[0], b[1]);
                    mma(s[n + 1], a, b[2], b[3]);
                }
            }
        }
    };
    auto values = [&](const uint32_t (&pa0)[4], const uint32_t (&pa1)[4], const uint4* sl) {
#pragma unroll
        for (int i = 0; i < NB; i += 2) {
            uint32_t b[4];
            ldmatrix4t(b, sl + sw<D>(8 * (mt % 2) + lane % 8, i + mt / 2));
            mma(o[i], pa0, b[0], b[1]);
            mma(o[i + 1], pa0, b[2], b[3]);
        }
#pragma unroll
        for (int i = 0; i < NB; i += 2) {
            uint32_t b[4];
            ldmatrix4t(b, sl + sw<D>(16 + 8 * (mt % 2) + lane % 8, i + mt / 2));
            mma(o[i], pa1, b[0], b[1]);
            mma(o[i + 1], pa1, b[2], b[3]);
        }
    };
    for (int t = 0; t < tiles; ++t) {
        const int it = 4 * t;
        scores(*reinterpret_cast<float(*)[4][4]>(&sc[0]), take(it));
        done(it);
        scores(*reinterpret_cast<float(*)[4][4]>(&sc[4]), take(it + 1));
        done(it + 1);
        {                                                // _tile's fold of the tile's 64 scores
            const int kb = t * BN + 2 * qd;
            float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int c = kb + 8 * n;
                sc[n][0] = c <= pos0 ? __fmul_rn(sc[n][0], scale) : -INFINITY;
                sc[n][1] = c + 1 <= pos0 ? __fmul_rn(sc[n][1], scale) : -INFINITY;
                sc[n][2] = c <= pos1 ? __fmul_rn(sc[n][2], scale) : -INFINITY;
                sc[n][3] = c + 1 <= pos1 ? __fmul_rn(sc[n][3], scale) : -INFINITY;
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
                const int c = kb + 8 * n;
                sc[n][0] = act0 && c <= pos0 ? texp(__fsub_rn(sc[n][0], n0)) : 0.0f;
                sc[n][1] = act0 && c + 1 <= pos0 ? texp(__fsub_rn(sc[n][1], n0)) : 0.0f;
                sc[n][2] = act1 && c <= pos1 ? texp(__fsub_rn(sc[n][2], n1)) : 0.0f;
                sc[n][3] = act1 && c + 1 <= pos1 ? texp(__fsub_rn(sc[n][3], n1)) : 0.0f;
                c0[n] = __fadd_rn(sc[n][0], sc[n][1]);
                c1[n] = __fadd_rn(sc[n][2], sc[n][3]);
            }
            l0 = __fmaf_rn(l0, a0, rowsum(c0));
            l1 = __fmaf_rn(l1, a1, rowsum(c1));
            m0 = n0;
            m1 = n1;
#pragma unroll
            for (int i = 0; i < NB; ++i) {
                o[i][0] = __fmul_rn(o[i][0], a0);
                o[i][1] = __fmul_rn(o[i][1], a0);
                o[i][2] = __fmul_rn(o[i][2], a1);
                o[i][3] = __fmul_rn(o[i][3], a1);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                pa[j][0] = pack(sc[2 * j][0], sc[2 * j][1]);
                pa[j][1] = pack(sc[2 * j][2], sc[2 * j][3]);
                pa[j][2] = pack(sc[2 * j + 1][0], sc[2 * j + 1][1]);
                pa[j][3] = pack(sc[2 * j + 1][2], sc[2 * j + 1][3]);
            }
        }
        values(pa[0], pa[1], take(it + 2));
        done(it + 2);
        values(pa[2], pa[3], take(it + 3));
        done(it + 3);
    }
    wait_group<0>();
    const int r0 = wrow + g, r1 = r0 + 8;
#pragma unroll
    for (int i = 0; i < NB; ++i) {
        const int d = 8 * i + 2 * qd;
        if (r0 < W)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)r0 * H + head) * D + d) =
                pack(tdiv(o[i][0], l0), tdiv(o[i][1], l0));
        if (r1 < W)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)r1 * H + head) * D + d) =
                pack(tdiv(o[i][2], l1), tdiv(o[i][3], l1));
    }
}

} // namespace tf_p4

// head dim 256, eight warps, two query heads a block, two staging slots (prefill_attention's Qwen3.8 shape)
extern "C" __global__ void __launch_bounds__(256, 1) tf_pattn4(const __nv_bfloat16* __restrict__ q,
        const int8_t* __restrict__ kc, const int8_t* __restrict__ vc, __nv_bfloat16* __restrict__ out, int p0, int W,
        int H, int HK, int G, float scale) {
    tf_p4::pattn4<256, 8, 2, 2>(q, kc, vc, out, p0, W, H, HK, G, scale);
}
