// The pair predicate keeps cached loads inside the score-pair loop, limiting query fragment lifetimes.
__device__ __forceinline__ uint32_t prompt_code(const uint32_t* p, int pair) {
    uint32_t v;
    asm volatile("{ .reg .pred in; setp.lt.u32 in, %2, 8; .reg .u32 data; mov.u32 data, 0; @in ld.global.ca.u32 data, [%1]; mov.u32 %0, data; }"
                 : "=r"(v) : "l"(p), "r"(pair) : "memory");
    return v;
}

// Prepare the original query fragments once per row tile, shared by all absolute key segments.
extern "C" __global__ void __launch_bounds__(32)
tf_kv4_prepare128(const __nv_bfloat16* q, float* prepared, int heads, int rows) {
    using namespace tiles;
    __shared__ float stage[16 * 32 + 16];
    const int lane = threadIdx.x, g = lane / 4, qd = lane % 4;
    const int head = blockIdx.y, wrow = blockIdx.x * 16;
    uint32_t qh[D / 16][4], ql[D / 16][4];
    // Range scaling and fragment splits use the original query arithmetic.
    float* down = stage + 16 * 32;
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
    float* packed = prepared + ((size_t)blockIdx.x * heads + head) * (16 * D + 16);
    uint32_t* codes = reinterpret_cast<uint32_t*>(packed);
#pragma unroll
    for (int k = 0; k < D / 16; ++k) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            codes[(k * 8 + e) * 32 + lane] = qh[k][e];
            codes[(k * 8 + 4 + e) * 32 + lane] = ql[k][e];
        }
    }
    if (qd == 0) { packed[16 * D + g] = up0; packed[16 * D + g + 8] = up1; }
}

// A warp per query head; segment partials retain the original per-row key order.
extern "C" __global__ void __maxnreg__(144)
tf_kv4_prompt128(const float* prepared, const unsigned char* kc, const unsigned char* vc, float* scratch, int heads,
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
    const int r0 = wrow + g, r1 = r0 + 8;
    const int pos0 = r0 < rows ? p0 + r0 : 0, pos1 = r1 < rows ? p0 + r1 : 0;
    const float* packed = prepared + ((size_t)blockIdx.x * heads + head) * (16 * D + 16);
    // On-demand fragment loads and shared scores keep both resident blocks within their register budgets.
    const uint32_t* codes = reinterpret_cast<const uint32_t*>(packed);
    const float up0 = packed[16 * D + g], up1 = packed[16 * D + g + 8];
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
        volatile float* sc = vs + BN * 4 + warp * 32 * 8 * 4;
#pragma unroll 1
        for (int n = 0; n < 8; n += 2) {
#pragma unroll
            for (int gi = 0; gi < D / 32; ++gi) {
                float x0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, x1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int k = 2 * gi + h;
                    uint32_t b[4];
                    ldm(b, kt + sw(8 * (n + mt / 2) + lane % 8, 2 * k + mt % 2));
                    uint32_t qh[4], ql[4];
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        qh[e] = prompt_code(codes + (k * 8 + e) * 32 + lane, n);
                        ql[e] = prompt_code(codes + (k * 8 + 4 + e) * 32 + lane, n);
                    }
                    mma(x0, qh, b[0], b[1]);
                    mma(x0, ql, b[0], b[1]);
                    mma(x1, qh, b[2], b[3]);
                    mma(x1, ql, b[2], b[3]);
                }
                const float* k0 = ks + (8 * n + 2 * qd) * 4 + gi;
                const float* k1 = k0 + 8 * 4;
                const float f[4] = {k0[0], k0[4], k1[0], k1[4]};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float y0 = __fmul_rn(x0[e], f[e % 2]), y1 = __fmul_rn(x1[e], f[2 + e % 2]);
                    sc[((n) * 4 + (e)) * 32 + lane] = gi ? __fadd_rn(sc[((n) * 4 + (e)) * 32 + lane], y0) : y0;
                    sc[((n + 1) * 4 + (e)) * 32 + lane] = gi ? __fadd_rn(sc[((n + 1) * 4 + (e)) * 32 + lane], y1) : y1;
                }
            }
        }
        {
            const int kb0 = t * BN + 2 * qd;
            float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int c = kb0 + 8 * n;
                sc[((n) * 4 + (0)) * 32 + lane] = c >= 1 && c < pos0 ? __fmul_rn(sc[((n) * 4 + (0)) * 32 + lane], scale0) : -INFINITY;
                sc[((n) * 4 + (1)) * 32 + lane] = c + 1 < pos0 ? __fmul_rn(sc[((n) * 4 + (1)) * 32 + lane], scale0) : -INFINITY;
                sc[((n) * 4 + (2)) * 32 + lane] = c >= 1 && c < pos1 ? __fmul_rn(sc[((n) * 4 + (2)) * 32 + lane], scale1) : -INFINITY;
                sc[((n) * 4 + (3)) * 32 + lane] = c + 1 < pos1 ? __fmul_rn(sc[((n) * 4 + (3)) * 32 + lane], scale1) : -INFINITY;
                tm0 = fmaxf(tm0, fmaxf(sc[((n) * 4 + (0)) * 32 + lane], sc[((n) * 4 + (1)) * 32 + lane]));
                tm1 = fmaxf(tm1, fmaxf(sc[((n) * 4 + (2)) * 32 + lane], sc[((n) * 4 + (3)) * 32 + lane]));
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
                sc[((n) * 4 + (0)) * 32 + lane] = act0 && sc[((n) * 4 + (0)) * 32 + lane] != -INFINITY ? texp(__fsub_rn(sc[((n) * 4 + (0)) * 32 + lane], n0)) : 0.0f;
                sc[((n) * 4 + (1)) * 32 + lane] = act0 && sc[((n) * 4 + (1)) * 32 + lane] != -INFINITY ? texp(__fsub_rn(sc[((n) * 4 + (1)) * 32 + lane], n0)) : 0.0f;
                sc[((n) * 4 + (2)) * 32 + lane] = act1 && sc[((n) * 4 + (2)) * 32 + lane] != -INFINITY ? texp(__fsub_rn(sc[((n) * 4 + (2)) * 32 + lane], n1)) : 0.0f;
                sc[((n) * 4 + (3)) * 32 + lane] = act1 && sc[((n) * 4 + (3)) * 32 + lane] != -INFINITY ? texp(__fsub_rn(sc[((n) * 4 + (3)) * 32 + lane], n1)) : 0.0f;
                c0[n] = __fadd_rn(sc[((n) * 4 + (0)) * 32 + lane], sc[((n) * 4 + (1)) * 32 + lane]);
                c1[n] = __fadd_rn(sc[((n) * 4 + (2)) * 32 + lane], sc[((n) * 4 + (3)) * 32 + lane]);
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
#pragma unroll 1
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int gi = 0; gi < D / 32; ++gi) {
                const float* v = vs + (16 * j + 2 * qd) * 4 + gi;
                const float f0 = v[0], f1 = v[4], f2 = v[32], f3 = v[36];
                uint32_t ah[4], al[4];
                split(weigh(sc[((2 * j) * 4 + (0)) * 32 + lane], f0), weigh(sc[((2 * j) * 4 + (1)) * 32 + lane], f1), ah[0], al[0]);
                split(weigh(sc[((2 * j) * 4 + (2)) * 32 + lane], f0), weigh(sc[((2 * j) * 4 + (3)) * 32 + lane], f1), ah[1], al[1]);
                split(weigh(sc[((2 * j + 1) * 4 + (0)) * 32 + lane], f2), weigh(sc[((2 * j + 1) * 4 + (1)) * 32 + lane], f3), ah[2], al[2]);
                split(weigh(sc[((2 * j + 1) * 4 + (2)) * 32 + lane], f2), weigh(sc[((2 * j + 1) * 4 + (3)) * 32 + lane], f3), ah[3], al[3]);
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
