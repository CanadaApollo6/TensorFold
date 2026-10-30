#pragma once
#include "logprob_rows.cuh"

struct DraftRow {
    unsigned ids[16];
    float values[16];
    double log_mass;
    unsigned invalid, pad;
};
static_assert(sizeof(DraftRow) == 144);

__device__ void draft_lattice(const __nv_bfloat16* L, unsigned V, unsigned k, const float* PV,
                             const unsigned* PI, unsigned n, DraftRow* output) {
    const unsigned t = threadIdx.x;
    DraftRow* out = output + blockIdx.x;
    const size_t base = static_cast<size_t>(blockIdx.x) * n;
    float pv = INFINITY;
    unsigned pi = 0;
    for (unsigned r = 0; r < k; ++r) {
        tf_logprob::topk_best(PV + base, PI + base, 0, n, pv, pi, pv, pi);
        if (t == 0) {
            out->ids[r] = pi;
            out->values[r] = pv;
        }
    }
    __syncthreads();
    const float maximum = out->values[0];
    const __nv_bfloat16* row = L + static_cast<size_t>(blockIdx.x) * V;
    double sum = 0.0;
    bool invalid = false;
    for (unsigned i = t; i < V; i += 256) {
        const float v = __bfloat162float(row[i]);
        invalid |= !isfinite(v);
        sum += exp(static_cast<double>(__fsub_rn(v, maximum)));
    }
    const int bad = __syncthreads_or(invalid);
    __shared__ double partial[256];
    partial[t] = sum;
    __syncthreads();
    for (unsigned step = 128; step > 0; step >>= 1) {
        if (t < step) partial[t] += partial[t + step];
        __syncthreads();
    }
    if (t == 0) {
        out->log_mass = log(partial[0]);
        out->invalid = bad;
        out->pad = 0;
        for (unsigned r = k; r < 16; ++r) {
            out->ids[r] = 0;
            out->values[r] = 0;
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) tf_draft_lattice(const __nv_bfloat16* L, unsigned V, unsigned k,
    const float* PV, const unsigned* PI, unsigned n, DraftRow* output) {
    draft_lattice(L, V, k, PV, PI, n, output);
}
