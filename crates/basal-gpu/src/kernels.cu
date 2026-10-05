// Fused element-wise kernels of the basal-1.0 forward, CUDA port of kernels.metal (same rounding points: a linear
// layer output is rounded to the storage type once its bias is added, activations are rounded per op). Built to PTX
// by build.rs with nvcc (no fast math), loaded through candle's custom module cache.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

// round through T
template <typename T> __device__ __forceinline__ float rt(float v) { return to_f<T>(from_f<T>(v)); }

// out[m, i] = T(silu(T(g + bg))) * T(u + bu), with gu = [M, 2I] = [gate | up] (fused projection without bias)
template <typename T>
__device__ void bias_silu_mul(const T *gu, const T *bias, T *out, unsigned I, unsigned total) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= total) return;
    unsigned m = id / I, i = id - m * I;
    float g = rt<T>(to_f(gu[(size_t)m * 2 * I + i]) + to_f(bias[i]));
    float u = rt<T>(to_f(gu[(size_t)m * 2 * I + I + i]) + to_f(bias[I + i]));
    float s = rt<T>(g / (1.0f + expf(-g)));
    out[id] = from_f<T>(s * u);
}

// out = T(x + T(y + b)): residual add of a linear output whose bias is still to be added
template <typename T>
__device__ void bias_residual(const T *x, const T *y, const T *b, T *out, unsigned H, unsigned total) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= total) return;
    out[id] = from_f<T>(to_f(x[id]) + rt<T>(to_f(y[id]) + to_f(b[id % H])));
}

// qkv = [B*L, (NH + 2*NKV) * HD] projection without bias. Adds the bias (rounded to T), applies rotate-half RoPE to q
// and k in f32 with cs = [2, B, L, HD/2] (cos, sin) and writes f32 heads:
//   out = [q: B, NH, L, HD | k: B, NKV, L, HD | v: B, NKV, L, HD]
template <typename T>
__device__ void qkv_rope(const T *qkv, const T *bias, const float *cs, float *out, unsigned B, unsigned L,
                         unsigned NH, unsigned NKV, unsigned HD) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned hd2 = HD / 2;
    unsigned heads = NH + 2 * NKV;
    if (id >= B * L * heads * hd2) return;
    unsigned d = id % hd2;
    unsigned t = id / hd2;
    unsigned j = t % heads;
    t /= heads;
    unsigned l = t % L;
    unsigned b = t / L;
    size_t row = (size_t)(b * L + l) * heads * HD;
    unsigned col = j * HD;
    float x1 = rt<T>(to_f(qkv[row + col + d]) + to_f(bias[col + d]));
    float x2 = rt<T>(to_f(qkv[row + col + d + hd2]) + to_f(bias[col + d + hd2]));
    float y1 = x1, y2 = x2;
    if (j < NH + NKV) {
        unsigned ci = (b * L + l) * hd2 + d;
        float c = cs[ci], s = cs[B * L * hd2 + ci];
        y1 = x1 * c - x2 * s;
        y2 = x2 * c + x1 * s;
    }
    size_t base;
    unsigned hh, nh;
    if (j < NH) {
        base = 0; hh = j; nh = NH;
    } else if (j < NH + NKV) {
        base = (size_t)B * NH * L * HD; hh = j - NH; nh = NKV;
    } else {
        base = (size_t)B * (NH + NKV) * L * HD; hh = j - NH - NKV; nh = NKV;
    }
    size_t o = base + ((size_t)(b * nh + hh) * L + l) * HD + d;
    out[o] = y1;
    out[o + hd2] = y2;
}

// [B, NH, L, HD] f32 attention output -> [B*L, NH*HD] in T
template <typename T>
__device__ void merge_heads(const float *in, T *out, unsigned B, unsigned L, unsigned NH, unsigned HD) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= B * L * NH * HD) return;
    unsigned d = id % HD;
    unsigned t = id / HD;
    unsigned h = t % NH;
    t /= NH;
    unsigned l = t % L;
    unsigned b = t / L;
    out[id] = from_f<T>(in[((size_t)(b * NH + h) * L + l) * HD + d]);
}

#define INSTANTIATE(T, S)                                                                                         \
    extern "C" __global__ void bias_silu_mul_##S(const T *gu, const T *bias, T *out, unsigned I, unsigned total) { \
        bias_silu_mul<T>(gu, bias, out, I, total);                                                                 \
    }                                                                                                              \
    extern "C" __global__ void bias_residual_##S(const T *x, const T *y, const T *b, T *out, unsigned H,           \
                                                 unsigned total) {                                                 \
        bias_residual<T>(x, y, b, out, H, total);                                                                  \
    }                                                                                                              \
    extern "C" __global__ void qkv_rope_##S(const T *qkv, const T *bias, const float *cs, float *out, unsigned B,  \
                                            unsigned L, unsigned NH, unsigned NKV, unsigned HD) {                  \
        qkv_rope<T>(qkv, bias, cs, out, B, L, NH, NKV, HD);                                                        \
    }                                                                                                              \
    extern "C" __global__ void merge_heads_##S(const float *in, T *out, unsigned B, unsigned L, unsigned NH,       \
                                               unsigned HD) {                                                      \
        merge_heads<T>(in, out, B, L, NH, HD);                                                                     \
    }

INSTANTIATE(float, f32)
INSTANTIATE(__half, f16)
INSTANTIATE(__nv_bfloat16, bf16)

// Attention softmax with the scale and the additive mask fused (f32): s = scores [R, LK] with R = heads * L rows in
// head-major order (row r is query position r % L), mask = [L, LK] shared by all heads. out = softmax(s * scale + mask)
// per row. One block per row.
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
template <bool MAX> __device__ float block_reduce(float v, float *red) {
    unsigned lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = MAX ? warp_max(v) : warp_sum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    v = lane < nw ? red[lane] : (MAX ? -INFINITY : 0.0f);
    return MAX ? warp_max(v) : warp_sum(v);
}

extern "C" __global__ void masked_softmax_f32(const float *s, const float *mask, float *out, unsigned L, unsigned LK,
                                              float scale) {
    __shared__ float red[32];
    size_t row = blockIdx.x;
    const float *x = s + row * LK;
    const float *m = mask + (size_t)(row % L) * LK;
    float *y = out + row * LK;
    float mx = -INFINITY;
    for (unsigned j = threadIdx.x; j < LK; j += blockDim.x) mx = fmaxf(mx, x[j] * scale + m[j]);
    mx = block_reduce<true>(mx, red);
    float sum = 0.0f;
    for (unsigned j = threadIdx.x; j < LK; j += blockDim.x) {
        float e = expf(x[j] * scale + m[j] - mx);
        y[j] = e;
        sum += e;
    }
    sum = block_reduce<false>(sum, red);
    float inv = 1.0f / sum;
    for (unsigned j = threadIdx.x; j < LK; j += blockDim.x) y[j] *= inv;
}

// Shared tiling of the f32 attention kernel below: ATT_BM query rows x ATT_BN keys per tile, 128 threads; each
// thread owns two query rows, computes 2 x 2 scores (keys cg, cg + 16) from float4 shared-memory reads and 2 x 8
// output columns (cg * 8 ..).
#define ATT_HD 128
#define ATT_BM 16
#define ATT_BN 32
#define ATT_THREADS 128
#define ATT_LD 132

// Letter readout: out[r, u] = sum_h hidden[r, h] * lm_head[ids[u], h] in f32, one warp per (r, u) with a fixed
// summation order (lane-strided partial sums, then a butterfly), so a row's logits do not depend on how many rows or
// letters share the call. hidden in T, lm_head in bf16 (checkpoint).
template <typename T>
__device__ void letter_logits(const T *hidden, const __nv_bfloat16 *lm, const unsigned *ids, float *out, unsigned R,
                              unsigned U, unsigned H) {
    unsigned warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (warp >= R * U) return;
    unsigned r = warp / U, u = warp % U;
    const T *h = hidden + (size_t)r * H;
    const __nv_bfloat16 *w = lm + (size_t)ids[u] * H;
    float s = 0.0f;
    for (unsigned i = lane; i < H; i += 32) s = fmaf(to_f(h[i]), __bfloat162float(w[i]), s);
    s = warp_sum(s);
    if (lane == 0) out[warp] = s;
}

#define INSTANTIATE_READOUT(T, S)                                                                                  \
    extern "C" __global__ void letter_logits_##S(const T *hidden, const __nv_bfloat16 *lm, const unsigned *ids,     \
                                                 float *out, unsigned R, unsigned U, unsigned H) {                  \
        letter_logits<T>(hidden, lm, ids, out, R, U, H);                                                            \
    }
INSTANTIATE_READOUT(float, f32)
INSTANTIATE_READOUT(__half, f16)
INSTANTIATE_READOUT(__nv_bfloat16, bf16)

// Tree attention (f32, flash style), invariant to how questions are packed: the work unit is one block (node) of a
// packed row. Its keys are, in this order, the precomputed prefix (np keys, pk/pv = [NKV, np, HD]), the token ranges
// of its ancestor blocks from the root down and its own range (last range, causal). Keys are tiled by their ordinal
// in that sequence, so a query's sums do not depend on what else shares the row or the forward: the same question
// asked alone, in a tree with other questions or with a cached state gives bitwise the same result. qkv / out as in
// attn_ragged_f32. Grid: x = sum over units of ceil(rep * q_len / ATT_BM), y = NKV.
#define TREE_MAXR 8
#define TREE_MAXU 128

// pk/pv: f32 prefix K/V [NKV, np, HD]; pkh/pvh: the same split into f16 hi / lo planes ([2][NKV, np, HD]), read by
// the tensor-core kernel
struct TreeUnit {
    unsigned long long pk, pv, pkh, pvh;
    unsigned np, q_off, q_len, tile0, nr, pad;
    unsigned r_off[TREE_MAXR], r_len[TREE_MAXR];
};
struct TreeTable {
    TreeUnit u[TREE_MAXU];
    unsigned n, pad;
};

// unit of a thread block: the last unit whose first tile is <= x (binary search over tile0)
__device__ __forceinline__ unsigned tree_unit_of(const TreeTable &tab, unsigned x) {
    unsigned lo = 0, hi = tab.n - 1;
    while (lo < hi) {
        const unsigned mid = (lo + hi + 1) / 2;
        if (tab.u[mid].tile0 <= x) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

extern "C" __global__ void __launch_bounds__(ATT_THREADS)
    attn_tree_f32(const float *qkv, float *out, TreeTable tab, unsigned T, unsigned NH, unsigned NKV, float scale) {
    __shared__ __align__(16) float Qs[ATT_BM * ATT_LD];
    __shared__ __align__(16) float Ks[ATT_BN * ATT_LD];  // keys, then the probabilities P [ATT_BM][ATT_BN + 1]
    __shared__ __align__(16) float Vs[ATT_BN * ATT_HD];
    float *Ps = Ks;
    const unsigned tid = threadIdx.x, g = blockIdx.y, rep = NH / NKV;
    const TreeUnit &U = tab.u[tree_unit_of(tab, blockIdx.x)];
    const unsigned rows = rep * U.q_len, r0 = (blockIdx.x - U.tile0) * ATT_BM;
    unsigned lk = U.np;
    for (unsigned i = 0; i < U.nr; i++) lk += U.r_len[i];
    const unsigned base_own = lk - U.r_len[U.nr - 1];  // ordinal of the first key of the own block
    const float *q = qkv, *k = qkv + (size_t)NH * T * ATT_HD, *v = k + (size_t)NKV * T * ATT_HD;
    const float *pk = (const float *)U.pk, *pv = (const float *)U.pv;
    for (unsigned i = tid; i < ATT_BM * ATT_HD / 4; i += ATT_THREADS) {
        unsigned rr = i / (ATT_HD / 4), d4 = i % (ATT_HD / 4), r = r0 + rr;
        float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
        if (r < rows)
            x = reinterpret_cast<const float4 *>(q + ((size_t)(g * rep + r / U.q_len) * T + U.q_off + r % U.q_len) *
                                                         ATT_HD)[d4];
        reinterpret_cast<float4 *>(Qs + rr * ATT_LD)[d4] = x;
    }
    const unsigned rg = tid / 16, cg = tid % 16, ra = 2 * rg;
    const unsigned rA = r0 + ra, rB = rA + 1;
    const bool ok[2] = {rA < rows, rB < rows};
    // last visible key ordinal of each of the thread's rows (causal within the own block)
    const unsigned lim[2] = {ok[0] ? base_own + rA % U.q_len : 0, ok[1] ? base_own + rB % U.q_len : 0};
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f}, acc[2][8];
#pragma unroll
    for (int c = 0; c < 8; c++) acc[0][c] = acc[1][c] = 0.f;
    for (unsigned j0 = 0; j0 < lk; j0 += ATT_BN) {
        // a tile beyond every row's last visible key contributes exactly 0 and is skipped
        bool vis = (ok[0] && j0 <= lim[0]) || (ok[1] && j0 <= lim[1]);
        if (!__syncthreads_or(vis)) break;
        for (unsigned i = tid; i < ATT_BN * ATT_HD / 4; i += ATT_THREADS) {
            unsigned jj = i / (ATT_HD / 4), d4 = i % (ATT_HD / 4), j = j0 + jj;
            float4 kk = make_float4(0.f, 0.f, 0.f, 0.f), vv = kk;
            if (j < U.np) {
                size_t o = ((size_t)g * U.np + j) * ATT_HD;
                kk = reinterpret_cast<const float4 *>(pk + o)[d4];
                vv = reinterpret_cast<const float4 *>(pv + o)[d4];
            } else if (j < lk) {
                unsigned jr = j - U.np, ri = 0;
                while (jr >= U.r_len[ri]) jr -= U.r_len[ri++];
                size_t o = ((size_t)g * T + U.r_off[ri] + jr) * ATT_HD;
                kk = reinterpret_cast<const float4 *>(k + o)[d4];
                vv = reinterpret_cast<const float4 *>(v + o)[d4];
            }
            reinterpret_cast<float4 *>(Ks + jj * ATT_LD)[d4] = kk;
            reinterpret_cast<float4 *>(Vs + jj * ATT_HD)[d4] = vv;
        }
        __syncthreads();
        float s00 = 0.f, s01 = 0.f, s10 = 0.f, s11 = 0.f;
        {
            const float4 *qa = reinterpret_cast<const float4 *>(Qs + ra * ATT_LD);
            const float4 *qb = reinterpret_cast<const float4 *>(Qs + (ra + 1) * ATT_LD);
            const float4 *k0 = reinterpret_cast<const float4 *>(Ks + cg * ATT_LD);
            const float4 *k1 = reinterpret_cast<const float4 *>(Ks + (cg + 16) * ATT_LD);
#pragma unroll 8
            for (int d4 = 0; d4 < ATT_HD / 4; d4++) {
                float4 a = qa[d4], b = qb[d4], x = k0[d4], y = k1[d4];
                s00 = fmaf(a.x, x.x, fmaf(a.y, x.y, fmaf(a.z, x.z, fmaf(a.w, x.w, s00))));
                s01 = fmaf(a.x, y.x, fmaf(a.y, y.y, fmaf(a.z, y.z, fmaf(a.w, y.w, s01))));
                s10 = fmaf(b.x, x.x, fmaf(b.y, x.y, fmaf(b.z, x.z, fmaf(b.w, x.w, s10))));
                s11 = fmaf(b.x, y.x, fmaf(b.y, y.y, fmaf(b.z, y.z, fmaf(b.w, y.w, s11))));
            }
        }
        const unsigned j0c = j0 + cg, j1c = j0 + cg + 16;
        float sc[2][2] = {{s00, s01}, {s10, s11}};
        float p[2][2];
#pragma unroll
        for (int a = 0; a < 2; a++) {
            float x0 = (ok[a] && j0c <= lim[a]) ? sc[a][0] * scale : -INFINITY;
            float x1 = (ok[a] && j1c <= lim[a]) ? sc[a][1] * scale : -INFINITY;
            float mt = fmaxf(x0, x1);
            for (int o = 8; o > 0; o >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, o));
            float mn = fmaxf(m[a], mt);
            p[a][0] = mn == -INFINITY ? 0.f : expf(x0 - mn);
            p[a][1] = mn == -INFINITY ? 0.f : expf(x1 - mn);
            float corr = m[a] == -INFINITY ? 0.f : expf(m[a] - mn);
            float rs = p[a][0] + p[a][1];
            for (int o = 8; o > 0; o >>= 1) rs += __shfl_xor_sync(0xffffffffu, rs, o);
            l[a] = l[a] * corr + rs;
#pragma unroll
            for (int c = 0; c < 8; c++) acc[a][c] *= corr;
            m[a] = mn;
        }
        __syncthreads();  // all scores read Ks; it now holds P
#pragma unroll
        for (int a = 0; a < 2; a++) {
            Ps[(ra + a) * (ATT_BN + 1) + cg] = p[a][0];
            Ps[(ra + a) * (ATT_BN + 1) + cg + 16] = p[a][1];
        }
        __syncthreads();
        const float *pa = Ps + ra * (ATT_BN + 1), *pb = pa + (ATT_BN + 1);
#pragma unroll 4
        for (int jj = 0; jj < ATT_BN; jj++) {
            float x = pa[jj], y = pb[jj];
            const float4 *vr = reinterpret_cast<const float4 *>(Vs + jj * ATT_HD + cg * 8);
            float4 v0 = vr[0], v1 = vr[1];
            float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
            for (int c = 0; c < 8; c++) {
                acc[0][c] = fmaf(x, vv[c], acc[0][c]);
                acc[1][c] = fmaf(y, vv[c], acc[1][c]);
            }
        }
    }
#pragma unroll
    for (int a = 0; a < 2; a++) {
        unsigned r = rA + a;
        if (r >= rows) continue;
        float inv = 1.0f / l[a];
        float4 *dst = reinterpret_cast<float4 *>(out + ((size_t)(g * rep + r / U.q_len) * T + U.q_off + r % U.q_len) *
                                                           ATT_HD +
                                                 cg * 8);
        dst[0] = make_float4(acc[a][0] * inv, acc[a][1] * inv, acc[a][2] * inv, acc[a][3] * inv);
        dst[1] = make_float4(acc[a][4] * inv, acc[a][5] * inv, acc[a][6] * inv, acc[a][7] * inv);
    }
}


// f32 -> two f16 planes: hi = f16(x), lo = f16(x - hi) (out = [hi: n | lo: n])
extern "C" __global__ void split_hilo_f32(const float *x, __half *out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    __half h = __float2half_rn(v);
    out[i] = h;
    out[n + i] = __float2half_rn(v - __half2float(h));
}

// Tree attention on tensor cores (FlashAttention-2 style, mma.sync m16n8k16, f32 accumulation), same units and
// result layout as attn_tree_f32. To keep f32 accuracy every operand is split into f16 hi + lo and each product is
// hi*hi + hi*lo + lo*hi (relative error ~2^-22). K and V come pre-split (split_hilo_f32 on the layer's K and V, and on
// the cached prefixes); with `v_exact` (f16 forward: V is an f16 value) the lo plane of V is zero and its products
// are skipped, which adds exactly nothing. Block: 4 warps, 64 query rows (16 per warp); keys in tiles of 32;
// fragments are read with ldmatrix (V transposed by ldmatrix.trans). Every row sees the same key tiles in the same
// order whatever else is packed, so results stay independent of batching.
#define TC_BM 128  // query rows per block: 8 warps x 16 (32 tokens x 4 heads of a K/V group)
#define TC_THREADS 256
#define TC_BN 32
#define TC_LD (ATT_HD + 8)

__device__ __forceinline__ void mma16816(float *c, const unsigned *a, const unsigned *b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void ldsm4(unsigned *r, const __half *p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

__device__ __forceinline__ void ldsm4t(unsigned *r, const __half *p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

__device__ __forceinline__ unsigned pack2(__half a, __half b) {
    __half2 v = __halves2half2(a, b);  // a: low 16 bits (lower index)
    return *reinterpret_cast<unsigned *>(&v);
}

// QK3: S = Qh·Kh + Qh·Kl + Ql·Kh (f16 hi/lo split, ~f32 products) or only Qh·Kh. PL: P·V with P split into
// hi/lo (two MMAs) or P rounded to f16 (one MMA). Softmax, accumulation and output stay f32 in every variant.
template <bool QK3, bool PL>
__device__ __forceinline__ void attn_tree_tc_body(const float *qkv, const __half *khl, const __half *vhl, float *out,
                                                  const TreeTable &tab, unsigned T, unsigned NH, unsigned NKV,
                                                  float scale, unsigned v_exact) {
    // Shared memory holds one Q plane (hi, then lo) until each warp has its Q fragments in registers, then the K / V
    // tiles: 34 KiB per block.
    extern __shared__ __align__(16) unsigned char smraw[];
    __half *Qs = reinterpret_cast<__half *>(smraw);
    __half *Kh = reinterpret_cast<__half *>(smraw), *Kl = Kh + TC_BN * TC_LD;
    __half *Vh = Kl + TC_BN * TC_LD, *Vl = Vh + TC_BN * TC_LD;  // V row-major [key][dim]
    const unsigned tid = threadIdx.x, w = tid / 32, lane = tid % 32, g = lane >> 2, t4 = lane & 3;
    const unsigned kvh = blockIdx.y, rep = NH / NKV;
    const TreeUnit &U = tab.u[tree_unit_of(tab, blockIdx.x)];
    const unsigned rows = rep * U.q_len, r0 = (blockIdx.x - U.tile0) * TC_BM;
    unsigned lk = U.np;
    for (unsigned i = 0; i < U.nr; i++) lk += U.r_len[i];
    const unsigned base_own = lk - U.r_len[U.nr - 1];
    const size_t nkv_all = (size_t)NKV * T * ATT_HD, nkv_past = (size_t)NKV * U.np * ATT_HD;
    const __half *pkh = (const __half *)U.pkh, *pvh = (const __half *)U.pvh;
    const float *q = qkv;
    // A fragments of the warp's 16 query rows for the 8 steps of 16 dims: plane 0 = f16(x), plane 1 = f16(x - hi)
    unsigned qh[ATT_HD / 16][4], ql[ATT_HD / 16][4];
#pragma unroll
    for (int plane = 0; plane < (QK3 ? 2 : 1); plane++) {
        for (unsigned i = tid; i < TC_BM * ATT_HD; i += TC_THREADS) {
            unsigned rr = i / ATT_HD, d = i % ATT_HD, r = r0 + rr;
            float x =
                r < rows ? q[((size_t)(kvh * rep + r / U.q_len) * T + U.q_off + r % U.q_len) * ATT_HD + d] : 0.f;
            __half h = __float2half_rn(x);
            Qs[rr * TC_LD + d] = plane == 0 ? h : __float2half_rn(x - __half2float(h));
        }
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < ATT_HD / 16; ks++) {
            const unsigned qo = (w * 16 + (lane % 16)) * TC_LD + ks * 16 + (lane / 16) * 8;
            ldsm4(plane == 0 ? qh[ks] : ql[ks], Qs + qo);
        }
        __syncthreads();  // the next plane / the K / V tiles reuse this memory
    }
    const unsigned rr2[2] = {w * 16 + g, w * 16 + g + 8};
    bool ok[2];
    unsigned lim[2];
    for (int a = 0; a < 2; a++) {
        unsigned r = r0 + rr2[a];
        ok[a] = r < rows;
        lim[a] = ok[a] ? base_own + r % U.q_len : 0;
    }
    float o[ATT_HD / 8][4];
#pragma unroll
    for (int i = 0; i < ATT_HD / 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    const unsigned planes = v_exact ? 1 : 2;
    for (unsigned j0 = 0; j0 < lk; j0 += TC_BN) {
        bool vis = (ok[0] && j0 <= lim[0]) || (ok[1] && j0 <= lim[1]);
        if (!__syncthreads_or(vis)) break;  // tiles are ordered: none after this one is visible either
        // 32 keys x 128 dims = 32 x 16 chunks of 8 halves (uint4) per plane
        for (unsigned i = tid; i < TC_BN * (ATT_HD / 8); i += TC_THREADS) {
            unsigned jj = i / (ATT_HD / 8), c8 = (i % (ATT_HD / 8)) * 8, j = j0 + jj;
            uint4 kh = make_uint4(0, 0, 0, 0), kl = kh, vh = kh, vl = kh;
            const __half *ks = nullptr, *vs = nullptr;
            size_t plane_k = 0;
            if (j < U.np) {
                size_t off = ((size_t)kvh * U.np + j) * ATT_HD + c8;
                ks = pkh + off;
                vs = pvh + off;
                plane_k = nkv_past;
            } else if (j < lk) {
                unsigned jr = j - U.np, ri = 0;
                while (jr >= U.r_len[ri]) jr -= U.r_len[ri++];
                size_t off = ((size_t)kvh * T + U.r_off[ri] + jr) * ATT_HD + c8;
                ks = khl + off;
                vs = vhl + off;
                plane_k = nkv_all;
            }
            if (ks) {
                kh = *reinterpret_cast<const uint4 *>(ks);
                if (QK3) kl = *reinterpret_cast<const uint4 *>(ks + plane_k);
                vh = *reinterpret_cast<const uint4 *>(vs);
                if (planes == 2) vl = *reinterpret_cast<const uint4 *>(vs + plane_k);
            }
            *reinterpret_cast<uint4 *>(Kh + jj * TC_LD + c8) = kh;
            if (QK3) *reinterpret_cast<uint4 *>(Kl + jj * TC_LD + c8) = kl;
            *reinterpret_cast<uint4 *>(Vh + jj * TC_LD + c8) = vh;
            if (planes == 2) *reinterpret_cast<uint4 *>(Vl + jj * TC_LD + c8) = vl;
        }
        __syncthreads();
        // S = Q K^T: the warp's 16 rows x 32 keys (4 n8 tiles), dims in 8 steps of 16
        float s[TC_BN / 8][4];
#pragma unroll
        for (int nt = 0; nt < TC_BN / 8; nt++) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < ATT_HD / 16; ks++) {
            const unsigned *ah = qh[ks], *al = ql[ks];
#pragma unroll
            for (int np2 = 0; np2 < TC_BN / 16; np2++) {
                // two n8 key tiles (2*np2, 2*np2+1) x two dim halves
                const unsigned ko = (np2 * 16 + (lane % 8) + (lane / 16) * 8) * TC_LD + ks * 16 + ((lane / 8) % 2) * 8;
                unsigned bh[4], bl[4];
                ldsm4(bh, Kh + ko);
                if (QK3) ldsm4(bl, Kl + ko);
#pragma unroll
                for (int h2 = 0; h2 < 2; h2++) {
                    float *acc = s[2 * np2 + h2];
                    mma16816(acc, ah, bh + 2 * h2);
                    if (QK3) {
                        mma16816(acc, ah, bl + 2 * h2);
                        mma16816(acc, al, bh + 2 * h2);
                    }
                }
            }
        }
        // online softmax: element c of tile nt is row (c >> 1 ? g + 8 : g), key nt*8 + 2*t4 + (c & 1)
        float p[TC_BN / 8][4];
#pragma unroll
        for (int a = 0; a < 2; a++) {
            float mt = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < TC_BN / 8; nt++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    unsigned j = j0 + nt * 8 + 2 * t4 + e;
                    float x = (ok[a] && j <= lim[a]) ? s[nt][2 * a + e] * scale : -INFINITY;
                    p[nt][2 * a + e] = x;
                    mt = fmaxf(mt, x);
                }
            mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 1));
            mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 2));
            float mn = fmaxf(m[a], mt);
            float corr = m[a] == -INFINITY ? 0.f : expf(m[a] - mn);
            float rs = 0.f;
#pragma unroll
            for (int nt = 0; nt < TC_BN / 8; nt++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    float y = mn == -INFINITY ? 0.f : expf(p[nt][2 * a + e] - mn);
                    p[nt][2 * a + e] = y;
                    rs += y;
                }
            rs += __shfl_xor_sync(0xffffffffu, rs, 1);
            rs += __shfl_xor_sync(0xffffffffu, rs, 2);
            l[a] = l[a] * corr + rs;
            m[a] = mn;
#pragma unroll
            for (int dt = 0; dt < ATT_HD / 8; dt++) {
                o[dt][2 * a] *= corr;
                o[dt][2 * a + 1] *= corr;
            }
        }
        // O += P V: P (16 x 32) as A in 2 k16 steps from the score fragments; V B-fragments via ldmatrix.trans
#pragma unroll
        for (int kk = 0; kk < TC_BN / 16; kk++) {
            __half h[8], lo[8];
            const float pv8[8] = {p[2 * kk][0], p[2 * kk][1], p[2 * kk][2], p[2 * kk][3],
                                  p[2 * kk + 1][0], p[2 * kk + 1][1], p[2 * kk + 1][2], p[2 * kk + 1][3]};
#pragma unroll
            for (int e = 0; e < 8; e++) {
                h[e] = __float2half_rn(pv8[e]);
                lo[e] = __float2half_rn(pv8[e] - __half2float(h[e]));
            }
            unsigned ph[4] = {pack2(h[0], h[1]), pack2(h[2], h[3]), pack2(h[4], h[5]), pack2(h[6], h[7])};
            unsigned pl[4] = {pack2(lo[0], lo[1]), pack2(lo[2], lo[3]), pack2(lo[4], lo[5]), pack2(lo[6], lo[7])};
#pragma unroll
            for (int dp = 0; dp < ATT_HD / 16; dp++) {
                // two n8 dim tiles (2*dp, 2*dp+1) x two key halves
                const unsigned vo = (kk * 16 + (lane % 8) + ((lane / 8) % 2) * 8) * TC_LD + dp * 16 + (lane / 16) * 8;
                unsigned bh[4];
                ldsm4t(bh, Vh + vo);
#pragma unroll
                for (int h2 = 0; h2 < 2; h2++) {
                    mma16816(o[2 * dp + h2], ph, bh + 2 * h2);
                    if (PL) mma16816(o[2 * dp + h2], pl, bh + 2 * h2);
                }
                if (planes == 2) {
                    unsigned bl[4];
                    ldsm4t(bl, Vl + vo);
#pragma unroll
                    for (int h2 = 0; h2 < 2; h2++) mma16816(o[2 * dp + h2], ph, bl + 2 * h2);
                }
            }
        }
    }
#pragma unroll
    for (int a = 0; a < 2; a++) {
        unsigned r = r0 + rr2[a];
        if (r >= rows) continue;
        float inv = 1.0f / l[a];
        float *dst = out + ((size_t)(kvh * rep + r / U.q_len) * T + U.q_off + r % U.q_len) * ATT_HD;
#pragma unroll
        for (int dt = 0; dt < ATT_HD / 8; dt++)
            *reinterpret_cast<float2 *>(dst + dt * 8 + 2 * t4) = make_float2(o[dt][2 * a] * inv, o[dt][2 * a + 1] * inv);
    }
}

#define ATTN_TC_ENTRY(name, qk3, pl)                                                                                 \
    extern "C" __global__ void __launch_bounds__(TC_THREADS)                                                       \
        name(const float *qkv, const __half *khl, const __half *vhl, float *out, TreeTable tab, unsigned T,          \
             unsigned NH, unsigned NKV, float scale, unsigned v_exact) {                                           \
        attn_tree_tc_body<qk3, pl>(qkv, khl, vhl, out, tab, T, NH, NKV, scale, v_exact);                           \
    }
ATTN_TC_ENTRY(attn_tree_tc, true, true)
ATTN_TC_ENTRY(attn_tree_tc_pv1, true, false)
ATTN_TC_ENTRY(attn_tree_tc_qk1, false, true)
ATTN_TC_ENTRY(attn_tree_tc_f16, false, false)
