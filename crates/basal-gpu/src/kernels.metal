// Fused element-wise kernels of the basal-1.0 forward. Rounding points follow the unfused bf16 graph (HF / MLX):
// a linear layer output is rounded to the storage type once its bias is added, activations are rounded per op.
// Exponentials use precise:: so fast-math settings do not change the result.
#include <metal_stdlib>
using namespace metal;

// out[m, i] = T(silu(T(g + bg)) ) * T(u + bu), with gu = [M, 2I] = [gate | up] (fused projection without bias)
template <typename T>
kernel void bias_silu_mul(device const T *gu [[buffer(0)]], device const T *bias [[buffer(1)]],
                          device T *out [[buffer(2)]], constant uint &I [[buffer(3)]],
                          constant uint &total [[buffer(4)]], uint id [[thread_position_in_grid]]) {
    if (id >= total) return;
    uint m = id / I, i = id - m * I;
    float g = float(T(float(gu[m * 2 * I + i]) + float(bias[i])));
    float u = float(T(float(gu[m * 2 * I + I + i]) + float(bias[I + i])));
    float s = float(T(g / (1.0f + precise::exp(-g))));
    out[id] = T(s * u);
}

// out = T(x + T(y + b)): residual add of a linear output whose bias is still to be added
template <typename T>
kernel void bias_residual(device const T *x [[buffer(0)]], device const T *y [[buffer(1)]],
                          device const T *b [[buffer(2)]], device T *out [[buffer(3)]],
                          constant uint &H [[buffer(4)]], constant uint &total [[buffer(5)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= total) return;
    out[id] = T(float(x[id]) + float(T(float(y[id]) + float(b[id % H]))));
}

// qkv = [B*L, (NH + 2*NKV) * HD] projection without bias. Adds the bias (rounded to T), applies rotate-half RoPE to q
// and k in f32 with cs = [2, B, L, HD/2] (cos, sin) and writes f32 heads:
//   out = [q: B, NH, L, HD | k: B, NKV, L, HD | v: B, NKV, L, HD]
template <typename T>
kernel void qkv_rope(device const T *qkv [[buffer(0)]], device const T *bias [[buffer(1)]],
                     device const float *cs [[buffer(2)]], device float *out [[buffer(3)]],
                     constant uint &B [[buffer(4)]], constant uint &L [[buffer(5)]],
                     constant uint &NH [[buffer(6)]], constant uint &NKV [[buffer(7)]],
                     constant uint &HD [[buffer(8)]], uint id [[thread_position_in_grid]]) {
    uint hd2 = HD / 2;
    uint heads = NH + 2 * NKV;
    if (id >= B * L * heads * hd2) return;
    uint d = id % hd2;
    uint t = id / hd2;
    uint j = t % heads;
    t /= heads;
    uint l = t % L;
    uint b = t / L;
    uint row = (b * L + l) * heads * HD;
    uint col = j * HD;
    float x1 = float(T(float(qkv[row + col + d]) + float(bias[col + d])));
    float x2 = float(T(float(qkv[row + col + d + hd2]) + float(bias[col + d + hd2])));
    float y1 = x1, y2 = x2;
    if (j < NH + NKV) {
        uint ci = (b * L + l) * hd2 + d;
        float c = cs[ci], s = cs[B * L * hd2 + ci];
        y1 = x1 * c - x2 * s;
        y2 = x2 * c + x1 * s;
    }
    uint base, hh, nh;
    if (j < NH) {
        base = 0; hh = j; nh = NH;
    } else if (j < NH + NKV) {
        base = B * NH * L * HD; hh = j - NH; nh = NKV;
    } else {
        base = B * (NH + NKV) * L * HD; hh = j - NH - NKV; nh = NKV;
    }
    uint o = base + ((b * nh + hh) * L + l) * HD + d;
    out[o] = y1;
    out[o + hd2] = y2;
}

// [B, NH, L, HD] f32 attention output -> [B*L, NH*HD] in T
template <typename T>
kernel void merge_heads(device const float *in [[buffer(0)]], device T *out [[buffer(1)]],
                        constant uint &B [[buffer(2)]], constant uint &L [[buffer(3)]],
                        constant uint &NH [[buffer(4)]], constant uint &HD [[buffer(5)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= B * L * NH * HD) return;
    uint d = id % HD;
    uint t = id / HD;
    uint h = t % NH;
    t /= NH;
    uint l = t % L;
    uint b = t / L;
    out[id] = T(in[((b * NH + h) * L + l) * HD + d]);
}

#define INSTANTIATE(T, S)                                                                                         \
    template [[host_name("bias_silu_mul_" #S)]] kernel void bias_silu_mul<T>(                                    \
        device const T *, device const T *, device T *, constant uint &, constant uint &, uint);                  \
    template [[host_name("bias_residual_" #S)]] kernel void bias_residual<T>(                                    \
        device const T *, device const T *, device const T *, device T *, constant uint &, constant uint &, uint); \
    template [[host_name("qkv_rope_" #S)]] kernel void qkv_rope<T>(                                              \
        device const T *, device const T *, device const float *, device float *, constant uint &,                \
        constant uint &, constant uint &, constant uint &, constant uint &, uint);                                \
    template [[host_name("merge_heads_" #S)]] kernel void merge_heads<T>(                                        \
        device const float *, device T *, constant uint &, constant uint &, constant uint &, constant uint &, uint);

INSTANTIATE(float, f32)
INSTANTIATE(half, f16)
INSTANTIATE(bfloat, bf16)

// Tree attention (f32, flash style) on simdgroup matrices, invariant to how questions are packed (the work units of
// attn_tree_f32 in kernels.cu): a unit is one block (node) of a packed row. Its keys are, in this order, the
// precomputed prefix (np keys, pk/pv = [NKV, np, HD]), the token ranges of its ancestor blocks from the root down and
// its own range (last range, causal). Keys are tiled by their ordinal in that sequence and every query row is
// computed from its own Q row and these tiles only, so its result does not depend on what else shares the row or the
// forward. qkv = the qkv_rope output over T tokens, out = [NH, T, HD]. Threadgroups: x = sum over units of
// ceil(rep * q_len / (8 * nsg)), y = NKV; nsg <= ATT_SG simdgroups of 8 query rows (any nsg gives the same result).
// Products and sums are f32 (8x8 simdgroup multiply-accumulate). Softmax works on the fragment elements of each lane:
// in an 8x8 fragment lane l holds row (l / 16) * 4 + (l / 2) % 4, columns (l / 8 % 2) * 4 + (l % 2) * 2 and + 1, so
// the lanes of one row differ in bits 0 and 3 (as in MLX's steel attention).
#include <metal_simdgroup_matrix>
#define ATT_HD 128
#define ATT_SG 8
#define ATT_BK 16
#define ATT_LD 132
#define ATT_HLD 68
#define TREE_MAXR 8
// masked score: finite, since the library is compiled with fast math (no infinities assumed)
#define ATT_NEG (-3.0e38f)

struct TreeUnit {
    uint np, q_off, q_len, tile0, nr, pad0, pad1, pad2;
    uint r_off[TREE_MAXR], r_len[TREE_MAXR];
};

// offset of the key / value row of ordinal j: in the [NKV, np, HD] prefix (j < np) or the [NKV, T, HD] current tokens
static inline ulong tree_kv_off(constant TreeUnit &U, uint g, uint T, uint j) {
    if (j < U.np) return ((ulong)g * U.np + j) * ATT_HD;
    uint jr = j - U.np, ri = 0;
    while (jr >= U.r_len[ri]) jr -= U.r_len[ri++];
    return ((ulong)g * T + U.r_off[ri] + jr) * ATT_HD;
}

kernel void attn_tree_f32(device const float *qkv [[buffer(0)]], device const float *pk [[buffer(1)]],
                          device const float *pv [[buffer(2)]], device float *out [[buffer(3)]],
                          constant TreeUnit *units [[buffer(4)]], constant uint &n_units [[buffer(5)]],
                          constant uint &T [[buffer(6)]], constant uint &NH [[buffer(7)]],
                          constant uint &NKV [[buffer(8)]], constant float &scale [[buffer(9)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint nsg [[simdgroups_per_threadgroup]]) {
    // half of the Q rows ([8 * nsg][ATT_HLD]), then the key and value tiles ([ATT_BK][ATT_LD] each)
    threadgroup float buf[8 * ATT_SG * ATT_HLD];
    threadgroup float *Kt = buf, *Vt = buf + ATT_BK * ATT_LD;
    const uint nthr = 32 * nsg, bq = 8 * nsg, g = tg.y, rep = NH / NKV;
    uint lo = 0, hi = n_units - 1;  // unit of this threadgroup: the last whose first tile is <= tg.x
    while (lo < hi) {
        const uint mid = (lo + hi + 1) / 2;
        if (units[mid].tile0 <= tg.x) lo = mid;
        else hi = mid - 1;
    }
    constant TreeUnit &U = units[lo];
    const uint rows = rep * U.q_len, r0 = (tg.x - U.tile0) * bq;
    uint lk = U.np;
    for (uint i = 0; i < U.nr; i++) lk += U.r_len[i];
    const uint base_own = lk - U.r_len[U.nr - 1];  // ordinal of the first key of the own block
    // keys after the last one visible to any row of this threadgroup contribute exactly 0 and are not visited
    uint vis_end = 0;
    for (uint r = r0; r < min(r0 + bq, rows); r++) vis_end = max(vis_end, base_own + r % U.q_len + 1);
    device const float *q = qkv, *k = qkv + (ulong)NH * T * ATT_HD, *v = k + (ulong)NKV * T * ATT_HD;
    simdgroup_float8x8 qf[ATT_HD / 8];
    for (uint h = 0; h < 2; h++) {
        for (uint i = tid; i < bq * ATT_HD / 8; i += nthr) {
            uint rr = i / (ATT_HD / 8), d4 = i % (ATT_HD / 8), r = r0 + rr;
            float4 x = float4(0.f);
            if (r < rows)
                x = ((device const float4 *)(q + ((ulong)(g * rep + r / U.q_len) * T + U.q_off + r % U.q_len) *
                                                     ATT_HD + h * ATT_HD / 2))[d4];
            ((threadgroup float4 *)(buf + rr * ATT_HLD))[d4] = x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < ATT_HD / 16; c++)
            simdgroup_load(qf[h * ATT_HD / 16 + c], buf + sg * 8 * ATT_HLD + c * 8, ATT_HLD);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_float8x8 of[ATT_HD / 8];
    for (uint c = 0; c < ATT_HD / 8; c++) of[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    // the lane's row of its simdgroup's 8 rows and its first column in each 8x8 fragment
    const uint fm = (lane / 16) * 4 + (lane / 2) % 4, fn = (lane / 8 % 2) * 4 + (lane % 2) * 2;
    const uint r = r0 + sg * 8 + fm;
    const bool ok = r < rows;
    const uint lim = ok ? base_own + r % U.q_len : 0;  // last visible key ordinal of the row
    float m = ATT_NEG, l = 0.f;
    for (uint j0 = 0; j0 < vis_end; j0 += ATT_BK) {
        for (uint i = tid; i < ATT_BK * ATT_HD / 4; i += nthr) {
            uint jj = i / (ATT_HD / 4), d4 = i % (ATT_HD / 4), j = j0 + jj;
            float4 x = float4(0.f), y = float4(0.f);
            if (j < lk) {
                const bool pre = j < U.np;
                const ulong o = tree_kv_off(U, g, T, j);
                x = ((device const float4 *)((pre ? pk : k) + o))[d4];
                y = ((device const float4 *)((pre ? pv : v) + o))[d4];
            }
            ((threadgroup float4 *)(Kt + jj * ATT_LD))[d4] = x;
            ((threadgroup float4 *)(Vt + jj * ATT_LD))[d4] = y;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 s[ATT_BK / 8];
        for (uint n = 0; n < ATT_BK / 8; n++) {
            simdgroup_float8x8 kt;
            s[n] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
            for (uint c = 0; c < ATT_HD / 8; c++) {
                simdgroup_load(kt, Kt + n * 8 * ATT_LD + c * 8, ATT_LD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s[n], qf[c], kt, s[n]);
            }
        }
        float x[ATT_BK / 4], mt = ATT_NEG;
        for (uint n = 0; n < ATT_BK / 8; n++)
            for (uint e = 0; e < 2; e++) {
                const float sv = s[n].thread_elements()[e];
                x[2 * n + e] = (ok && j0 + n * 8 + fn + e <= lim) ? sv * scale : ATT_NEG;
                mt = fmax(mt, x[2 * n + e]);
            }
        mt = fmax(mt, simd_shuffle_xor(mt, 1));
        mt = fmax(mt, simd_shuffle_xor(mt, 8));
        const float mn = fmax(m, mt);
        float rs = 0.f;
        for (uint n = 0; n < ATT_BK / 8; n++)
            for (uint e = 0; e < 2; e++) {
                const float p = mn <= ATT_NEG ? 0.f : precise::exp(x[2 * n + e] - mn);
                s[n].thread_elements()[e] = p;
                rs += p;
            }
        rs += simd_shuffle_xor(rs, 1);
        rs += simd_shuffle_xor(rs, 8);
        const float corr = m <= ATT_NEG ? 0.f : precise::exp(m - mn);
        l = l * corr + rs;
        m = mn;
        for (uint c = 0; c < ATT_HD / 8; c++) {
            of[c].thread_elements()[0] *= corr;
            of[c].thread_elements()[1] *= corr;
            simdgroup_float8x8 vf;
            for (uint n = 0; n < ATT_BK / 8; n++) {
                simdgroup_load(vf, Vt + n * 8 * ATT_LD + c * 8, ATT_LD);
                simdgroup_multiply_accumulate(of[c], s[n], vf, of[c]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // the tiles are read before the next ones are loaded
    }
    if (ok) {
        const float inv = 1.0f / l;
        device float *dst = out + ((ulong)(g * rep + r / U.q_len) * T + U.q_off + r % U.q_len) * ATT_HD + fn;
        for (uint c = 0; c < ATT_HD / 8; c++)
            *(device float2 *)(dst + c * 8) = float2(of[c].thread_elements()[0] * inv, of[c].thread_elements()[1] * inv);
    }
}
