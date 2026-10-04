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
