enable f16;
// 32x32 output tile, 32-deep K tiles; every invocation accumulates 4x4 in FP32.
struct Params { shape: vec4<u32>, extra: vec4<u32> }
@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(1) var<storage, read> w: array<f16>;
@group(0) @binding(2) var<storage, read_write> y: array<f16>;
@group(0) @binding(3) var<uniform> p: Params;
var<workgroup> a: array<f16, 1024>;
var<workgroup> b: array<f16, 1024>;
@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let m = p.shape.x; let n = p.shape.y; let k = p.shape.z;
    let row = g.y * 32u + l.y * 4u; let col = g.x * 32u + l.x * 4u;
    let tid = l.y * 8u + l.x;
    var sums: array<vec4<f32>, 4>;
    for(var base = 0u; base < k; base += 32u) {
        for(var j = tid; j < 1024u; j += 64u) {
            let r = j / 32u; let c = j % 32u;
            var av = 0h; var bv = 0h;
            if(g.y * 32u + r < m && base + c < k) { av = x[(g.y * 32u + r) * k + base + c]; }
            if(g.x * 32u + r < n && base + c < k) { bv = w[(g.x * 32u + r) * k + base + c]; }
            a[j] = av; b[c * 32u + r] = bv;
        }
        workgroupBarrier();
        for(var z = 0u; z < 32u; z++) {
            let off = z * 32u + l.x * 4u;
            let bv = vec4<f32>(f32(b[off]), f32(b[off+1u]), f32(b[off+2u]), f32(b[off+3u]));
            for(var r = 0u; r < 4u; r++) { sums[r] = fma(vec4<f32>(f32(a[(l.y*4u+r)*32u+z])), bv, sums[r]); }
        }
        workgroupBarrier();
    }
    for(var r = 0u; r < 4u; r++) { for(var c = 0u; c < 4u; c++) {
        if(row+r < m && col+c < n) { y[(row+r)*n+col+c] = f16(sums[r][c]); }
    }}
}
