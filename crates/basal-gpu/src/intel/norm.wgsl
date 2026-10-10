enable f16;
struct Params { shape: vec4<u32>, extra: vec4<u32> }
@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(1) var<storage, read> w: array<f16>;
@group(0) @binding(2) var<storage, read_write> y: array<f16>;
@group(0) @binding(3) var<uniform> p: Params;
var<workgroup> sums: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let h = p.shape.y; let row = g.x;
    var sum = 0f;
    for(var d=lane; d<h; d+=128u) { let v=f32(x[row*h+d]); sum += v*v; }
    sums[lane]=sum; workgroupBarrier();
    for(var stride=64u; stride>0u; stride/=2u) { if(lane<stride) { sums[lane]+=sums[lane+stride]; } workgroupBarrier(); }
    let scale=inverseSqrt(sums[0]/f32(h)+bitcast<f32>(p.shape.z));
    for(var d=lane; d<h; d+=128u) { y[row*h+d]=f16(f32(x[row*h+d])*scale*f32(w[d])); }
}
