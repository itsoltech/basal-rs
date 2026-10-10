enable f16;
struct Params { shape: vec4<u32>, extra: vec4<u32> }
@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
var<workgroup> sums: array<f32,128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let h=p.shape.z; var sum=0f;
    for(var d=lane; d<h; d+=128u) { sum=fma(f32(x[g.x*h+d]),w[g.y*h+d],sum); }
    sums[lane]=sum; workgroupBarrier();
    for(var s=64u;s>0u;s/=2u) { if(lane<s) { sums[lane]+=sums[lane+s]; } workgroupBarrier(); }
    if(lane==0u) { out[g.x*p.shape.y+g.y]=sums[0]; }
}
