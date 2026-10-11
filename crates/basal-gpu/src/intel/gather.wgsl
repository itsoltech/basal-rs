enable f16;
struct Params { shape: vec4<u32>, extra: vec4<u32> }
@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f16>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x + id.y * 65535u * 128u; let h=p.shape.y; if(i<p.shape.x*h) { y[i]=x[indices[i/h]*h+i%h]; }
}
