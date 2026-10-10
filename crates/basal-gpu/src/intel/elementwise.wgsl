enable f16;
struct Params { shape: vec4<u32>, extra: vec4<u32> }
@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(1) var<storage, read> bias: array<f16>;
@group(0) @binding(2) var<storage, read_write> y: array<f16>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n=p.shape.x; let h=p.shape.y; let i=id.x + id.y * 65535u * 128u;
    if(i>=n*h) { return; }
    if(p.shape.z==0u) { y[i]=y[i]+(x[i]+bias[i%h]); }
    else {
        let row=i/h; let col=i%h;
        let gate=f32(x[row*2u*h+col]+bias[col]);
        let up=f32(x[row*2u*h+h+col]+bias[h+col]);
        y[i]=f16(f32(f16(gate/(1f+exp(-gate))))*up);
    }
}
