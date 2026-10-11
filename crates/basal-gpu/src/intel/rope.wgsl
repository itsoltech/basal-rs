enable f16;
struct Params { shape: vec4<u32>, extra: vec4<u32> }
@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(1) var<storage, read> bias: array<f16>;
@group(0) @binding(2) var<storage, read> cs: array<f32>;
@group(0) @binding(3) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n=p.shape.x; let width=p.shape.y; let d=p.shape.z; let qk=p.shape.w;
    let i=id.x + id.y * 65535u * 128u; if(i>=n*width) { return; }
    let col=i%width; let t=i/width;
    // Match the decoder's half-precision projection and bias addition before FP32 RoPE.
    let v=f32(x[i]+bias[col]);
    if(col<qk) {
        let j=col%d; let half=d/2u; let other=select(i+half,i-half,j>=half);
        let ob=select(col+half,col-half,j>=half);
        let rotated=f32(x[other]+bias[ob])*select(-1f,1f,j>=half);
        qkv[i]=v*cs[t*d+j%half]+rotated*cs[t*d+half+j%half];
    } else { qkv[i]=v; }
}
