enable f16;
struct Params { shape: vec4<u32>, extra: vec4<u32> }
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
// Per query: count, then up to eight (start,end) ancestor ranges, ending at this query.
@group(0) @binding(1) var<storage, read> ranges: array<u32>;
@group(0) @binding(2) var<storage, read> prefix: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f16>;
@group(0) @binding(4) var<uniform> p: Params;
fn kv(token: u32, column: u32, width: u32) -> f32 {
    if(token < p.extra.x) { return prefix[token * width + column]; }
    return qkv[(token - p.extra.x) * width + column];
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(subgroup_id) subgroup: u32,
    @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) size: u32) {
    let n=p.shape.x; let nh=p.shape.y; let nkv=p.shape.z; let hd=p.shape.w;
    let t=g.x*(64u/size)+subgroup; let head=g.y;
    if(t>=n) { return; }
    let width=(nh+2u*nkv)*hd; let kh=head/(nh/nkv);
    var query: array<f32,32>; var acc: array<f32,32>;
    for(var j=0u; j<hd/size; j++) { query[j]=qkv[t*width+head*hd+lane+j*size]; }
    var max_score=-3.402823e38f; var denom=0f;
    let count=ranges[t*17u];
    for(var r=0u; r<count; r++) {
        let start=ranges[t*17u+1u+2u*r]; let end=ranges[t*17u+2u+2u*r];
        for(var s=start; s<end; s++) {
            var dot=0f;
            for(var j=0u; j<hd/size; j++) { dot=fma(query[j],kv(s, nh*hd+kh*hd+lane+j*size, width),dot); }
            let score=subgroupAdd(dot)*inverseSqrt(f32(hd));
            let next_max=max(max_score,score); let alpha=exp(max_score-next_max); let beta=exp(score-next_max);
            denom=denom*alpha+beta; max_score=next_max;
            for(var j=0u; j<hd/size; j++) {
                acc[j]=acc[j]*alpha+beta*kv(s, (nh+nkv)*hd+kh*hd+lane+j*size, width);
            }
        }
    }
    for(var j=0u; j<hd/size; j++) { out[t*nh*hd+head*hd+lane+j*size]=f16(acc[j]/denom); }
}
