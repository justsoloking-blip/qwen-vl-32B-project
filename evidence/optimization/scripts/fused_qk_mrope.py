"""Experimental Qwen3-VL BF16 Q/K RMSNorm + interleaved MRoPE.
Narrow contract: head_dim=rotary_dim=128, Neox layout, sections [24,20,20].
Outputs remain separate Q/K tensors; this does not fuse the KV cache write.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _qk_norm_mrope(Q,K,QW,KW,POS,CACHE,OQ,OK, N, NQ:tl.constexpr,NK:tl.constexpr,QS:tl.constexpr,KS:tl.constexpr, EPS:tl.constexpr):
    t=tl.program_id(0);h=tl.program_id(1);d=tl.arange(0,64)
    if h<NQ:
        src=Q+t*QS+h*128;w=QW;dst=OQ+t*NQ*128+h*128
    else:
        src=K+t*KS+(h-NQ)*128;w=KW;dst=OK+t*NK*128+(h-NQ)*128
    x0=tl.load(src+d).to(tl.float32);x1=tl.load(src+d+64).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x0*x0+x1*x1,axis=0)/128+EPS)
    w0=tl.load(w+d).to(tl.float32);w1=tl.load(w+d+64).to(tl.float32)
    # Preserve the BF16 intermediate rounding of the original RMSNorm.
    a=(x0*inv*w0).to(Q.dtype.element_ty);b=(x1*inv*w1).to(Q.dtype.element_ty)
    ax=tl.where((d%3==1)&(d<=60),1,tl.where((d%3==2)&(d<=60),2,0))
    pos=tl.load(POS+ax*N+t)
    c=tl.load(CACHE+pos*128+d);s=tl.load(CACHE+pos*128+d+64)
    tl.store(dst+d,a*c-b*s);tl.store(dst+d+64,b*c+a*s)

def fused_qk_norm_mrope(q,k,qweight,kweight,positions,cache,eps=1e-6):
    oq=torch.empty(q.shape,device=q.device,dtype=q.dtype);ok=torch.empty(k.shape,device=k.device,dtype=k.dtype)
    _qk_norm_mrope[(q.shape[0],q.shape[1]//128+k.shape[1]//128)](q,k,qweight,kweight,positions,cache,oq,ok,q.shape[0],q.shape[1]//128,k.shape[1]//128,q.stride(0),k.stride(0),eps,num_warps=4,enable_fp_fusion=False)
    return oq,ok
