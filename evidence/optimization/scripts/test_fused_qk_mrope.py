import sys,json,torch,triton
from pathlib import Path
B=Path('/deployment/qwen3-vl-32b');sys.path.insert(0,str(B/'overlay'))
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from flashinfer.norm import rmsnorm
from fused_qk_mrope import fused_qk_norm_mrope
from sglang.srt.server_args import set_global_server_args_for_scheduler
from types import SimpleNamespace
set_global_server_args_for_scheduler(SimpleNamespace(rl_on_policy_target=None))
results=[];torch.manual_seed(731)
rope=MRotaryEmbedding(128,128,32768,5000000,True,torch.bfloat16,[24,20,20],True).cuda()
for nt in [1,4,8,32,128,551,1075,4050]:
 for nq,nk in [(32,4),(64,8)]:
  packed=torch.randn(nt,(nq+2*nk)*128,device='cuda',dtype=torch.bfloat16);q,k,v=packed.split([nq*128,nk*128,nk*128],-1);qw=torch.rand(128,device='cuda',dtype=torch.bfloat16)+0.5;kw=torch.rand_like(qw)+0.5;pos=torch.randint(0,32000,(3,nt),device='cuda')
  def ref():
   qr=rmsnorm(q.reshape(-1,128).contiguous(),qw,1e-6).reshape(nt,nq*128);kr=rmsnorm(k.reshape(-1,128).contiguous(),kw,1e-6).reshape(nt,nk*128);return rope.forward(pos,qr,kr)
  def new():return fused_qk_norm_mrope(q,k,qw,kw,pos,rope.cos_sin_cache,1e-6)
  expected=ref();actual=new();diff=max((a-b).abs().max().item() for a,b in zip(expected,actual));assert all(torch.allclose(a,b,atol=0.03125,rtol=0.015625) for a,b in zip(expected,actual)),(nt,nq,diff)
  match=sum((a==b).sum().item() for a,b in zip(expected,actual))/sum(a.numel() for a in actual)
  rt=triton.testing.do_bench_cudagraph(ref,rep=60);ntm=triton.testing.do_bench_cudagraph(new,rep=60)
  results.append({'tokens':nt,'q_heads':nq,'k_heads':nk,'max_abs_error':diff,'exact_fraction':match,'reference_us':rt*1000,'fused_us':ntm*1000})
  print(results[-1],flush=True)
(B/'optimization/fused-kernel-test.json').write_text(json.dumps(results,indent=2))
