"""CPU mathematical reference only. Does not execute the Triton CUDA kernel."""
from pathlib import Path
import ast,json,sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from course_utils import *
torch.set_num_threads(2);torch.manual_seed(731)
# Oracle constructs the interleaved layout directly, independent of the kernel predicate.
axis_oracle=torch.tensor([0,1,2]*20+[0,0,0,0]);d=torch.arange(64)
axis_kernel=torch.where((d%3==1)&(d<=60),1,torch.where((d%3==2)&(d<=60),2,0))
assert torch.equal(axis_oracle,axis_kernel);assert torch.bincount(axis_kernel).tolist()==[24,20,20]
cache=torch.randn(32000,128,dtype=torch.bfloat16)
def oracle(x,w,pos):
 shape=x.shape;x=x.reshape(shape[0],-1,128).float()
 norm=(x*torch.rsqrt((x*x).mean(-1,keepdim=True)+1e-6)*w.float()).to(torch.bfloat16).float()
 p=pos[axis_oracle,:].T;cos=cache[p,d].float().unsqueeze(1);sin=cache[p,d+64].float().unsqueeze(1)
 return torch.cat([norm[...,:64]*cos-norm[...,64:]*sin,norm[...,64:]*cos+norm[...,:64]*sin],-1).to(torch.bfloat16).reshape(shape)
def kernel_formula(x,w,pos):
 shape=x.shape;x=x.reshape(shape[0],-1,128);a=x[...,:64].float();b=x[...,64:].float()
 inv=torch.rsqrt((a*a+b*b).sum(-1,keepdim=True)/128+1e-6)
 a=(a*inv*w[:64].float()).to(torch.bfloat16).float();b=(b*inv*w[64:].float()).to(torch.bfloat16).float()
 p=pos[axis_kernel,:].T;cos=cache[p,d].float().unsqueeze(1);sin=cache[p,d+64].float().unsqueeze(1)
 return torch.cat([a*cos-b*sin,b*cos+a*sin],-1).to(torch.bfloat16).reshape(shape)
rows=[]
for n in [1,4,8,32,128,551,1075,4050]:
 for nq,nk in [(32,4),(64,8)]:
  packed=torch.randn(n,(nq+2*nk)*128,dtype=torch.bfloat16);q,k,_=packed.split([nq*128,nk*128,nk*128],-1)
  pos=torch.randint(0,32000,(3,n));weights=[torch.rand(128,dtype=torch.bfloat16)+.5 for _ in range(2)]
  errors=[]
  for x,w in zip([q,k],weights):
   ref=oracle(x,w,pos);actual=kernel_formula(x,w,pos)
   assert torch.allclose(ref,actual,atol=.03125,rtol=.015625)
   errors.append(float((ref-actual).abs().max()))
  rows.append({'tokens':n,'q_heads':nq,'k_heads':nk,'max_abs_error':max(errors),'input_q_stride':q.stride(0)})
source=PROJECT/'experiments/fused_qk_mrope.py';tree=ast.parse(source.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_qk_norm_mrope');n_arg=next(x for x in fn.args.args if x.arg=='N');assert n_arg.annotation is None
hist=json.loads((PROJECT/'benchmarks/results/fused-kernel-test.json').read_text());assert len(hist)==16
assert max(x['max_abs_error'] for x in hist)<=.03125
micro=[dict(x,local_speedup=x['reference_us']/x['fused_us']) for x in hist]
save('04_fusion','historical_microbench_recomputed.json',micro)
record('04_fusion','16 CPU mathematical-reference cases and archived microbenchmark arithmetic; actual Triton JIT/CUDA/model integration NOT rerun.',{'torch':torch.__version__,'axis_counts':[24,20,20],'cpu_cases':rows,'runtime_N_source_check':True,'historical_microbench_cases':16})
print('PASS lesson 4: 16 CPU reference cases and 16 historical GPU microbenchmark records')
