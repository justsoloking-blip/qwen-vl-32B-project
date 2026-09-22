"""CUDA-only experiment comparing runtime N and constexpr N, separate from serving."""
from pathlib import Path
import importlib.util,json,tempfile,time,sys
import torch
ROOT=Path(__file__).resolve().parents[1]
assert torch.cuda.is_available()
source=(ROOT/'common/project/experiments/fused_qk_mrope.py').read_text();results=[]
with tempfile.TemporaryDirectory() as td:
 for label,code in [('runtime_N',source),('constexpr_N',source.replace(' N, NQ:tl.constexpr',' N:tl.constexpr, NQ:tl.constexpr'))]:
  p=Path(td)/(label+'.py');p.write_text(code);spec=importlib.util.spec_from_file_location(label,p);mod=importlib.util.module_from_spec(spec);sys.modules[label]=mod;spec.loader.exec_module(mod)
  # Record compile-inclusive wall times for new and repeated lengths. No fixed timing assertions.
  for n in [551,552,553,551,552,553]:
   packed=torch.randn(n,5120,device='cuda',dtype=torch.bfloat16);q,k,v=packed.split([4096,512,512],-1);w=torch.ones(128,device='cuda',dtype=torch.bfloat16);pos=torch.randint(0,32000,(3,n),device='cuda');cache=torch.randn(32000,128,device='cuda',dtype=torch.bfloat16)
   torch.cuda.synchronize();start=time.perf_counter();oq,ok=mod.fused_qk_norm_mrope(q,k,w,w,pos,cache);torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-start)
   assert oq.shape==q.shape and ok.shape==k.shape and torch.isfinite(oq).all() and torch.isfinite(ok).all()
   results.append({'variant':label,'tokens':n,'compile_inclusive_wall_ms':elapsed})
(ROOT/'04_fusion/results/specialization-gpu.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
