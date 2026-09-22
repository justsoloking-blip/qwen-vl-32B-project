"""Reproduce a rejected experimental kernel's position-stride boundary bug.
Same logical positions in contiguous and sliced storage must give equal results.
This diagnostic expects the archived experimental implementation to violate it.
"""
from pathlib import Path
import importlib.util,json,sys
import torch
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'common/project/experiments/fused_qk_mrope.py'
spec=importlib.util.spec_from_file_location('stride_candidate',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
torch.manual_seed(731);results=[]
for n in [1,4,8]:
 packed=torch.randn(n,5120,device='cuda',dtype=torch.bfloat16);q,k,_=packed.split([4096,512,512],-1)
 w=torch.ones(128,device='cuda',dtype=torch.bfloat16)
 storage=torch.randint(0,100,(3,17),device='cuda');pos=storage[:,:n]
 cache=torch.randn(100,128,device='cuda',dtype=torch.bfloat16)
 a=m.fused_qk_norm_mrope(q,k,w,w,pos,cache);b=m.fused_qk_norm_mrope(q,k,w,w,pos.contiguous(),cache)
 diff=max((x-y).abs().max().item() for x,y in zip(a,b));assert diff>0.03125
 results.append({'tokens':n,'position_stride':list(pos.stride()),'contiguous_position_stride':list(pos.contiguous().stride()),'max_abs_difference':diff,'layout_invariance_passed':False})
report={'known_boundary_regression_reproduced':True,'candidate_accepted':False,'scope':'Same values, different positions storage strides. This diagnoses a kernel defect, not a proof that every service failure has this sole cause.','cases':results}
out=ROOT/'04_fusion/results/position-stride-gpu.json';out.write_text(json.dumps(report,indent=2));print(out.read_text())
