"""Execute unchanged method bodies extracted by AST with real CPU PyTorch.
SGLang global initialization and GPU collectives are NOT exercised.
"""
from pathlib import Path
import ast,copy,json,sys,types
from unittest.mock import MagicMock
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from course_utils import *
def source_class(version,relative,classname,names,namespace,bases=None):
 file=PROJECT/'versions'/version/'src/sglang/srt'/relative
 tree=ast.parse(file.read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==classname)
 methods=[copy.deepcopy(n) for n in cl.body if isinstance(n,ast.FunctionDef) and n.name in names]
 assert len(methods)==len(names)
 c=ast.ClassDef(name=classname,bases=bases or [],keywords=[],body=methods,decorator_list=[])
 mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),c],type_ignores=[])
 exec(compile(ast.fix_missing_locations(mod),str(file),'exec'),namespace)
 return namespace[classname]
def pad_case(C,n,target,bs,padded_bs):
 b=object.__new__(C);b.input_ids=torch.arange(n);b.req_pool_indices=torch.arange(bs);b.lora_ids=[None]*bs
 b.seq_lens=torch.full((bs,),max(n,1));b.seq_lens_cpu=b.seq_lens.clone();b.seq_lens_sum=int(b.seq_lens.sum());b.out_cache_loc=torch.arange(n)
 b.encoder_lens=None;b.positions=torch.arange(n);original=torch.arange(3*n).reshape(3,n);b.mrope_positions=original.clone();b.extend_seq_lens=None;b.spec_info=None
 runner=types.SimpleNamespace(attn_backend=types.SimpleNamespace(get_cuda_graph_seq_len_fill_value=lambda:1))
 b._pad_inputs_to_size(runner,target,padded_bs)
 assert b.mrope_positions.shape==(3,target)
 torch.testing.assert_close(b.mrope_positions[:,:n],original);assert (b.mrope_positions[:,n:]==0).all()
 assert b.mrope_positions.dtype==original.dtype
cases=[(78,78,1,1),(5,8,2,2),(1,4,1,4),(0,4,0,4),(7,10,3,3),(7,7,3,3),(78,128,1,1)]
results=[]
for version in ['original','dp-attention-fixed']:
 C=source_class(version,'model_executor/forward_batch_info.py','ForwardBatch',['_pad_tensor_to_size','_pad_inputs_to_size'],{'torch':torch})
 for case in cases:
  try:pad_case(C,*case);passed=True;error=None
  except (RuntimeError,AssertionError) as e:passed=False;error=type(e).__name__
  # original happens to work with exactly three requests and no token padding.
  expected=version!='original' or case==(7,7,3,3)
  assert passed==expected,(version,case,passed)
  results.append({'version':version,'case':case,'invariant_passed':passed,'expected':expected,'error':error})
 vision_calls=[]
 for dp in [False,True]:
  for modality in ['image','video']:
   local=torch.full((2,4),3.);gathered=torch.full((2,4),7.)
   visual=MagicMock(return_value=local);visual.dtype=torch.float32
   sharded=MagicMock(return_value=gathered)
   ns={'torch':torch,'is_dp_attention_enabled':lambda:dp,'run_dp_sharded_mrope_vision_model':sharded}
   C=source_class(version,'models/qwen3_vl.py','Qwen3VLForConditionalGeneration',['get_image_feature','get_video_feature'],ns)
   instance=types.SimpleNamespace(visual=visual,use_data_parallel=True)
   item=types.SimpleNamespace(feature=torch.ones(2,8),image_grid_thw=torch.tensor([[1,2,2]]),video_grid_thw=torch.tensor([[2,2,2]]))
   actual=getattr(C,f'get_{modality}_feature')(instance,[item])
   expected_local=version!='original' and dp
   assert (actual is local)==expected_local
   assert visual.call_count==int(expected_local) and sharded.call_count==int(not expected_local)
   vision_calls.append({'dp_attention':dp,'modality':modality,'local':expected_local,'collectives_mocked':True})
 # Execute real constructor body with expensive model components replaced by mocks.
 for dp in [False,True]:
  language=MagicMock();vision=MagicMock();cfg=types.SimpleNamespace(vision_config=types.SimpleNamespace(),tie_word_embeddings=False,vocab_size=16,hidden_size=4,rope_scaling={'mrope_section':[24,20,20]})
  ns={'torch':torch,'nn':torch.nn,'Qwen3LLMModel':language,'Qwen3VLMoeVisionModel':vision,'is_dp_attention_enabled':lambda:dp,'get_global_server_args':lambda:types.SimpleNamespace(mm_enable_dp_encoder=False),'add_prefix':lambda a,b:a,'ParallelLMHead':MagicMock(),'LogitsProcessor':MagicMock(),'Pooler':MagicMock(),'PoolingType':types.SimpleNamespace(LAST=1),'Modality':types.SimpleNamespace(IMAGE=1,VIDEO=2)}
  C=source_class(version,'models/qwen3_vl.py','Qwen3VLForConditionalGeneration',['__init__'],ns,[ast.Attribute(value=ast.Name(id='nn',ctx=ast.Load()),attr='Module',ctx=ast.Load())])
  obj=C(cfg);assert bool(vision.call_args.kwargs['use_data_parallel'])==(dp and version!='original')
 results.append({'version':version,'vision_dispatch':vision_calls,'vision_constructor_checks':2})
record('02_dp_fix','Real source method bodies via AST + CPU torch; vision/collectives mocked; not SGLang integration.',{'torch':torch.__version__,'padding_checks':14,'vision_dispatch_checks':8,'constructor_checks':4,'results':results})
print('PASS lesson 2: 14 padding + 8 dispatch + 4 constructor checks, expected failures verified')
