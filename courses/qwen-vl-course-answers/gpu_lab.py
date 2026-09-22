"""Fresh GPU execution runner. Must be run on an isolated Linux CUDA GPU allocation.
--plan generates and checks candidate configurations without starting a model.
Full execution results are separate from local/archived checks.
"""
from pathlib import Path
import argparse,copy,datetime,json,os,shutil,socket,subprocess,sys,time,urllib.request
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT/'common/project'
BASE=json.loads((PROJECT/'versions/dp-attention-fixed/config.json').read_text())
def config(label,model,gpus,port):
 c=copy.deepcopy(BASE);c['gpus']=gpus;c['base_url']=f'http://127.0.0.1:{port}/v1';a=c['arguments'];a[a.index('--model-path')+1]=model;a[a.index('--port')+1]=str(port);a[a.index('--host')+1]='127.0.0.1'
 if label!='A_dp2':
  i=a.index('--dp-size');del a[i:i+2];a.remove('--enable-dp-attention')
 if label in ['C_compile','E_compile_fusion','F_keep_device','G_ipc','H_final']:a+=['--enable-torch-compile','--torch-compile-max-bs','8']
 if label in ['D_fusion','E_compile_fusion','H_final']:a+=['--enable-flashinfer-allreduce-fusion']
 if label in ['F_keep_device','G_ipc','H_final']:a+=['--keep-mm-feature-on-device']
 c['env']={'SGLANG_USE_CUDA_IPC_TRANSPORT':'1','SGLANG_MM_FEATURE_CACHE_MB':'1024'} if label in ['G_ipc','H_final'] else {}
 return c
STAGES={'baseline':['A_dp2'],'fix':['A_dp2'],'ablation':['A_dp2','B_tp2','C_compile','D_fusion','E_compile_fusion','F_keep_device','G_ipc'],'fusion':['B_tp2','Q_qk'],'final':['H_final']}
def wait_until_ready(base_url,key,timeout=1200,interval=5):
 deadline=time.monotonic()+timeout
 while True:
  try:
   request=urllib.request.Request(base_url.removesuffix('/v1')+'/health',headers={'Authorization':'Bearer '+key})
   with urllib.request.urlopen(request,timeout=min(5,timeout)) as r:
    if r.status==200:return
  except (OSError,TimeoutError):pass
  if time.monotonic()>deadline:raise TimeoutError('Model health did not become ready; inspect private server log')
  time.sleep(interval)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=STAGES,required=True);ap.add_argument('--model-path',default='/models/Qwen3-VL-32B-Instruct');ap.add_argument('--venv',type=Path,default=PROJECT/'.venv');ap.add_argument('--gpus',default='0,1');ap.add_argument('--port',type=int,default=30100);ap.add_argument('--plan',action='store_true');ap.add_argument('--labels',help='Comma-separated subset of this stage, for resuming validated runs');ap.add_argument('--rounds',type=int,default=5);args=ap.parse_args()
 labels=args.labels.split(',') if args.labels else STAGES[args.stage]
 assert labels and set(labels)<=set(STAGES[args.stage]),'Labels must belong to selected stage'
 configs={label:config(label,args.model_path,args.gpus,args.port) for label in labels}
 assert all(x['arguments'][x['arguments'].index('--tp-size')+1]=='2' for x in configs.values())
 if args.plan:
  print(json.dumps(configs,indent=2));return
 assert sys.platform=='linux','Actual GPU run requires Linux'
 model=Path(args.model_path);assert (model/'model.safetensors.index.json').is_file()
 ix=json.loads((model/'model.safetensors.index.json').read_text());assert all((model/f).is_file() for f in set(ix['weight_map'].values()))
 python=args.venv.resolve()/'bin/python';assert python.is_file();assert args.rounds>=5
 with socket.socket() as sock:
  sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
  sock.bind(('127.0.0.1',args.port))
 subprocess.run([str(python),'-c','import torch; assert torch.cuda.is_available(); print(torch.__version__)'],check=True)
 run=ROOT/'gpu_runs'/datetime.datetime.now().strftime('%Y%m%d-%H%M%S');run.mkdir(parents=True)
 def call(cmd,env=None,timeout=1200,name='step',allow_failure=False):
  with (run/f'{name}.log').open('w') as f:r=subprocess.run([str(x) for x in cmd],env=env,cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
  if r.returncode and not allow_failure:raise subprocess.CalledProcessError(r.returncode,cmd)
  return r.returncode
 status={'stage':args.stage,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':False,'completed':[],'quality_failures':[],'scope':'Fresh Linux CUDA execution'}
 try:
  for label,c in configs.items():
   target=run/label
   call([python,PROJECT/'scripts/prepare_runtime.py','--version','optimized','--model-path',model,'--venv',args.venv.resolve(),'--gpus',args.gpus,'--port',str(args.port),'--runtime-root',target],name=label+'-prepare')
   runtime=target/'optimized'
   # A randomly generated key may begin with '-'; bind it to the option explicitly.
   service=runtime/'service.py';source=service.read_text();source=source.replace('"--api-key", (BASE / "api.key").read_text().strip()', '"--api-key=" + (BASE / "api.key").read_text().strip()');service.write_text(source)
   (runtime/'config.json').write_text(json.dumps(c,indent=2));env=os.environ.copy();env.update(MODEL_RUNTIME=str(runtime),MODEL_API_URL=f'http://127.0.0.1:{args.port}',PYTHONPATH=str(runtime/'overlay'))
   if label=='Q_qk':
    shutil.copy2(PROJECT/'experiments/fused_qk_mrope.py',runtime/'overlay/sglang/srt/layers/fused_qk_mrope_local.py')
    with (PROJECT/'patches/qk-fusion.patch').open('rb') as f:subprocess.run(['patch','--dry-run','-p0'],cwd=runtime/'overlay',stdin=f,check=True)
    with (PROJECT/'patches/qk-fusion.patch').open('rb') as f:subprocess.run(['patch','-p0'],cwd=runtime/'overlay',stdin=f,check=True)
   if args.stage=='fix':call(['bash',PROJECT/'scripts/run_regressions.sh'],env,name=label+'-regressions')
   try:
    call([python,runtime/'service.py','start'],env,name=label+'-start')
    wait_until_ready(c['base_url'],(runtime/'api.key').read_text().strip())
    call([python,PROJECT/'tests/smoke_test.py'],env,name=label+'-smoke')
    if args.stage in ['fix','fusion','final']:
     code=call([python,PROJECT/'tests/test_dp_integration.py'],env,name=label+'-integration',allow_failure=label=='Q_qk')
     if code:status['quality_failures'].append({'label':label,'check':'integration','returncode':code})
     code=call([python,PROJECT/'benchmarks/quality_compare.py',args.port,label],env,name=label+'-quality',allow_failure=label=='Q_qk')
     if code:status['quality_failures'].append({'label':label,'check':'quality','returncode':code})
    if args.stage in ['fusion','final']:
     code=call([python,ROOT/'evaluate_course_data.py','--runtime',runtime,'--port',args.port,'--output',runtime/'optimization/course-quality.json'],env,name=label+'-course-quality',allow_failure=label in ['Q_qk','H_final'])
     if code:status['quality_failures'].append({'label':label,'check':'course-quality','returncode':code})
    call([python,PROJECT/'benchmarks/opt_probe.py','--port',args.port,'--label',label,'--rounds',args.rounds],env,name=label+'-benchmark')
    if args.stage=='ablation' and label in ['B_tp2','E_compile_fusion','G_ipc']:
     call([python,PROJECT/'benchmarks/opt_probe.py','--port',args.port,'--label',label,'--profile','--cases','short,image'],env,name=label+'-profile')
    if label=='Q_qk':call([python,PROJECT/'experiments/test_fused_qk_mrope.py'],env,name=label+'-microbenchmark')
    status['completed'].append(label)
   finally:call([python,runtime/'service.py','stop'],env,name=label+'-stop')
  status['passed']=not status['quality_failures']
  status['execution_completed']=True
 finally:
  status['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(run/'gpu-verification.json').write_text(json.dumps(status,indent=2))
 print('GPU verification output:',run)
if __name__=='__main__':main()
