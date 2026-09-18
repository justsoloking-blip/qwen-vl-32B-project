"""Reproducible fixed-output, streaming performance probes; never persist secrets."""
import argparse,base64,concurrent.futures,io,json,os,statistics,time,uuid,urllib.request
from pathlib import Path
from PIL import Image,ImageDraw
B=Path('/deployment/qwen3-vl-32b')
KEY=(B/'api.key').read_text().strip()
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=30000);p.add_argument('--label',default='dp_attn');p.add_argument('--profile',action='store_true');p.add_argument('--cases',default='short,long,image,concurrent');p.add_argument('--rounds',type=int,default=3);p.add_argument('--concurrency',type=int,default=8);args=p.parse_args()
URL=f'http://127.0.0.1:{args.port}'
OUT=B/'optimization';OUT.mkdir(exist_ok=True)
def api(path,data=None):
 r=urllib.request.Request(URL+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+KEY})
 return opener.open(r,timeout=240)
def messages(case,seed):
 text=f'Trial {seed}.\n'+('这是一份关于推理系统性能的背景材料。我们需要分析模型的计算、内存访问、通信以及请求调度。\n'*(160 if case=='long' else 20))+'\n请详细分析这个系统的优化方法，持续回答。'
 if case=='image':
  im=Image.new('RGB',(1024,1024),'white');d=ImageDraw.Draw(im);d.rectangle((64,64,480,480),fill='red');d.ellipse((520,520,960,960),fill='blue');d.text((20,980),seed,fill='black');buf=io.BytesIO();im.save(buf,format='PNG')
  text=[{'type':'text','text':f'Trial {seed}. Describe this image and its geometric relationships in detail.'},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()}}]
 return [{'role':'user','content':text}]
def request(case,nout=128):
 seed=uuid.uuid4().hex;data={'model':'ieg-infra-qwen3-vl-32b-instruct','messages':messages(case,seed),'temperature':0,'max_tokens':nout,'ignore_eos':True,'stream':True,'stream_options':{'include_usage':True}}
 times=[];usage=None;t=time.perf_counter()
 with api('/v1/chat/completions',data) as r:
  for line in r:
   if not line.startswith(b'data: ') or line.strip()==b'data: [DONE]':continue
   x=json.loads(line[6:]);choices=x.get('choices',[])
   if choices and choices[0].get('delta',{}).get('content'):times.append(time.perf_counter())
   if x.get('usage'):usage=x['usage']
 end=time.perf_counter();assert usage and usage['completion_tokens']==nout,(usage,nout)
 return {'case':case,'elapsed_s':end-t,'ttft_ms':1000*(times[0]-t),'tpot_ms':1000*(end-times[0])/(nout-1),'chunks':len(times),'usage':usage,'chunk_gap_p50_ms':statistics.median([1000*(b-a) for a,b in zip(times,times[1:])])}
def batch(case,nout):
 concurrency=args.concurrency if case=='concurrent' else 1;beg=time.perf_counter()
 with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:results=list(pool.map(lambda _:request('short' if case=='concurrent' else case,nout),range(concurrency)))
 return {'case':case,'concurrency':concurrency,'wall_s':time.perf_counter()-beg,'requests':results}
def main():
 allresults=[]
 for case in args.cases.split(','):
  print('WARMUP',args.label,case,flush=True);batch(case,16);batch(case,16)
  if args.profile:
   directory=str(OUT/'traces'/args.label/case)
   print('START_PROFILE',directory,flush=True)
   with api('/start_profile',{'output_dir':directory,'activities':['CPU','GPU'],'with_stack':False,'record_shapes':True,'profile_prefix':case}) as r:r.read()
   try:r=batch(case,32);allresults.append(r)
   finally:
    with api('/stop_profile',{}) as response:print(response.read().decode().strip(),flush=True)
  else:
   for repeat in range(args.rounds):
    r=batch(case,128);r['repeat']=repeat;allresults.append(r);print(json.dumps(r),flush=True)
  (OUT/(args.label+('-profile' if args.profile else '-baseline')+'.json')).write_text(json.dumps({'label':args.label,'port':args.port,'profiled':args.profile,'results':allresults},indent=2))
 print('DONE',flush=True)

if __name__ == "__main__":main()
