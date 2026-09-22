from pathlib import Path
import datetime,json,hashlib,statistics
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT/'common/project'
def save(folder,name,value):
 p=ROOT/folder/'results'/name;p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(value,ensure_ascii=False,indent=2));return p

def summary(path):
 d=json.loads(Path(path).read_text());assert d['profiled'] is False
 out={}
 for case in ['short','long','image','concurrent']:
  batches=[r for r in d['results'] if r['case']==case]
  if not batches:continue
  req=[v for b in batches for v in b['requests']]
  assert all(r['usage']['completion_tokens']>1 for r in req)
  tp=[(r['elapsed_s']*1000-r['ttft_ms'])/(r['usage']['completion_tokens']-1) for r in req]
  assert all(x>=0 for x in tp)
  wall=sum(b['wall_s'] for b in batches);assert wall>0
  out[case]={'requests':len(req),'rounds':len(batches),'ttft_ms':statistics.median(r['ttft_ms'] for r in req),'tpot_ms':statistics.median(tp),'output_tokens_per_s':sum(r['usage']['completion_tokens'] for r in req)/wall}
 return out

def record(folder,scope,details):
 return save(folder,'verification.json',{'passed':True,'scope':scope,'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'details':details})
