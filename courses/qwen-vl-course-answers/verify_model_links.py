"""Verify official pinned model files. Does not download complete model weights."""
from pathlib import Path
import concurrent.futures,datetime,hashlib,json,struct,time,urllib.request
ROOT=Path(__file__).resolve().parent
REPO='Qwen/Qwen3-VL-32B-Instruct'
REV='0cfaf48183f594c314753d30a4c4974bc75f3ccb'
BASE=f'https://huggingface.co/{REPO}'
def fetch(url,method='GET',headers=None,limit=None):
 for attempt in range(3):
  try:
   req=urllib.request.Request(url,method=method,headers=headers or {})
   with urllib.request.urlopen(req,timeout=35) as r:
    body=r.read(limit) if limit else (b'' if method=='HEAD' else r.read())
    return r.status,dict(r.headers),body
  except Exception:
   if attempt==2:raise
   time.sleep(1)
api=f'https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true'
_,_,body=fetch(api);meta=json.loads(body);assert meta['sha']==REV
(ROOT/'common/model_metadata/repository.json').write_bytes(body)
rows=[]
for label,url in [('official',BASE),('pinned_files',f'{BASE}/tree/{REV}')]:
 status,_,_=fetch(url,limit=512);rows.append({'label':label,'url':url,'status':status})
def one(file):
 name=file['rfilename'];url=f'{BASE}/resolve/{REV}/{name}'
 status,headers,_=fetch(url,'HEAD')
 row={'file':name,'url':url,'status':status,'expected_bytes':file.get('size'),'lfs_sha256':file.get('lfs',{}).get('sha256')}
 if name.endswith('.safetensors'):
  code,h,b=fetch(url,headers={'Range':'bytes=0-15'},limit=16)
  assert len(b)==16 and code in (200,206)
  header_len=struct.unpack('<Q',b[:8])[0];assert 0<header_len<file['size']
  row.update(range_status=code,read_bytes=len(b),safetensors_header_bytes=header_len)
 elif name in ['config.json','preprocessor_config.json','video_preprocessor_config.json','tokenizer_config.json','model.safetensors.index.json','generation_config.json','chat_template.json']:
  _,_,b=fetch(url);json.loads(b);(ROOT/'common/model_metadata'/name).write_bytes(b)
  row.update(downloaded_bytes=len(b),download_sha256=hashlib.sha256(b).hexdigest())
 return row
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for row in pool.map(one,meta['siblings']):rows.append(row);print(row['file'],row['status'],flush=True)
idx=json.loads((ROOT/'common/model_metadata/model.safetensors.index.json').read_text());shards=set(idx['weight_map'].values());assert len(shards)==14
assert shards<={r.get('file') for r in rows}
result={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'revision':REV,'passed':True,'weight_shards':len(shards),'scope':'Official pages + all file HEAD + small JSON download/parse + 16 bytes per weight shard. Full weight download and model loading NOT tested by this script.','links':rows}
(ROOT/'verification/model-links.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('PASS:',len(rows),'links,14 shard headers')
