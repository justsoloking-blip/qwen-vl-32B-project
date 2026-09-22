"""Run the generated 40-case quality set against an existing local model service."""
from pathlib import Path
import argparse,base64,json,urllib.request
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--port',type=int,default=30100);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
key=(a.runtime/'api.key').read_text().strip();data=ROOT/'01_baseline/data';cases=json.loads((data/'manifest.json').read_text());out=[]
for c in cases:
 content=[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode((data/f).read_bytes()).decode()}} for f in c['files']]
 content.append({'type':'text','text':c['prompt']})
 payload={'model':'ieg-infra-qwen3-vl-32b-instruct','messages':[{'role':'user','content':content}],'temperature':0,'max_tokens':32}
 req=urllib.request.Request(f'http://127.0.0.1:{a.port}/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
 with urllib.request.urlopen(req,timeout=180) as r:answer=json.load(r)['choices'][0]['message']['content'].strip()
 out.append({'id':c['id'],'answer':answer,'expected':c['expected'],'exact_match':answer==c['expected']})
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'count':len(out),'passed':all(x['exact_match'] for x in out),'results':out},ensure_ascii=False,indent=2))
print(f"Exact matches: {sum(x['exact_match'] for x in out)}/{len(out)}")
raise SystemExit(0 if all(x['exact_match'] for x in out) else 1)
