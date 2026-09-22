import os
"""Small deterministic task accuracy check; does not replace a full quality evaluation."""
from pathlib import Path
import json,urllib.request,re,base64,io,concurrent.futures
from PIL import Image,ImageDraw,ImageFont
import sys
B=Path(os.environ.get('MODEL_RUNTIME', 'runtime/optimized')).resolve();KEY=(B/'api.key').read_text().strip();port=int(sys.argv[1]);label=sys.argv[2]
cases=[]
for a,b in [(17,23),(29,31),(124,7),(203,9)]:cases.append((f'mul-{a}-{b}',[{'role':'user','content':f'计算 {a} × {b}。只输出最终整数，不要解释。'}],str(a*b)))
for a,b in [(935,267),(8204,3987),(4901,1649),(6000,1789)]:cases.append((f'sub-{a}-{b}',[{'role':'user','content':f'计算 {a} - {b}。只输出最终整数，不要解释。'}],str(a-b)))
for repeat,value in [(0,8372),(200,6841),(1000,5926),(4000,4738)]:cases.append((f'copy-{repeat}',[{'role':'user','content':'背景材料用于评估推理系统。'*repeat+f'\n请只输出数字 {value}，不要解释。'}],str(value)))
font=ImageFont.load_default(size=72)
for value in ['5837','9246','6712','3085']:
 im=Image.new('RGB',(448,224),'white');ImageDraw.Draw(im).text((100,65),value,font=font,fill='black');buf=io.BytesIO();im.save(buf,format='PNG');content=[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()}},{'type':'text','text':'Read the four-digit number in the image. Output the digits only.'}];cases.append(('ocr-'+value,[{'role':'user','content':content}],value))
def run(case):
 name,messages,expected=case;req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps({'model':'ieg-infra-qwen3-vl-32b-instruct','messages':messages,'temperature':0,'max_tokens':32}).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+KEY})
 with urllib.request.urlopen(req,timeout=90) as r:x=json.load(r)
 answer=x['choices'][0]['message']['content'].strip();return {'case':name,'expected':expected,'answer':answer,'passed':answer==expected,'usage':x.get('usage')}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,cases))
r={'label':label,'port':port,'passed':all(x['passed'] for x in results),'count':len(results),'results':results};(B/'optimization'/(label+'-quality.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False,indent=2));assert r['passed']
