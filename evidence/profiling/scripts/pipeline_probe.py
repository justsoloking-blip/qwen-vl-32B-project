"""Cache and image resolution probes with no profiler; main-service only."""
import perf_probe as b
import uuid,json,time,statistics,base64,io
from PIL import Image,ImageDraw
results=[]
original=b.messages
for case in ['short','long','image']:
 fixed=original(case,uuid.uuid4().hex)
 b.messages=lambda case,seed:fixed
 qs=[b.request(case,32) for _ in range(6)]
 results.append({'test':'identical_repeat','case':case,'requests':qs})
 print(case,[round(q['ttft_ms'],2) for q in qs],flush=True)
for size in [512,1024,1536]:
 qs=[]
 for _ in range(5):
  im=Image.new('RGB',(size,size),'white');d=ImageDraw.Draw(im);d.rectangle((size//8,size//8,size//2,size//2),fill='red');d.ellipse((size//2,size//2,7*size//8,7*size//8),fill='blue');seed=uuid.uuid4().hex;d.text((5,size-20),seed,fill='black');buf=io.BytesIO();im.save(buf,format='PNG')
  fixed=[{'role':'user','content':[{'type':'text','text':f'Trial {seed}. Describe the image in detail.'},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()}}]}]
  b.messages=lambda case,seed:fixed
  qs.append(b.request('image',32))
 results.append({'test':'image_resolution','size':size,'requests':qs});print(size,statistics.median(q['ttft_ms'] for q in qs),[q['usage']['prompt_tokens'] for q in qs],flush=True)
(b.OUT/'pipeline-probes.json').write_text(json.dumps(results,indent=2))
