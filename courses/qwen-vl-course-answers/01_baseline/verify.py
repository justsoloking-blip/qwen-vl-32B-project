"""Recompute archived service metrics; does NOT call a running GPU server."""
from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from course_utils import *
from PIL import Image,ImageDraw,ImageFont
out={k:summary(PROJECT/'benchmarks/results'/f'{k}-baseline.json') for k in ['before_cutover','deployed']}
assert abs(out['deployed']['short']['ttft_ms']-52.91)<0.01
assert abs(out['deployed']['short']['tpot_ms']-11.25)<0.01
assert abs(out['before_cutover']['short']['tpot_ms']-15.03)<0.01
assert abs(out['deployed']['concurrent']['output_tokens_per_s']-572.04)<0.02
save('01_baseline','metrics.json',out)
meta=ROOT/'common/model_metadata';idx=json.loads((meta/'model.safetensors.index.json').read_text());repo=json.loads((meta/'repository.json').read_text())
shards=set(idx['weight_map'].values());assert len(shards)==14
assert shards<={x['rfilename'] for x in repo['siblings']}
config=json.loads((meta/'config.json').read_text());tc=config['text_config'];assert tc['hidden_size']==5120 and tc['num_hidden_layers']==64
# Deterministic course quality set: 10 text + 10 OCR + 10 color + 10 two-image ordering.
data=ROOT/'01_baseline/data';data.mkdir(exist_ok=True);cases=[]
for i in range(10):
 a,b=17+i,23+i;cases.append({'id':f'text-{i}','modality':'text','prompt':f'Compute {a} * {b}. Return the integer only.','expected':str(a*b),'files':[]})
for i in range(10):
 value=str(5837+i);p=data/f'ocr-{i}.png';im=Image.new('RGB',(448,224),'white');ImageDraw.Draw(im).text((100,65),value,font=ImageFont.load_default(size=72),fill='black');im.save(p)
 cases.append({'id':f'ocr-{i}','modality':'image','prompt':'Read the four-digit number. Return the digits only.','expected':value,'files':[p.name]})
colors=['red','blue','green','yellow']
for i in range(10):
 c=colors[i%4];p=data/f'color-{i}.png';im=Image.new('RGB',(1024,1024),c);ImageDraw.Draw(im).text((10,10),str(i),fill='black');im.save(p)
 cases.append({'id':f'color-{i}','modality':'image','prompt':'What is the background color? Return one lowercase English color word.','expected':c,'files':[p.name]})
for i in range(10):
 colors2=[colors[i%4],colors[(i+1)%4]];files=[]
 for j,c in enumerate(colors2):
  p=data/f'multi-{i}-{j}.png';Image.new('RGB',(256,256),c).save(p);files.append(p.name)
 cases.append({'id':f'multi-{i}','modality':'multi-image','prompt':'Return the two background colors in image order, lowercase and separated by one comma, no spaces.','expected':','.join(colors2),'files':files})
for c in cases:
 c['source']='Course-generated, no external images';c['sha256']={f:hashlib.sha256((data/f).read_bytes()).hexdigest() for f in c['files']}
 for f in c['files']:
  with Image.open(data/f) as im:im.verify()
assert len(cases)==40 and len({x['id'] for x in cases})==40
(data/'manifest.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
record('01_baseline','Archived metrics recomputed locally; 40 quality fixtures generated/decoded. No model inference.',{'metrics':out,'model_shards':14,'quality_cases':40,'config_hidden_size':tc['hidden_size']})
print('PASS lesson 1: archived metrics, model metadata, 40 data fixtures')
