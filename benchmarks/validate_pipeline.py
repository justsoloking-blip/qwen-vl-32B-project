import os
"""Exercise CUDA IPC feature-pool reuse, video, and prefix-cache behavior."""
import base64, concurrent.futures, io, json, sys, time, urllib.request, uuid
from pathlib import Path
from PIL import Image, ImageDraw

B = Path(os.environ.get('MODEL_RUNTIME', 'runtime/optimized')).resolve()
port, label = int(sys.argv[1]), sys.argv[2]
key = (B/'api.key').read_text().strip()
def request(content, max_tokens=16):
    payload = {'model':'ieg-infra-qwen3-vl-32b-instruct', 'messages':[{'role':'user','content':content}], 'temperature':0, 'max_tokens':max_tokens}
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as r: result=json.load(r)
    return {'answer':result['choices'][0]['message']['content'].strip(), 'elapsed_ms':1000*(time.perf_counter()-start), 'usage':result.get('usage')}
def image_case(i):
    color=['red','blue','green','yellow'][i%4]
    im=Image.new('RGB',(1024,1024),color)
    ImageDraw.Draw(im).text((10,10),f'{i}-{uuid.uuid4().hex}',fill='black')
    buf=io.BytesIO();im.save(buf,format='PNG')
    data='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()
    result=request([{'type':'image_url','image_url':{'url':data}},{'type':'text','text':'What is the background color? Output one English color word only.'}],8)
    result.update(index=i,expected=color,passed=result['answer'].lower().rstrip('.')==color)
    return result
start=time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool: images=list(pool.map(image_case,range(72)))
image_seconds=time.perf_counter()-start
video=Path(os.environ['MODEL_TEST_VIDEO'])
data='data:video/mp4;base64,'+base64.b64encode(video.read_bytes()).decode()
v=request([{'type':'video_url','video_url':{'url':data}},{'type':'text','text':'Describe the background color change from the beginning to the end. Answer briefly in English.'}],64)
s=v['answer'].lower();v['passed']='red' in s and 'blue' in s and s.index('red')<s.index('blue')
prefix=f'Trial {uuid.uuid4().hex}. '+'这是一份关于模型推理和性能的背景材料。'*300+'\n只输出数字 8372。'
cache=[request(prefix,8) for _ in range(6)]
result={'label':label,'port':port,'passed':all(x['passed'] for x in images) and v['passed'] and all(x['answer']=='8372' for x in cache),'image_count':len(images),'image_concurrency':8,'image_seconds':image_seconds,'images':images,'video':v,'prefix_cache':cache}
(B/'optimization'/(label+'-pipeline.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='images'},ensure_ascii=False,indent=2))
assert result['passed']
