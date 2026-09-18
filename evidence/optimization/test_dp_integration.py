"""Uneven mixed text/vision requests, exercised concurrently across DP ranks."""
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import time
import urllib.request
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
URL = os.environ.get('MODEL_API_URL', 'http://127.0.0.1:30001')
KEY = (BASE / 'api.key').read_text().strip()
MODEL = 'ieg-infra-qwen3-vl-32b-instruct'


def picture(size, color, shape):
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    w, h = size
    box = (w//4, h//4, 3*w//4, 3*h//4)
    if shape == 'rectangle':
        draw.rectangle(box, fill=color)
    else:
        draw.ellipse(box, fill=color)
    buf = io.BytesIO()
    image.save(buf, 'PNG')
    return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()}}


CASES = []
for i, repeats in enumerate([0, 50, 300, 900]):
    value = 1300 + i
    CASES.append((f'text-{i}', [{'role': 'user', 'content': '背景材料。'*repeats + f'\n请只输出数字 {value}，不要解释。'}], [str(value)]))
for i, (size, color, shape) in enumerate([
    ((448,224), 'red','rectangle'), ((224,448), 'blue','ellipse'),
    ((640,320), 'green','rectangle'), ((320,640), 'purple','ellipse'),
]):
    CASES.append((f'image-{i}', [{'role':'user','content':[
        picture(size,color,shape),
        {'type':'text','text':'What color is the shape? Reply with one English color word only.'},
    ]}], [color]))
CASES.append(('two-images', [{'role':'user','content':[
    picture((224,224),'red','rectangle'),
    picture((448,224),'blue','ellipse'),
    {'type':'text','text':'State the color of the shape in the first image, then in the second image. Reply with two English color words only.'},
]}], ['red','blue']))


def run(case):
    name, messages, expected = case
    start = time.monotonic()
    payload = {'model':MODEL, 'messages':messages, 'temperature':0, 'max_tokens':40}
    req = urllib.request.Request(URL+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={
        'Content-Type':'application/json','Authorization':'Bearer '+KEY,
    })
    with urllib.request.urlopen(req,timeout=45) as r:
        result = json.load(r)
    answer = result['choices'][0]['message']['content']
    if name.startswith('text-'):
        assert answer.strip() == expected[0], (name,answer)
    else:
        assert all(word in answer.lower() for word in expected), (name,answer)
        if len(expected)==2:
            assert answer.lower().index(expected[0]) < answer.lower().index(expected[1]), answer
    return {'case':name,'answer':answer,'seconds':round(time.monotonic()-start,3),'usage':result.get('usage')}


if __name__ == '__main__':
    results = []
    # A lone request exercises an idle partner; concurrent rounds cover uneven DP loads.
    results.append(run(CASES[0]))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for repeat in range(2):
            results.extend(pool.map(run, CASES))
    report = {'passed':True,'base_url':URL,'request_count':len(results),'results':results}
    (BASE/'dp-integration-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
