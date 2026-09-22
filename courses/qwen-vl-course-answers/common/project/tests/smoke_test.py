import os
import base64
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
from PIL import Image, ImageDraw

BASE = Path(os.environ.get('MODEL_RUNTIME', 'runtime/optimized')).resolve()
URL = os.environ.get('MODEL_API_URL', 'http://127.0.0.1:30000').rstrip('/')
MODEL = 'ieg-infra-qwen3-vl-32b-instruct'
KEY = (BASE / 'api.key').read_text().strip()


def request(path, payload=None, auth=True):
    headers = {'Content-Type': 'application/json'}
    if auth:
        headers['Authorization'] = 'Bearer ' + KEY
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(URL + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.status, response.read().decode()


results = {'model': MODEL, 'tested_at': time.strftime('%FT%T%z')}
try:
    request('/v1/models', auth=False)
    raise AssertionError('Unauthenticated request unexpectedly accepted')
except urllib.error.HTTPError as exc:
    assert exc.code in (401, 403), exc.code
    results['unauthenticated_status'] = exc.code
status, body = request('/v1/models')
models = json.loads(body)
assert MODEL in [x['id'] for x in models['data']], models
results['models'] = models
status, body = request('/health')
assert status == 200
results['health_status'] = status

started = time.monotonic()
status, body = request('/v1/chat/completions', {
    'model': MODEL, 'messages': [{'role': 'user', 'content': '计算 17×23，只输出整数结果。'}],
    'temperature': 0, 'max_tokens': 32,
})
text_result = json.loads(body)
assert text_result['choices'][0]['message']['content'].strip() == '391', text_result
results['text'] = {'elapsed_seconds': round(time.monotonic()-started, 2), 'response': text_result}

image = Image.new('RGB', (448, 224), 'white')
draw = ImageDraw.Draw(image)
draw.rectangle((32, 48, 160, 176), fill='red')
draw.ellipse((280, 48, 408, 176), fill='blue')
image.save(BASE / 'smoke-test-image.png')
buffer = io.BytesIO()
image.save(buffer, format='PNG')
data_url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
started = time.monotonic()
status, body = request('/v1/chat/completions', {
    'model': MODEL,
    'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': data_url}},
        {'type': 'text', 'text': 'Name the color and shape of each object from left to right. Answer in English in one short sentence.'},
    ]}], 'temperature': 0, 'max_tokens': 96,
})
vision_result = json.loads(body)
answer = vision_result['choices'][0]['message']['content'].lower()
assert all(term in answer for term in ('red', 'square', 'blue', 'circle')), vision_result
assert answer.index('red') < answer.index('blue'), vision_result
results['vision'] = {'elapsed_seconds': round(time.monotonic()-started, 2), 'response': vision_result}

status, body = request('/v1/chat/completions', {
    'model': MODEL, 'messages': [{'role': 'user', 'content': '请只回复：部署成功'}],
    'temperature': 0, 'max_tokens': 32, 'stream': True,
})
assert 'data: [DONE]' in body, body
chunks = [json.loads(line[6:]) for line in body.splitlines() if line.startswith('data: {')]
stream_text = ''.join(c['choices'][0]['delta'].get('content', '') or '' for c in chunks if c['choices'])
assert '部署成功' in stream_text, stream_text
results['streaming'] = {'completed': True, 'content': stream_text}
results['passed'] = True
(BASE / 'smoke-results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(results, ensure_ascii=False, indent=2))
