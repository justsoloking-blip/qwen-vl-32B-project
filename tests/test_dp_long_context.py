import os
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request

from test_dp_integration import KEY, MODEL, URL, picture

BASE = Path(os.environ.get('MODEL_RUNTIME', 'runtime/optimized')).resolve()
cases = [
    ('long-text', [{'role':'user','content':'背景材料。'*10000+'\n请只输出数字 8197，不要解释。'}], '8197'),
    ('long-vision', [{'role':'user','content':[
        picture((448,224),'green','rectangle'),
        {'type':'text','text':'背景材料。'*6000+'\nWhat color is the shape in the image? Reply with one English color word only.'},
    ]}], 'green'),
]


def run(case):
    name,messages,expected=case
    payload={'model':MODEL,'messages':messages,'temperature':0,'max_tokens':32}
    req=urllib.request.Request(URL+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={
        'Content-Type':'application/json','Authorization':'Bearer '+KEY,
    })
    started=time.monotonic()
    with urllib.request.urlopen(req,timeout=90) as r:
        data=json.load(r)
    answer=data['choices'][0]['message']['content']
    assert answer.strip().lower()==expected,(name,answer)
    assert data['usage']['prompt_tokens']>16384,data['usage']
    return {'case':name,'answer':answer,'usage':data['usage'],'seconds':round(time.monotonic()-started,3)}


with ThreadPoolExecutor(max_workers=2) as pool:
    results=list(pool.map(run,cases))
report={'passed':True,'base_url':URL,'results':results}
(BASE/'long-context-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
