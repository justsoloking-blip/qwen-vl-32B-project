"""Promote the validated BF16 candidate, with automatic startup/smoke rollback."""
from pathlib import Path
import json, os, shutil, subprocess, time, urllib.request

B=Path('/deployment/qwen3-vl-32b'); R=B/'optimization'
PY=str(B/'venv/bin/python'); key=(B/'api.key').read_text().strip()
for name in ['combined-functional.json','combined-quality.json','combined-pipeline.json','combined-long-context.json']:
    assert json.loads((R/name).read_text())['passed'],name
bench=json.loads((R/'combined-baseline.json').read_text())
assert len(bench['results'])==20 and not bench['profiled']
before=json.loads((R/'before/config.json').read_text())
candidate=json.loads((R/'configs/combined.json').read_text())
candidate['gpus']=before['gpus']; candidate['base_url']=before['base_url']
for flag in ['--host','--port']:
    candidate['arguments'][candidate['arguments'].index(flag)+1]=before['arguments'][before['arguments'].index(flag)+1]
assert '--quantization' not in candidate['arguments']
(R/'configs/deployed.json').write_text(json.dumps(candidate,indent=2)+'\n')
service=(R/'before/service.py').read_text()
marker='        os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["gpus"]'
assert marker in service
service=service.replace(marker,marker+'\n        os.environ.update(CONFIG.get("env", {}))')
compile(service,str(B/'service.py'),'exec')
def control(action): subprocess.run([PY,str(B/'service.py'),action],check=True)
def ready():
    for _ in range(120):
        try:
            req=urllib.request.Request('http://127.0.0.1:30000/v1/models',headers={'Authorization':'Bearer '+key})
            with urllib.request.urlopen(req,timeout=3) as r: assert r.status==200
            return
        except Exception: time.sleep(4)
    raise RuntimeError('Model did not become ready within eight minutes')
started=time.time(); control('stop')
try:
    shutil.copytree(R/'tp_compile/cache',B/'cache',dirs_exist_ok=True)
    (B/'config.json').write_text(json.dumps(candidate,indent=2)+'\n')
    (B/'service.py').write_text(service)
    control('start'); ready()
    with (R/'deployed-smoke.json').open('w') as f:
        subprocess.run([PY,str(B/'smoke_test.py')],stdout=f,check=True)
except Exception:
    control('stop')
    for name in ['service.py','config.json']: shutil.copy2(R/'before'/name,B/name)
    control('start');ready()
    raise
(R/'deployment.json').write_text(json.dumps({'deployed':True,'timestamp':time.strftime('%FT%T%z'),'cutover_and_smoke_seconds':time.time()-started,'configuration':candidate,'backup_directory':str(R/'before')},indent=2))
print('Validated BF16 configuration deployed; smoke checks passed.',flush=True)
