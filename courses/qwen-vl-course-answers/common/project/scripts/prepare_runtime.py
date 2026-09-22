#!/usr/bin/env python3
"""Materialize one preserved version outside Git. Never starts a server."""
import argparse, importlib.util, json, os, secrets, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--version',choices=['original','dp-attention-fixed','optimized'],required=True)
p.add_argument('--model-path',type=Path,required=True)
p.add_argument('--venv',type=Path,default=ROOT/'.venv')
p.add_argument('--gpus',help='Override the GPUs of the preserved configuration')
p.add_argument('--port',type=int,default=30000)
p.add_argument('--runtime-root',type=Path,default=ROOT/'runtime')
a=p.parse_args();target=(a.runtime_root/a.version).resolve();source=ROOT/'versions'/a.version
if target.exists():raise SystemExit(f'{target} already exists. Choose another --runtime-root to preserve its state.')
if not a.model_path.is_dir():raise SystemExit('--model-path must point to a local model snapshot directory')
python=a.venv.resolve()/'bin/python'
if not python.exists():raise SystemExit('Create a Linux Python 3.12 virtual environment first.')
info=json.loads(subprocess.check_output([str(python),'-c','import importlib.metadata as m,importlib.util,json;print(json.dumps({"version":m.version("sglang"),"path":str(next(iter(importlib.util.find_spec("sglang").submodule_search_locations)))}))'],text=True))
if info['version']!='0.5.6.post2':raise SystemExit('The preserved patch targets exactly sglang==0.5.6.post2')
c=json.loads((source/'config.json').read_text());args=c['arguments'];args[args.index('--model-path')+1]=str(a.model_path.resolve());args[args.index('--port')+1]=str(a.port);c['base_url']=f'http://127.0.0.1:{a.port}/v1'
if a.gpus:c['gpus']=a.gpus
old_umask=os.umask(0o077)
try:
 target.mkdir(parents=True);(target/'logs').mkdir();(target/'optimization').mkdir();(target/'venv').symlink_to(a.venv.resolve(),target_is_directory=True)
 shutil.copytree(info['path'],target/'overlay/sglang',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for item in (source/'src').rglob('*.py'):
  out=target/'overlay'/item.relative_to(source/'src');out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(item,out)
 service=(source/'service.py').read_text()
 # Packaging-only adaptation: the historical status check had a literal port 30000.
 service=service.replace('"http://127.0.0.1:30000/v1/models"','CONFIG["base_url"] + "/models"')
 if 'sys.path.insert(0, str(BASE / "overlay"))' not in service:
  service=service.replace('BASE = Path(__file__).resolve().parent', 'BASE = Path(__file__).resolve().parent\nsys.path.insert(0, str(BASE / "overlay"))\nos.environ["PYTHONPATH"] = str(BASE / "overlay")')
 (target/'service.py').write_text(service);(target/'config.json').write_text(json.dumps(c,indent=2)+'\n')
 key=os.environ.get('MODEL_API_KEY') or secrets.token_urlsafe(32);(target/'api.key').write_text(key+'\n');(target/'api.key').chmod(0o600)
finally:os.umask(old_umask)
print(f'Prepared {target}. API key stored in api.key with mode 0600, never printed.\nStart: {python} {target}/service.py start')
