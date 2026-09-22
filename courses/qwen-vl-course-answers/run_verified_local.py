"""Portable answer checks. GPU execution is explicitly outside this runner."""
from pathlib import Path
import datetime,json,platform,subprocess,sys
ROOT=Path(__file__).resolve().parent
rows=[]
for folder in ['01_baseline','02_dp_fix','03_profiling','04_fusion']:
 log=ROOT/folder/'results/local-run.log'
 with log.open('w') as f:r=subprocess.run([sys.executable,str(ROOT/folder/'verify.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 rows.append({'folder':folder,'passed':r.returncode==0,'log':str(log.relative_to(ROOT))});print(folder,'PASS' if r.returncode==0 else 'FAIL',flush=True)
 if r.returncode:print(log.read_text()[-2500:])
result={'executed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),'local_checks':rows,'local_passed':all(x['passed'] for x in rows),'gpu_execution_performed_by_this_runner':False,'gpu_status':'This runner checks CPU methods and archived data only. See verification/full-course-verification.json for separately executed GPU validation.'}
(ROOT/'verification/local-run.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
raise SystemExit(0 if result['local_passed'] else 1)
