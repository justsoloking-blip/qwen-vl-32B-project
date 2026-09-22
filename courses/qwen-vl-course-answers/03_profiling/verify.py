"""Re-analyze real archived traces and derive the candidate comparison table."""
from pathlib import Path
import ast,csv,gzip,json,math,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from course_utils import *
files={'A_dp2':'before','B_tp2':'tp_base','C_compile':'tp_compile','D_fusion':'allreduce_fusion','E_compile_fusion':'compile_allreduce','F_keep_device':'compile_device','G_ipc':'compile_ipc','H_final':'deployed'}
rows=[]
for name,label in files.items():
 s=summary(PROJECT/'benchmarks/results'/f'{label}-baseline.json')
 rows.append({'candidate':name,'ttft_ms':s['short']['ttft_ms'],'tpot_ms':s['short']['tpot_ms'],'image_ttft_ms':s['image']['ttft_ms'],'concurrent_tokens_s':s['concurrent']['output_tokens_per_s']})
save('03_profiling','candidate_metrics.json',rows)
with (ROOT/'03_profiling/results/candidate_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
# Independent correctness examples for the union function used by the project analyzer.
source=PROJECT/'benchmarks/analyze_traces.py';tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='union');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
for intervals,expected in [([],0),([(0,10),(5,15)],15),([(0,5),(5,10)],10),([(0,10),(2,3)],10),([(20,25),(0,5)],10)]:assert ns['union'](intervals)==expected
out=ROOT/'03_profiling/results/trace_analysis'
subprocess.run([sys.executable,str(source),'--root',str(ROOT/'common/archive'),'--out',str(out)],check=True)
traces=json.loads((out/'trace-summary.json').read_text());assert len(traces)==12
for r in traces:
 assert 0<=r['decode']['no_gpu_activity_pct']<=100
 assert r['stable_graphs']>0 and r['graph_kernel_count_median']>0
# Verify raw transfer events independently, not merely a copied summary.
copies=[]
for p in sorted((ROOT/'common/archive/traces').glob('*/image/*.gz')):
 events=json.load(gzip.open(p))['traceEvents'];mem=[e for e in events if e.get('cat')=='gpu_memcpy']
 large=[{'name':e['name'],'bytes':e.get('args',{}).get('bytes'),'dur_us':e['dur']} for e in mem if e.get('args',{}).get('bytes')==24*1024*1024]
 copies.append({'file':str(p.relative_to(ROOT)),'large_24MiB_copies':large})
 if p.parent.parent.name=='tp2':assert any('HtoD' in e['name'] or 'HtoD' in str(e) or 'H2D' in e['name'] for e in large),large
 else:assert not any('HtoD' in e['name'] or 'H2D' in e['name'] for e in large),large
save('03_profiling','memcpy_evidence.json',copies)
record('03_profiling','Fresh local recomputation of archived JSON and 12 actual GPU traces, not a new GPU run.',{'candidates':rows,'traces':len(traces),'interval_union_unit_checks':5,'memcpy':copies})
print('PASS lesson 3: 8 candidates, 12 traces, interval union and raw memcpy evidence')
