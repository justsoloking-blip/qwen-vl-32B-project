"""Validate exported real GPU evidence and recompute metrics; does not run a GPU."""
from pathlib import Path
import csv,datetime,gzip,hashlib,json,shutil,statistics,subprocess,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'common'))
from course_utils import summary
E=ROOT/'verification/gpu';C=E/'candidates'
labels=['A_dp2','B_tp2','C_compile','D_fusion','E_compile_fusion','F_keep_device','G_ipc','Q_qk','H_final']
def read(p):return json.loads(p.read_text())
data_integrity=read(E/'data-integrity.json');assert data_integrity['passed'] and data_integrity['cases']==40
assert hashlib.sha256((ROOT/'01_baseline/data/manifest.json').read_bytes()).hexdigest()==data_integrity['manifest_sha256']
snapshot=read(E/'environment/gpu-model-snapshot.json');assert snapshot['passed']
assert sum(x.get('sha256_checked',False) for x in snapshot['files'] if x['file'].endswith('.safetensors'))==14
upstream=read(E/'upstream-regression/expected-failures.json');assert len(upstream)==2 and all(x['expected_failure'] and x['returncode']!=0 for x in upstream)
for name in ['test_dp_mrope_padding.py','test_dp_vision_dispatch.py']:
 text=(E/'upstream-regression'/(name+'.log')).read_text();assert 'Ran 5 tests' in text and 'FAILED (' in text
fixed=(C/'A_dp2/logs/A_dp2-regressions.log').read_text();assert fixed.count('Ran 5 tests')==2 and fixed.count('OK')>=2
rows=[];counts={'benchmark_batches':0,'benchmark_requests':0,'smoke_configurations':0,'mixed_integration_requests':0,'quality_requests':0,'course_quality_requests':0}
for label in labels:
 d=C/label;raw=read(d/'optimization'/f'{label}-baseline.json');metrics=summary(d/'optimization'/f'{label}-baseline.json')
 assert set(metrics)=={'short','long','image','concurrent'}
 assert all(v['rounds']==5 for v in metrics.values())
 assert all(r['usage']['completion_tokens']==128 for b in raw['results'] for r in b['requests'])
 assert read(d/'smoke-results.json')['passed']
 counts['benchmark_batches']+=len(raw['results']);counts['benchmark_requests']+=sum(len(b['requests']) for b in raw['results']);counts['smoke_configurations']+=1
 for case,v in metrics.items():
  req=[r for b in raw['results'] if b['case']==case for r in b['requests']]
  v['tpot_ms_min']=min(r['tpot_ms'] for r in req);v['tpot_ms_max']=max(r['tpot_ms'] for r in req)
 rows.append({'candidate':label,'metrics':metrics})
 if label in ['A_dp2','H_final']:
  x=read(d/'dp-integration-results.json');assert x['passed'] and x['request_count']==19;counts['mixed_integration_requests']+=19
  x=read(d/'optimization'/f'{label}-quality.json');assert x['passed'] and x['count']==16;counts['quality_requests']+=16
 if label in ['Q_qk','H_final']:
  x=read(d/'optimization/course-quality.json');assert x['count']==40
  if label=='H_final':
   assert sum(r['exact_match'] for r in x['results'])==39
   failures=[r for r in x['results'] if not r['exact_match']]
   assert failures==[{'id':'ocr-5','answer':'5 8 4 2','expected':'5842','exact_match':False}]
  counts['course_quality_requests']+=40
out=ROOT/'03_profiling/results/fresh-gpu';out.mkdir(parents=True,exist_ok=True)
(out/'candidate_metrics.json').write_text(json.dumps(rows,indent=2))
table=[]
for r in rows:
 s=r['metrics'];table.append({'candidate':r['candidate'],'short_tpot_ms':s['short']['tpot_ms'],'short_tpot_min_ms':s['short']['tpot_ms_min'],'short_tpot_max_ms':s['short']['tpot_ms_max'],'image_ttft_ms':s['image']['ttft_ms'],'concurrent_tokens_s':s['concurrent']['output_tokens_per_s']})
with (out/'candidate_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
trace_root=E/'profiles'
for label in ['B_tp2','E_compile_fusion','G_ipc']:
 source=C/label/'optimization/traces'/label
 shutil.copytree(source,trace_root/'traces'/label,dirs_exist_ok=True)
subprocess.run([sys.executable,str(ROOT/'common/project/benchmarks/analyze_traces.py'),'--root',str(trace_root),'--out',str(out/'trace_analysis')],check=True)
traces=read(out/'trace_analysis/trace-summary.json');assert len(traces)==12
copies=[]
for p in sorted((trace_root/'traces').glob('*/image/*.gz')):
 events=json.load(gzip.open(p))['traceEvents'];large=[{'name':e['name'],'bytes':e.get('args',{}).get('bytes'),'dur_us':e['dur']} for e in events if e.get('cat')=='gpu_memcpy' and e.get('args',{}).get('bytes')==24*1024*1024]
 source_file=C/p.parent.parent.name/'optimization/traces'/p.parent.parent.name/p.parent.name/p.name
 copies.append({'file':str(source_file.relative_to(E)),'large_24MiB_copies':large})
 if p.parent.parent.name=='B_tp2':assert any('HtoD' in e['name'] for e in large)
 if p.parent.parent.name=='G_ipc':assert not any('HtoD' in e['name'] for e in large)
(out/'memcpy_evidence.json').write_text(json.dumps(copies,indent=2))
shutil.rmtree(trace_root) # Generated analysis staging only; originals remain under candidates.
micro=read(C/'Q_qk/optimization/fused-kernel-test.json');assert len(micro)==16
assert all(r['max_abs_error']<=0.03125 and r['fused_us']>0 and r['reference_us']>0 for r in micro)
special=read(E/'specialization-gpu.json');assert len(special)==12 and {r['variant'] for r in special}=={'runtime_N','constexpr_N'}
assert read(E/'rollback.json')['passed']
stride=read(E/'position-stride-gpu.json');assert stride['known_boundary_regression_reproduced'] and len(stride['cases'])==3
assert all(not x['layout_invariance_passed'] and x['max_abs_difference']>0.03125 for x in stride['cases'])
assert "AssertionError: ('text-0', '130')" in (E/'rejected-qk/first-integration-failure.log').read_text()
q_quality=read(C/'Q_qk/optimization/Q_qk-quality.json');assert q_quality['count']==16
q_course=read(C/'Q_qk/optimization/course-quality.json')
counts['quality_requests']+=16
q_decision={'candidate':'Q_qk','accepted':False,'reason':'Observed mixed-concurrency output 130 instead of 1300; preserve failure even if a later rerun passes.','small_quality_matches':sum(x['passed'] for x in q_quality['results']),'course_quality_matches':sum(x['exact_match'] for x in q_course['results'])}
report={'evidence_validation_passed':True,'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Fresh Linux H200 pair inference + GPU kernel experiments, followed by local evidence recomputation','labels':labels,'course_verification_completed':True,'default_candidate_quality':{'strict_matches':39,'cases':40,'content_matches_ignoring_OCR_whitespace':40,'strict_passed':False,'note':'Post-hoc classification of one formatting failure; original strict grading unchanged.'},'all_candidates_quality_passed':False,'experimental_candidate_decision':q_decision,'counts':{**counts,'gpu_traces':12,'gpu_kernel_cases':16,'specialization_calls':12,'position_stride_counterexamples':3},'evidence_root':'verification/gpu','limitations':['Five rounds per workload, sequential configurations; not a reliable P99 or causal estimate free of drift.','Same prompt generator with random request salts; actual input token counts recorded, prompts not byte-identical.','GPUs 0/1 reserved for experiments; another job remained on GPU6 of the same host.','40 synthetic quality cases do not establish general business quality or video quality.','Profiler timings separate from unprofiled 128-token benchmarks.']}
(ROOT/'verification/full-course-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
for folder,label in [('01_baseline','A_dp2'),('02_dp_fix','A_dp2'),('03_profiling','H_final'),('04_fusion','Q_qk')]:
 (ROOT/folder/'results/fresh-gpu-evidence.json').write_text(json.dumps({'evidence_verified':True,'primary_candidate':label,'evidence':f'verification/gpu/candidates/{label}','aggregate':'verification/full-course-verification.json'},indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
