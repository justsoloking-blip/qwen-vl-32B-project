"""Analyze Kineto traces: graph replay correlation, interval unions and kernel groups.
All durations are microseconds in Chrome JSON; sums are not critical-path time.
"""
import argparse,collections,csv,gzip,json,statistics
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'performance');ap.add_argument('--out',type=Path);args=ap.parse_args();ROOT=args.root;OUT=args.out or ROOT/'analysis';OUT.mkdir(parents=True,exist_ok=True)
def category(n):
 l=n.lower()
 if 'triton_' in l and 'fused' in l and '_mm' in l:return 'compiler_fused'
 if any(s in l for s in ['nccl','cross_device','allreduce','all_reduce']):return 'communication'
 if any(s in l for s in ['nvjet','gemm','gemv','triton_mm','cutlass_','splitkreduce']):return 'gemm'
 if any(s in l for s in ['flashattn','flash_fwd','fmha','attention']):return 'attention'
 if 'rmsnorm' in l or 'layernorm' in l or 'layer_norm' in l:return 'normalization'
 if 'silu' in l or 'gelu' in l or 'act_and_mul' in l:return 'activation'
 if 'rope' in l:return 'rope'
 if any(s in l for s in ['index','copy','memcpy','fill','gather','scatter','catarray','elementwise']):return 'layout_elementwise'
 return 'other'
def union(intervals):
 pairs=sorted(intervals);end=-float('inf');u=0
 for a,b in pairs:
  u+=max(0,b-max(a,end));end=max(end,b)
 return u

def stats(events,a=None,b=None):
 if not events:return {}
 if a is None:a=min(e['ts'] for e in events)
 if b is None:b=max(e['ts']+e['dur'] for e in events)
 ev=[e for e in events if e['ts']<b and e['ts']+e['dur']>a];ints=[(max(e['ts'],a),min(e['ts']+e['dur'],b)) for e in ev];u=union(ints);s=collections.defaultdict(float);cnt=collections.Counter()
 for e in ev:
  if e.get('cat')=='kernel':c=category(e['name']);s[c]+=min(e['ts']+e['dur'],b)-max(e['ts'],a);cnt[c]+=1
 return {'window_ms':(b-a)/1000,'gpu_busy_union_ms':u/1000,'no_gpu_activity_pct':100*(1-u/(b-a)),'kernel_sum_ms':sum(s.values())/1000,'kernel_groups_ms':{k:v/1000 for k,v in s.items()},'kernel_groups_count':dict(cnt)}
results=[];rows=[];frontends=[]
for p in sorted((ROOT/'traces').rglob('*.gz')):
 x=json.load(gzip.open(p));es=x['traceEvents'];kernels=[e for e in es if e.get('cat')=='kernel'];gpu=[e for e in es if e.get('cat') in ['kernel','gpu_memcpy','gpu_memset']];graphs=sorted([e for e in es if e.get('name')=='cudaGraphLaunch'],key=lambda e:e['ts']);graphids={g['args']['correlation'] for g in graphs};gks=collections.defaultdict(list)
 for e in kernels:
  if e['args'].get('correlation') in graphids:gks[e['args']['correlation']].append(e)
 if not graphs:
  scopes=[e for e in es if e.get('cat')=='user_annotation' and 'pipeline' in e['name']]
  frontends.append({'trace':str(p.relative_to(ROOT)),'scope_ms':[e['dur']/1000 for e in scopes],'gpu':stats(gpu) if gpu else {},'copies':[{'name':e['name'],'duration_us':e['dur'],'bytes':e.get('args',{}).get('bytes')} for e in es if e.get('cat')=='gpu_memcpy']})
  continue
 gs=[(g,gks[g['args']['correlation']]) for g in graphs if gks[g['args']['correlation']]]
 stable=gs[5:-2] if len(gs)>10 else gs
 first=min(e['ts'] for e in gpu);decbegin=min(e['ts'] for e in gs[0][1]) if gs else first
 a=min(e['ts'] for e in stable[0][1]);b=max(e['ts']+e['dur'] for e in stable[-1][1]);decodeevents=[e for e in gpu if a<=e['ts']<b]
 d={'trace':str(p.relative_to(ROOT)),'label':p.parent.parent.name,'case':p.parent.name,'rank':p.name.split('TP-')[1].split('.')[0],'events':len(es),'graph_replays':len(graphs),'stable_graphs':len(stable),'prefill_and_setup':stats(gpu,first,decbegin),'decode':stats(gpu,a,b),'graph_kernel_count_median':statistics.median(len(ks) for _,ks in stable),'graph_gpu_span_ms_median':statistics.median((max(e['ts']+e['dur'] for e in ks)-min(e['ts'] for e in ks))/1000 for _,ks in stable),'graph_cpu_launch_ms_median':statistics.median(g['dur']/1000 for g,_ in stable),'graph_start_interval_ms_median':statistics.median((min(e['ts'] for e in g2[1])-min(e['ts'] for e in g1[1]))/1000 for g1,g2 in zip(stable,stable[1:]))}
 per=collections.defaultdict(list)
 for e in decodeevents:
  if e['cat']=='kernel':per[e['name']].append(e['dur'])
 d['top_decode_kernels']=[{'name':n,'category':category(n),'count':len(ds),'total_ms':sum(ds)/1000,'mean_us':statistics.mean(ds)} for n,ds in sorted(per.items(),key=lambda kv:sum(kv[1]),reverse=True)[:25]]
 for phase,events in [('prefill',[e for e in kernels if e['ts']<decbegin]),('decode',[e for e in decodeevents if e['cat']=='kernel'])]:
  totals=collections.defaultdict(list)
  for e in events:totals[e['name']].append(e['dur'])
  for n,ds in totals.items():rows.append({'label':d['label'],'case':d['case'],'rank':d['rank'],'phase':phase,'category':category(n),'kernel':n,'count':len(ds),'total_ms':sum(ds)/1000,'mean_us':statistics.mean(ds)})
 cpu={e['args'].get('External id'):e for e in es if e.get('cat')=='cpu_op' and 'External id' in e.get('args',{})};shapes=collections.defaultdict(lambda:[0,0])
 for e in kernels:
  c=cpu.get(e['args'].get('External id'))
  if c and c['name'] in ['aten::mm','aten::addmm','aten::bmm']:
   key=(c['name'],json.dumps(c['args'].get('Input Dims')));shapes[key][0]+=1;shapes[key][1]+=e['dur']/1000
 d['eager_gemm_shapes']=[{'op':k[0],'shapes':json.loads(k[1]),'count':v[0],'kernel_ms':v[1]} for k,v in sorted(shapes.items(),key=lambda kv:kv[1][1],reverse=True)[:20]]
 results.append(d)
 print(d['label'],d['case'],d['rank'],'prefill',round(d['prefill_and_setup']['window_ms'],2),'decode',round(d['graph_start_interval_ms_median'],3),'bubble',round(d['decode']['no_gpu_activity_pct'],2),'kernels',d['graph_kernel_count_median'],{k:round(v/d['decode']['kernel_sum_ms']*100,1) for k,v in d['decode']['kernel_groups_ms'].items()})
(OUT/'frontend-summary.json').write_text(json.dumps(frontends,indent=2))
(OUT/'trace-summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
with (OUT/'kernels.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
