import json,statistics
from pathlib import Path
R=Path('/deployment/qwen3-vl-32b/optimization');s=[]
for p in sorted(R.glob('*-baseline.json')):
 x=json.loads(p.read_text());print(x['label'])
 for c in sorted(set(r['case'] for r in x['results'])):
  rs=[r for r in x['results'] if r['case']==c];qs=[q for r in rs for q in r['requests']]
  row={'label':x['label'],'case':c,'rounds':len(rs),'concurrency':rs[0]['concurrency'],'n':len(qs),'ttft_p50_ms':statistics.median(q['ttft_ms'] for q in qs),'tpot_p50_ms':statistics.median((q['elapsed_s']*1000-q['ttft_ms'])/(q['usage']['completion_tokens']-1) for q in qs),'output_tokens_per_s':sum(q['usage']['completion_tokens'] for q in qs)/sum(r['wall_s'] for r in rs),'ttft_min_ms':min(q['ttft_ms'] for q in qs),'ttft_max_ms':max(q['ttft_ms'] for q in qs)};s.append(row);print(c,round(row['ttft_p50_ms'],1),round(row['tpot_p50_ms'],2),round(row['output_tokens_per_s'],1))
(R/'benchmark-summary.json').write_text(json.dumps(s,indent=2))
