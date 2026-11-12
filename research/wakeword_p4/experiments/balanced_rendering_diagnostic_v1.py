"""All-source rendering census from authenticated training-fit reports."""
import collections,hashlib,json
from pathlib import Path
base=Path('.local/wakeword');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=base/'piper-balanced-training-features-v1/report.json';assert sha(p)=='93733755fec50b41cf71fcc8e2efa9fdfc14956a61e8b50e98e4bd110d5248b3';pack=json.loads(p.read_text());result={};inputs={}
for v in ['v45','v46']:
 root=base/f'{v}-balanced-training-support-v1';r=json.loads((root/'report.json').read_text());assert sha(root/'scores.npz')==r['scores_sha256'];groups={};sources={}
 for d,x in zip(r['details'],pack['records']):
  assert d['index']==x['pack_index'] and d['source']==x['source'] and d['variant']==x['variant']
  h=d['modes']['int8']['thresholds']['0.99609375'];positive=bool(d['label']);failure=not bool(h['support_high_positions']) if positive else bool(h['full_high'])
  for key in [f"label/{d['label']}/speed/{x['length_multiplier']}",f"label/{d['label']}/variant/{x['variant']}",f"phrase/{x['target']}/variant/{x['variant']}"]:
   g=groups.setdefault(key,dict(variants=0,failures=0));g['variants']+=1;g['failures']+=int(failure)
  source=sources.setdefault(x['source'],{'label':d['label'],'failure_by_variant':{}});source['failure_by_variant'][str(x['variant'])]=failure
 patterns=collections.Counter()
 for s in sources.values():
  assert len(s['failure_by_variant'])==3
  pattern=''.join(str(int(s['failure_by_variant'][str(i)])) for i in range(3));patterns[f"{s['label']}/{pattern}"]+=1
 assert len(sources)==8064
 result[v]={'groups':groups,'source_failure_patterns':dict(patterns)};inputs[v]={'report_sha256':sha(root/'report.json'),'scores_sha256':r['scores_sha256']}
d=json.loads((base/'v46-old-development-float-probe-v1/report.json').read_text());w=[{'protocol':x['protocol'],'record':x['record'],'float_events':x['float_events'],'int8_events':x['int8_events']} for x in d['details'] if x['record']['target']!='hey_nyx' and (x['float_events'] or x['int8_events'])]
out=dict(inputs=inputs,pack_report_sha256=sha(p),versions=result,development_failure_witnesses=w,threshold=.99609375,script_sha256=sha(Path(__file__)),limits='Descriptive full census, not source pruning/failure mining or threshold selection. Variant2 combines RIR and15dB noise; neither factor isolated. Positive failure is missing container-support;negative failure is warmed five high, not the same metric. Training rendering concentration does not explain unaugmented development failures. No new/final data accessed.')
Path('research/wakeword_p4/results/4090-balanced-rendering-diagnostic-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({v:x['source_failure_patterns'] for v,x in result.items()}))
