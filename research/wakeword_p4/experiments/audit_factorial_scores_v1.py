"""Authenticate every factorial trace and independently recompute paired metrics."""
import collections,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from evaluate import smooth,event_indices
base=Path('.local/wakeword');root=base/'balanced-rendering-factorial-v1-scores';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();r=json.loads((root/'report.json').read_text());assert sha(root/'scores.npz')==r['scores_sha256'];assert len(r['details'])==32256
original=json.loads((base/'piper-balanced-training-features-v1/report.json').read_text())['records'];controls=json.loads((base/'balanced-rendering-factorial-v1/controls.json').read_text())['records'];groups={};paired={};seen=set()
with np.load(root/'scores.npz') as z:
 assert len(z.files)==32256
 for j,d in enumerate(r['details']):
  i=j//4;arm=['quiet','noise15','rir','combined'][j%4];assert d['source_index']==i and d['arm']==arm and (i,arm) not in seen;seen.add((i,arm));q=z[f'{i}_{arm}'];assert hashlib.sha256(q.tobytes()).hexdigest()==d['score_sha256']
  source=original[3*i+(0 if arm=='quiet' else 2)] if arm in ['quiet','combined'] else controls[2*i+(0 if arm=='noise15' else 1)]
  assert d['target']==source['target'] and d['voice']==source['voice'];assert len(q)==d['raw_shape'][0]//3
  five=np.convolve(q[49:].astype(np.float64),np.ones(5)/5,mode='valid');first=150+3*np.arange(len(five));valid=((first-150)*160<=32000)&((first+12-150)*160<=32000)&((first+1)*160>=32000+source['signal_samples']);assert int(valid.sum())==d['valid_support_positions']
  for t,h in d['thresholds'].items():
   v=float(t);assert float(five.max())==h['warmed_max_five'];assert event_indices(smooth(q,5),v)==h['full_trace_events'];assert event_indices(five,v)==h['warmed_restarted_events'];assert int(np.count_nonzero(five[valid]>=v))==h['container_support_positions']
   for group in [f"{arm}/label/{d['label']}",f"{arm}/phrase/{d['target']}",f"{arm}/speed/{d['speed']}/label/{d['label']}"]:
    g=groups.setdefault(group+'/'+t,dict(variants=0,full_trace_event_clips=0,warmed_high_clips=0,warmed_restarted_event_clips=0,container_support_hit_clips=0));g['variants']+=1;g['full_trace_event_clips']+=bool(h['full_trace_events']);g['warmed_high_clips']+=bool(five.max()>=v);g['warmed_restarted_event_clips']+=bool(h['warmed_restarted_events']);g['container_support_hit_clips']+=h['container_support_positions']>0
 for i in range(8064):
  ds=r['details'][4*i:4*i+4];assert len({(d['voice'],d['target'],d['speed']) for d in ds})==1
  for t in ds[0]['thresholds']:
   for metric in ['full_trace_events','container_support_positions']:
    pattern=''.join(str(int(bool(d['thresholds'][t][metric]))) for d in ds);key=f"{ds[0]['label']}/{t}/{metric}";g=paired.setdefault(key,{});g[pattern]=g.get(pattern,0)+1
assert groups==r['groups']
out=dict(traces=32256,all_hashes_events_support_and_groups_exact=True,source_pairs=8064,arm_order=['quiet','noise15','rir','combined'],paired_patterns=paired,report_sha256=sha(root/'report.json'),scores_sha256=sha(root/'scores.npz'),script_sha256=sha(Path(__file__)),historical_groups={k:v for k,v in groups.items() if k.endswith('/0.99609375') and '/label/' in k and '/speed/' not in k},limits='Saved-score independent recomputation shares detector functions; no independent inference. Training-only sensitivity, no threshold selection/generalization/causal deployment claim.')
Path('research/wakeword_p4/results/4090-factorial-score-audit-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['historical_groups']))
