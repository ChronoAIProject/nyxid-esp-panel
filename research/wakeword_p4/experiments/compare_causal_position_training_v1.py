"""Paired training census, no decision on qualification."""
import collections,hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results')
def main():
 paths=[R/f'4090-causal-{kind}-all-training-wave-diagnostic-v1.json' for kind in ['unit','position']];a,b=[json.load(p.open()) for p in paths];assert len(a['records'])==len(b['records'])==4320;groups={}
 for x,y in zip(a['records'],b['records']):
  assert all(x[k]==y[k] for k in ['index','path','sha256','class_id','carrier']);key=f"{x['carrier']}/class{x['class_id']}";g=groups.setdefault(key,dict(recordings=0,baseline_hits=0,position_hits=0,kept=0,added=0,lost=0));h=bool(x['events_samples']);j=bool(y['events_samples']);g['recordings']+=1;g['baseline_hits']+=h;g['position_hits']+=j;g['kept']+=h and j;g['added']+=not h and j;g['lost']+=h and not j
 out=dict(schema='causal-position-paired-training-census-v1',files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*paths]},baseline={k:a[k] for k in ['target_hits','target_count','negative_hits','negative_count']},position={k:b[k] for k in ['target_hits','target_count','negative_hits','negative_count']},groups=groups,qualification=False,limitations='Consumed training audio only. Raw float argmax exact matcher, no deployed confidence/span/cooldown. No new threshold selection or heldout data. Improvement here does not establish FAR or generalization.')
 with (R/'4090-causal-position-paired-training-census-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
