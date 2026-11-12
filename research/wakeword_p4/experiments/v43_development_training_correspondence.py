"""Group consumed development float/INT8 failures alongside authenticated training support."""
import collections,hashlib,json
from pathlib import Path
import numpy as np
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=Path('.local/wakeword');devroot=base/'v43-old-development-float-probe-v1';trainroot=base/'v43-training-support-v1'
dev=json.loads((devroot/'report.json').read_text());train=json.loads((trainroot/'report.json').read_text())
assert dev['model_sha256']==train['model_sha256']=='fe8e46206069e841eb8171beaca727b147602814e4130394ffd16fc8c8fff2ba'
assert dev['threshold']==.99609375 and dev['all_int8_traces_exact']
assert sha(devroot/'scores.npz')==dev['scores_sha256'] and sha(trainroot/'scores.npz')==train['scores_sha256']
pack=base/'mac-evidence-import-v1/old-complete-container-pack-v1/study'
assert sha(pack/'sources.json')=='0f2b008c2525ee9c093580008af1029618823f92d2718622acd513e682cbc981'
assert sha(pack/'records.json')=='757f1e6df724943c0e0bf68b9fe7de9b3402cbcebb9cf6c87a89e876ee9af090'
sources=json.loads((pack/'sources.json').read_text());records=json.loads((pack/'records.json').read_text());threshold=dev['threshold']
training_groups={}
with np.load(trainroot/'scores.npz') as arrays:
 for d in train['details']:
  r=records[d['pack_index']];alias=sources[r['source_index']]['aliases'][0];q=arrays[f"int8_{d['pack_index']}"]
  assert hashlib.sha256(q.tobytes()).hexdigest()==d['score_sha256']
  five=np.convolve(q[49:].astype(np.float64),np.ones(5)/5,mode='valid');assert float(five.max())==d['full_int8_max_five']
  identity=json.dumps(sorted(alias['speakers']));phrase=alias['phrase'].replace(' ','_')
  for key in [f"{d['engine']}/{phrase}",f"{d['engine']}/{phrase}/{identity}"]:
   g=training_groups.setdefault(key,dict(variants=0,center_float_high=0,full_int8_five_high=0));g['variants']+=1;g['center_float_high']+=int(d['window_float']>=threshold);g['full_int8_five_high']+=int(five.max()>=threshold)
groups={};witnesses=[];decisions=collections.Counter()
with np.load(devroot/'scores.npz') as arrays:
 for d in dev['details']:
  r=d['record'];f=arrays[f"{d['protocol']}_{d['index']}"];assert float(f.max())==d['float_max_five']
  engine={'phone':'kokoro','vctk':'piper'}[d['protocol']];phrase=r['intended_phrase'];identity=r['voice'];negative=r['target']!='hey_nyx'
  a=bool(d['float_events']);b=bool(d['int8_events']);decision=('both' if a and b else 'float_only' if a else 'int8_only' if b else 'neither')
  if negative:decisions[decision]+=1
  for key in [f'{engine}/{phrase}',f'{engine}/{phrase}/{identity}']:
   g=groups.setdefault(key,dict(clips=0,negative_clips=0,float_events=0,int8_events=0,both_negative=0,float_only_negative=0,int8_only_negative=0));g['clips']+=1;g['negative_clips']+=int(negative);g['float_events']+=int(a);g['int8_events']+=int(b)
   if negative and decision!='neither':g[decision+'_negative']+=1
  if negative and (a or b):witnesses.append(dict(protocol=d['protocol'],index=d['index'],engine=engine,phrase=phrase,conditioning_identity=identity,path=r['path'],source_sha256=r['source_sha256'],decision=decision,float_max_five=d['float_max_five'],int8_max_five=d['int8_max_five'],float_events=d['float_events'],int8_events=d['int8_events'],training_engine_phrase=training_groups.get(f'{engine}/{phrase}'),exact_conditioning_identity_match=False))
assert sum(decisions.values())==304
out=dict(scope='All380 consumed old development clips vs all6597 eligible old-training variants; historical .99609375 only',model_sha256=dev['model_sha256'],threshold=threshold,inputs=dict(development_report_sha256=sha(devroot/'report.json'),training_report_sha256=sha(trainroot/'report.json'),sources_sha256=sha(pack/'sources.json'),records_sha256=sha(pack/'records.json')),negative_decisions=dict(decisions),development_groups=groups,training_groups=training_groups,negative_witnesses=witnesses,script_sha256=sha(Path(__file__)),limits='Engine/phrase is an aggregate correspondence, not paired examples. Development voices are held-out conditioning identities: no exact training identity mapping is asserted; Piper training speaker arrays are recorded as original metadata, not equated with VCTK p identifiers. Center training float vs full training INT8 confounds phase/state; full actual float is measured only for development. No causal checkpoint comparison, new data, threshold selection, training or promotion.')
Path('research/wakeword_p4/results/4090-v43-development-training-correspondence-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['negative_decisions']))
