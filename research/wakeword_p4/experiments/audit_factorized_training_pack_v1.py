"""Full original-subset/control links, arrays/crops/labels/identity partition audit."""
import collections,hashlib,json
from pathlib import Path
import numpy as np
base=Path('.local/wakeword');root=base/'factorized-training-features-v1';old=base/'piper-balanced-training-features-v1';controls=base/'balanced-rendering-factorial-v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((root/'report.json').read_text());p=json.loads((old/'report.json').read_text());c=json.loads((controls/'controls.json').read_text());assert sha(old/'report.json')=='93733755fec50b41cf71fcc8e2efa9fdfc14956a61e8b50e98e4bd110d5248b3';assert sha(root/'training.npz')==r['pack_sha256'];assert len(r['records'])==40320 and r['wake_models_scored'] and not r['final_data_used'];counts=collections.Counter();arms=collections.Counter();original=0;sha_controls=sha(controls/'controls.json')
with np.load(root/'training.npz',allow_pickle=False) as z:
 x,y=z['x'],z['y'];assert x.dtype==np.uint16 and x.shape==(40320,150,40) and y.dtype==np.float32 and y.shape==(40320,)
 for j,d in enumerate(r['records']):
  i,v=divmod(j,5);assert d['pack_index']==j and d['source_index']==i and d['variant']==v
  s=p['records'][3*i+v] if v<3 else c['records'][2*i+v-3];originroot=old if v<3 else controls;rawpath=originroot/(s['raw'] if v<3 else s['files']['raw']['path']);expected=s['raw_file_sha256'] if v<3 else s['files']['raw']['sha256'];assert (root/d['raw']).resolve()==rawpath.resolve();assert sha(rawpath)==d['raw_file_sha256']==expected
  raw=np.load(rawpath,allow_pickle=False);assert raw.dtype==np.uint16 and list(raw.shape)==d['raw_shape'] and hashlib.sha256(raw.tobytes()).hexdigest()==d['raw_sha256'];end=d['feature_crop_end'];np.testing.assert_array_equal(x[j],raw[end-150:end]);assert y[j]==int(d['target']=='hey_nyx')
  assert d['split']=='training' and d['mechanical_pass'] and d['entire_signal_fits'];assert d['voice']==s['voice'] and d['target']==s['target'] and d['source']==s['source'] and d['signal_samples']==s['signal_samples'];assert d['length_multiplier']==s['length_multiplier' if v<3 else 'speed']
  assert d['origin']['report_sha256']==('93733755fec50b41cf71fcc8e2efa9fdfc14956a61e8b50e98e4bd110d5248b3' if v<3 else sha_controls)
  assert 150<=end<=min(350,len(raw)) and (min(350,len(raw))-150)*160<=32000 and (end+1)*160>=32000+d['signal_samples'];counts[d['voice']+'/'+d['target']]+=1;arms[d['arm']]+=1;original+=v<3
  if j%2000==0:print(json.dumps(dict(verified=j+1,total=40320)),flush=True)
voices=sorted({int(d['voice']) for d in r['records']});assert voices==sorted(np.random.default_rng(726).permutation(904)[:672].tolist());assert len(counts)==2688 and set(counts.values())=={15};assert original==24192 and set(arms.values())=={8064}
out=dict(verified=40320,original_subset_exact=24192,new_controls_exact=16128,identity_partition_exact=True,groups=2688,variants_per_group=15,arms=dict(arms),pack_sha256=r['pack_sha256'],report_sha256=sha(root/'report.json'),script_sha256=sha(Path(__file__)),limits='Full stored-array/crop/hashlink replay, not new waveform synthesis or independent frontend implementation. Training-only previously diagnosed sources; no inference/newdev/final access.')
Path('research/wakeword_p4/results/4090-factorized-training-pack-audit-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
