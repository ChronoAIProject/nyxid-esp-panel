"""Fresh-noise full source geometry and authenticated120k schedule; no inference."""
import hashlib,json
from pathlib import Path
import numpy as np
import soundfile as sf
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=Path('.local/wakeword'); source=base/'piper-balanced-training-full-v1';pack=base/'piper-balanced-training-features-v1';factor=base/'factorized-training-features-v1'
assert sha(source/'manifest.jsonl')=='e3a69b86294bf91db29777428e785252411710834d22ba92a037d595c963cdc0'
assert sha(pack/'report.json')=='93733755fec50b41cf71fcc8e2efa9fdfc14956a61e8b50e98e4bd110d5248b3'
assert sha(factor/'report.json')=='7b798653f73e2dcd3b01fd5b7092c947a26b9c8ec2525ff295de2e6478042e44'
rows=json.loads((pack/'report.json').read_text())['records'];dry={};source_count=0
for r in rows:
 if r['variant']!=0:continue
 p=source/r['source'];assert sha(p)==r['source_sha256'];pcm,rate=sf.read(p,dtype='int16')
 assert rate==16000 and pcm.ndim==1 and len(pcm)==r['source_samples'] and np.any(pcm)
 # Source clipping is reported separately; renderer enforces final headroom.
 dry[int(r['voice']),r['length_multiplier'],r['target']]=len(pcm);source_count+=1
records=json.loads((factor/'report.json').read_text())['records'];lookup={(int(r['voice']),r['length_multiplier'],r['target'],r['arm']):r for r in records}
voices=sorted({int(r['voice']) for r in records});assert voices==sorted(np.random.default_rng(726).permutation(904)[:672].tolist())
phrases=['hey_nyx','hey_nick','hey_next','hey_mix'];arms=['quiet','noise25','combined','noise15','rir'];groups={}
for voice in voices:
 for speed in [.9,1.,1.1]:
  lengths=[dry[voice,speed,p] for p in phrases]
  newlow=max((32000+n+960+159)//160-1 for n in lengths)
  rr=[lookup[voice,speed,p,a] for p in phrases for a in arms]
  oldlow=max(r['feature_crop_end'] for r in rr);oldhigh=min(350,*(r['raw_shape'][0]-1 for r in rr))
  assert oldlow>=newlow,(voice,speed,oldlow,newlow)
  assert ((oldlow+2)//3)*3<=oldhigh
  groups[voice,speed]=(oldlow,oldhigh,newlow)
traces=json.loads((base/'paired-consistency-init-gradient-probe-v1/gradient-probe.json').read_text())['pair_trace']
old_support=Path('research/wakeword_p4/results/4090-matched-noise-full-support-v1.json')
old=json.loads(old_support.read_text())
formal=base/'model-matched-noise-independent-formal-v1'
trace_path=formal/'paired-sampling-trace.jsonl'
training=json.loads((formal/'training.json').read_text())
assert sha(trace_path)==training['paired_rendering']['sampling_trace_sha256']
trace_file=trace_path.open()
rng=np.random.default_rng(227726);chain=hashlib.sha256();seen=set()
seed_chain=hashlib.sha256();seed_count=0
for step in range(120000):
 speed=float(rng.choice([.9,1.,1.1]));vv=[int(v) for v in rng.choice(voices,size=2,replace=False)]
 low=max(groups[v,speed][0] for v in vv);high=min(groups[v,speed][1] for v in vv)
 end=int(rng.choice(np.arange(((low+2)//3)*3,high+1,3,dtype=np.int64)))
 actual=json.loads(next(trace_file))
 assert actual['block']==step and actual['voices']==vv and actual['speed']==speed and actual['endpoint_inclusive']==end
 for v in vv:
  assert end>=groups[v,speed][2];seen.add((v,speed))
  # Match renderer buffer geometry, including trailing silence, for every scheduled draw.
  size=32000+max(dry[v,speed,p]+960 for p in phrases)+16000
  assert end>=149 and end*160+400<=size
  for phrase in phrases:
   domain=f'per-draw-noise-v1:{step}:{v}:{speed}:{phrase}'
   seed=int(hashlib.sha256(domain.encode()).hexdigest()[:16],16)
   seed_chain.update(f'{seed}\n'.encode());seed_count+=1
 if step<50:
  t=traces[step];assert vv==t['voices'] and speed==t['speed'] and end==t['endpoint_inclusive']
 chain.update((json.dumps({'step':step,'voices':vv,'speed':speed,'endpoint':end},sort_keys=True)+'\n').encode())
assert next(trace_file,None) is None
assert chain.hexdigest()==old['schedule_sha256']
r={'authenticated_formal_trace_sha256':sha(trace_path),'prior_geometry_receipt_sha256':sha(old_support),'per_draw_seed_count':seed_count,'per_draw_seed_schedule_sha256':seed_chain.hexdigest(),'runtime_renderer_sha256':sha(Path('research/wakeword_p4/training/per_draw_noise_sampler.py')),'script_sha256':sha(Path(__file__)),'source_manifest_sha256':sha(source/'manifest.jsonl'),'source_pcm_files_authenticated':source_count,'all_voice_speed_groups':len(groups),'formal_steps_replayed':120000,'formal_voice_speed_groups_seen':len(seen),'all_original_endpoints_support_complete_source_and_rir':True,'first50_schedule_equal_original':True,'schedule_sha256':chain.hexdigest(),'optimizer_updates':0,'limits':'Metadata geometry and source PCM format/hash audit; all120k schedules and seed domains checked, but no full120k waveform/frontend rendering or model inference. Not phonetic truth, heldout or acceptance evidence.'}
Path('research/wakeword_p4/results/4090-per-draw-noise-full-support-v1.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
