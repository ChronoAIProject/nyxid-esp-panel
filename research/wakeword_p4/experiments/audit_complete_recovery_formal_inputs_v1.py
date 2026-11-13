"""Authenticate matched recovery source schedule; pair identical, main intentionally differs."""
import hashlib,json,itertools
from pathlib import Path
base=Path('.local/wakeword');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
roots=[base/f'model-complete-recovery-{m}-formal-v1' for m in ['control','recovered']]
assert (roots[0]/'actual-initialization-inventory.json').read_bytes()==(roots[1]/'actual-initialization-inventory.json').read_bytes()
chain=hashlib.sha256();changed=0
with (roots[0]/'paired-sampling-trace.jsonl').open() as f,(roots[1]/'paired-sampling-trace.jsonl').open() as g:
 for i,(x,y) in enumerate(itertools.zip_longest(f,g)):
  assert x is not None and y is not None
  x,y=json.loads(x),json.loads(y)
  for k in ['block','voices','speed','endpoint_inclusive','pack_indices','original_window_sha256']:assert x[k]==y[k],(i,k)
  assert x['block']==i and x['draw_noise_condition']=='fixed' and y['draw_noise_condition']=='fixed'
  assert x['network_tensor_sha256']==y['network_tensor_sha256'], (i,'paired tensor')
  changed+=x['main_tensor_sha256']!=y['main_tensor_sha256']
  chain.update((json.dumps({'step':i,'voices':x['voices'],'speed':x['speed'],'endpoint':x['endpoint_inclusive']},sort_keys=True)+'\n').encode())
 assert i+1==120000
expected=json.loads(Path('research/wakeword_p4/results/4090-per-draw-noise-full-support-v1.json').read_text())
assert chain.hexdigest()==expected['schedule_sha256']
for mode,root in zip(['control','recovered'],roots):
 r=json.loads((root/'training.json').read_text());p=r['paired_rendering']
 assert r['steps']==120000 and p['optimizer_updates']==120000 and p['coefficient']==1 and p['draw_noise_condition']=='fixed'
 assert r['recovered_erased_complete_coverage'] == (mode=='recovered')
 assert sha(root/'paired-sampling-trace.jsonl')==p['sampling_trace_sha256']
 assert sha(root/'paired-losses.jsonl')==p['loss_trace_sha256']
 with (root/'paired-losses.jsonl').open() as losses:
  for step,line in enumerate(losses,1):
   loss=json.loads(line)
   assert loss['step']==step
   assert all(__import__('math').isfinite(v) for v in loss.values())
  assert step==120000
 assert sha(root/'actual-initialization.weights.h5')==p['initialization_snapshot_sha256']
 assert sha(root/'actual-initialization-inventory.json')==p['initialization_inventory_sha256']
assert changed==120000
out=base/'complete-recovery-formal-input-audit-v1.json';assert not out.exists()
out.write_text(json.dumps({'steps_per_arm':120000,'original_ordered_paired_source_draws_exact':True,'paired_network_inputs_exact':True,'initialization_inventory_exact':True,'schedule_sha256':chain.hexdigest(),'different_main_network_batches':changed,'limits':'Full trace hashes authenticate actual paired tensors and main hashes, not all108 preserved rows at every formal step; row equality is covered by the two-update integration.','acceptance_claim':False},indent=2)+'\n')
