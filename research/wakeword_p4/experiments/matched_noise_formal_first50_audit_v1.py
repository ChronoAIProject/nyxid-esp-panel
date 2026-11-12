"""Bounded immutable-prefix audit of a live formal run, not full completion."""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--condition',choices=['independent','common'],required=True);a=p.parse_args()
base=Path('.local/wakeword');root=base/f'model-matched-noise-{a.condition}-formal-v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with (root/'paired-sampling-trace.jsonl').open() as f:traces=[json.loads(l) for l in itertools.islice(f,50)]
with (root/'paired-losses.jsonl').open() as f:losses=[json.loads(l) for l in itertools.islice(f,50)]
assert len(traces)==len(losses)==50
ref=base/'paired-consistency-init-gradient-probe-v1';probe=json.loads((ref/'gradient-probe.json').read_text())
assert sha(ref/'probe-batches.npz')=='7d781dfc435417632aaa2c8995d6a955772b29b3ca3f88fbfd87d57108e994cb'
mr=base/'matched-noise-rendering-preflight-v1';meta=json.loads((mr/'report.json').read_text());item=meta['conditions'][a.condition]
assert sha(Path(item['path']))==item['sha256']
with np.load(ref/'probe-batches.npz',allow_pickle=False) as z, np.load(item['path'],allow_pickle=False) as new:
 for i,(trace,old,loss) in enumerate(zip(traces,probe['pair_trace'],losses)):
  for k in ['block','voices','speed','endpoint_inclusive','pack_indices']:assert trace[k]==old[k],(i,k)
  assert trace['original_window_sha256']==old['window_sha256']
  assert trace['noise_condition']==a.condition and trace['block']==i
  main=hashlib.sha256(b''.join(z[k][i].tobytes() for k in ['main_x','main_y','main_weight'])).hexdigest()
  assert main==trace['main_tensor_sha256']
  assert hashlib.sha256(new['x'][i].tobytes()).hexdigest()==trace['network_tensor_sha256']
  assert loss['step']==i+1 and all(np.isfinite(v) for v in loss.values())
 for step in [1,2]:
  with np.load(root/f'actual-network-input-step{step}.npz',allow_pickle=False) as actual:
   for k in ['main_x','main_y','main_weight','pair_y','pair_weight']:np.testing.assert_array_equal(actual[k],z[k][step-1])
   np.testing.assert_array_equal(actual['pair_x'],new['x'][step-1])
smoke=base/f'model-matched-noise-{a.condition}-smoke-v1'
assert (root/'actual-initialization-inventory.json').read_bytes()==(smoke/'actual-initialization-inventory.json').read_bytes()
r={'condition':a.condition,'script_sha256':sha(Path(__file__)),'prefix_batches':50,'all_original_source_draws_and_windows_exact':True,'all_main_tensor_hashes_equal_authenticated_original':True,'all_pair_tensor_hashes_equal_offline_frontend':True,'saved_actual_inputs_steps1_2_exact':True,'initialization_inventory_equal_fresh_smoke':True,'consecutive_finite_loss_rows':50,'trace_prefix_sha256':hashlib.sha256((json.dumps(traces,sort_keys=True)).encode()).hexdigest(),'loss_prefix_sha256':hashlib.sha256((json.dumps(losses,sort_keys=True)).encode()).hexdigest(),'formal_completion_claim':False,'limits':'Immutable first50 prefix only of live formal execution; no quality, later-step or full-run claim. No newdev/final data.'}
out=Path(f'research/wakeword_p4/results/4090-matched-noise-{a.condition}-formal-first50-audit-v1.json');assert not out.exists();out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
