"""Actual trainer network tensors versus independently constructed first50 windows."""
import hashlib,json
from pathlib import Path
import numpy as np
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=Path('.local/wakeword');dest=Path('research/wakeword_p4/results');inventories=[];traces={};losses={};inputs={};receipts={}
with np.load(base/'paired-consistency-init-gradient-probe-v1/probe-batches.npz',allow_pickle=False) as z:original={k:z[k] for k in z.files}
for mode in ['fixed','per_draw']:
 root=base/f'model-per-draw-noise-{mode}-smoke-v1'
 r=json.loads((root/'training.json').read_text());assert r['steps']==2 and r['paired_rendering']['optimizer_updates']==2
 assert r['paired_rendering']['coefficient']==1 and r['paired_rendering']['draw_noise_condition']==mode
 assert r['batchnorm_initial']==r['batchnorm_selected']
 inventories.append(json.loads((root/'actual-initialization-inventory.json').read_text()))
 traces[mode]=[json.loads(l) for l in (root/'paired-sampling-trace.jsonl').read_text().splitlines()];assert len(traces[mode])==2
 losses[mode]=[json.loads(l) for l in (root/'paired-losses.jsonl').read_text().splitlines()];assert len(losses[mode])==2
 for l in losses[mode]:assert all(np.isfinite(v) for v in l.values())
 with np.load(base/f'per-draw-noise-rendering-preflight-v1/{mode}-windows.npz',allow_pickle=False) as z:expected=z['x']
 inputs[mode]=[]
 for step in [1,2]:
  p=root/f'actual-network-input-step{step}.npz'
  with np.load(p,allow_pickle=False) as z:batch={k:z[k] for k in z.files}
  np.testing.assert_array_equal(batch['pair_x'],expected[step-1])
  for k in ['main_x','main_y','main_weight','pair_y','pair_weight']:np.testing.assert_array_equal(batch[k],original[k][step-1])
  assert hashlib.sha256(batch['pair_x'].tobytes()).hexdigest()==traces[mode][step-1]['network_tensor_sha256']
  inputs[mode].append(batch)
 export=json.loads((dest/f'4090-per-draw-noise-{mode}-smoke-v1-export-audit.json').read_text())
 assert export['model_sha256']==sha(root/'hey_nyx.tflite') and export['all_saved_weights_finite'] and export['frozen_bn_unchanged']
 receipts[mode]={'training_sha256':sha(root/'training.json'),'model_sha256':sha(root/'hey_nyx.tflite'),'model_bytes':(root/'hey_nyx.tflite').stat().st_size,'losses':losses[mode],'input_files':[{'name':f'actual-network-input-step{s}.npz','sha256':sha(root/f'actual-network-input-step{s}.npz')} for s in [1,2]]}
 (dest/f'4090-per-draw-noise-{mode}-smoke-v1-training.json').write_bytes((root/'training.json').read_bytes())
assert inventories[0]==inventories[1]
for x,y in zip(traces['fixed'],traces['per_draw']):
 for k in ['block','voices','speed','endpoint_inclusive','pack_indices','original_window_sha256','main_tensor_sha256']:assert x[k]==y[k],k
assert losses['fixed'][0]['main_bce']==losses['per_draw'][0]['main_bce']
assert any(np.any(x['pair_x']!=y['pair_x']) for x,y in zip(inputs['fixed'],inputs['per_draw']))
r={'script_sha256':sha(Path(__file__)),'fresh_optimizer_updates_per_condition':2,'initialization_inventory_exact':True,'source_draws_and_original_windows_exact':True,'actual_network_pair_tensors_equal_independent_offline_render':True,'main_labels_weights_equal_original_first2':True,'first_main_bce_exact':True,'paired_network_inputs_differ':True,'receipts':receipts,'limits':'Two-update integration and export-size smoke only. Not quality acceptance, numerical streaming parity or development evaluation. No promotion.'}
(dest/'4090-per-draw-noise-smoke-comparison-v1.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
