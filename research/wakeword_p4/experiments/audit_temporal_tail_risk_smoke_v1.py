"""Authenticate matched fresh two-update integration, not model acceptance."""
import hashlib,json
from pathlib import Path
import numpy as np
base=Path('.local/wakeword');results=Path('research/wakeword_p4/results');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
roots={arm:base/f'model-temporal-tail-risk-{arm}-smoke-v1' for arm in ('control','treatment')}
reports={arm:json.loads((root/'training.json').read_text()) for arm,root in roots.items()}
init=[json.loads((root/'actual-initialization-inventory.json').read_text()) for root in roots.values()]
assert init[0]==init[1]==json.loads((base/'model-phrase-ranking-control-smoke-v1/actual-initialization-inventory.json').read_text())
traces={arm:[json.loads(line) for line in (root/'paired-sampling-trace.jsonl').read_text().splitlines()] for arm,root in roots.items()}
assert traces['control']==traces['treatment'] and len(traces['control'])==2
old=[json.loads(line) for line in (base/'model-phrase-ranking-control-smoke-v1/paired-sampling-trace.jsonl').read_text().splitlines()]
assert [r['main_tensor_sha256'] for r in traces['control']]==[r['main_tensor_sha256'] for r in old]
losses={arm:[json.loads(line) for line in (root/'paired-losses.jsonl').read_text().splitlines()] for arm,root in roots.items()}
for key,value in losses['control'][0].items():
 if key!='total':assert value==losses['treatment'][0][key],key
probe=base/'temporal-tail-risk-network-preflight-v1'
pre=json.loads((probe/'report.json').read_text());assert sha(probe/'arrays.npz')==pre['arrays_sha256']
with np.load(probe/'arrays.npz',allow_pickle=False) as z:expected=z['metrics'][0]
np.testing.assert_allclose([losses['control'][0][k] for k in ('pair_bce','ranking','temporal_risk')],expected,rtol=1e-5,atol=1e-5)
for trace,original in zip(traces['control'],pre['traces'][:2],strict=True):
 assert {k:v for k,v in trace.items() if k!='main_tensor_sha256'}==original
for arm,report in reports.items():
 assert report['steps']==2 and report['paired_rendering']['optimizer_updates']==2
 assert report['paired_rendering']['objective']=='temporal-tail-risk'
 assert report['paired_rendering']['temporal_coefficient']==(0 if arm=='control' else 1)
 assert report['batchnorm_initial']==report['batchnorm_selected']
 audit=json.loads((results/f'4090-temporal-tail-risk-{arm}-smoke-v1-export-audit.json').read_text())
 assert audit['all_saved_weights_finite'] and audit['frozen_bn_unchanged']
 assert audit['input_shape']==[1,3,40] and audit['output_shape']==[1,1]
 assert audit['input_dtype']=='int8' and audit['output_dtype']=='uint8'
 assert audit['model_sha256']==sha(roots[arm]/'hey_nyx.tflite')
 (results/f'4090-temporal-tail-risk-{arm}-smoke-v1-training.json').write_bytes((roots[arm]/'training.json').read_bytes())
out={'initial_inventory_exact':True,'ordered_two_step_draws_exact':True,'main_draws_equal_original_smoke':True,
 'first_observations_exact_except_total':True,'first_pair_losses_match_original_zero_update_probe':True,
 'sampler_extension':'Added first50 trace retention only since original probe; original windows replayed exactly in actual smoke.',
 'first_observations':{arm:rows[0] for arm,rows in losses.items()},
 'training_sha256':{arm:sha(root/'training.json') for arm,root in roots.items()},
 'script_sha256':sha(Path(__file__)),'limits':'Fresh two-update engineering smoke only. No quality gates, new dev/final, formal training or promotion.'}
(results/'4090-temporal-tail-risk-smoke-comparison-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
