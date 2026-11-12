"""Authenticate saved zero-update probes and compare without selecting a condition."""
import hashlib,json
from pathlib import Path
import numpy as np
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=Path('.local/wakeword');reports={};arrays={};summary={}
for mode in ['fixed','per_draw']:
    p=base/f'per-draw-noise-init-gradient-{mode}-v1';r=json.loads((p/'report.json').read_text())
    assert r['all_variables_unchanged'] and r['all_gradients_connected_and_finite'] and r['optimizer_updates']==0
    assert sha(p/'arrays.npz')==r['arrays_sha256']
    with np.load(p/'arrays.npz',allow_pickle=False) as a:arrays[mode]={k:a[k] for k in a.files}
    reports[mode]=r
    Path(f'research/wakeword_p4/results/4090-per-draw-noise-init-gradient-{mode}-v1.json').write_bytes((p/'report.json').read_bytes())
    obs=r['observations'];summary[mode]={'negative_cosine_batches':sum(x['cosine']<0 for x in obs),'median_cosine':float(np.median([x['cosine'] for x in obs])),'median_ranking_supervised_norm_ratio':float(np.median([x['ranking_norm']/x['supervised_norm'] for x in obs])),'mean_main_pair_ranking':arrays[mode]['metrics'].mean(axis=0).tolist()}
a,b=arrays['fixed'],arrays['per_draw'];np.testing.assert_array_equal(a['metrics'][:,0],b['metrics'][:,0])
assert reports['fixed']['snapshot_sha256']==reports['per_draw']['snapshot_sha256']
r={'script_sha256':sha(Path(__file__)),'probes':summary,'main_bce_exact_across_conditions':True,'same_derived_initialization':True,'pair_logits_changed':bool(np.any(a['pair_logits']!=b['pair_logits'])),'pair_logit_absolute_difference_max':float(np.max(np.abs(a['pair_logits']-b['pair_logits']))),'optimizer_updates':0,'limits':'Numerical connectivity and training-input perturbation only. Loss/gradient differences do not establish acceptance or future quality. No condition selection, coefficient fitting, newdev/final or model promotion.'}
Path('research/wakeword_p4/results/4090-per-draw-noise-probe-comparison-v1.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
