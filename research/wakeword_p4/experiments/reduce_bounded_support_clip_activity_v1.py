"""Saved-array clipping activity; no inference or measured autodiff claim."""
import hashlib
import json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
    rp=R/'4090-bounded-training-support-forward-result-v1.json';ap=R/'4090-bounded-training-support-forward-independent-audit-v1.json'
    d=json.loads(rp.read_text());assert json.loads(ap.read_text())['result_sha256']==sha(rp)
    rows=[]
    for c in d['cases']:
        assert sha(c['arrays_path'])==c['arrays_sha256']
        with np.load(c['inputs']['path']) as z:labels=z['y']
        with np.load(c['arrays_path']) as z:
            for mode in ('nominal','phase'):
                f=z[mode+'_r0_float_logit'];q=z[mode+'_r0_raw_qat_logit'];fc=z[mode+'_r0_exact_fc']
                bf=np.clip(f,-8,8);preq=(fc.astype(np.float32)-3)*np.float32(8/3);bq=np.clip(preq,-8,8)
                gap=bf.astype(float)-bq.astype(float);nonzero=gap!=0
                # Inclusive TensorFlow clip gradient; downstream C seed only.
                float_clip_active=(f>=-8)&(f<=8)
                quant_coordinate_active=(preq>=-8)&(preq<=8)
                same=((f>=8)&(q>=8))|((f<=-8)&(q<=-8))
                groups=[]
                for label in (0,1):
                    mask=labels==label
                    groups.append(dict(label=label,examples=int(mask.sum()),nonzero_bounded_gaps=int(np.sum(mask&nonzero)),nonzero_gap_float_clip_inactive=int(np.sum(mask&nonzero&~float_clip_active)),nonzero_gap_both_coordinate_clips_inactive=int(np.sum(mask&nonzero&~float_clip_active&~quant_coordinate_active)),nonzero_gap_both_raw_same_side_saturated=int(np.sum(mask&nonzero&same))))
                rows.append(dict(target=c['target'],condition=c['condition'],mode=mode,groups=groups))
    out=dict(result_sha256=sha(rp),audit_sha256=sha(ap),reducer_sha256=sha(__file__),rows=rows,
             limits='Saved forward values with analytic output-coordinate clipping activity only. No autodiff, per-parameter gradients, scalarFC accumulator masks, encoder STE masks or gradient cancellation measured. Active output coordinate cannot prove useful nonzero parameter gradient. Zero C and raw same-side saturation do not imply label error. Diagnostic, no training change, gate relaxation or promotion.')
    with (R/'4090-bounded-support-clip-activity-v1.json').open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
    print(json.dumps(rows))
if __name__=='__main__':run()
