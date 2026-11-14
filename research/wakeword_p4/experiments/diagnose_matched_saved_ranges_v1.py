"""Saved-tensor range geometry; no inference or calibration changes."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 source=R/'4090-matched-internal-boundaries-original-result-v1.json';report=json.loads(source.read_text());rows=[]
 for c in report['cases']:
  assert sha(c['captures_path'])==c['captures_sha256'] and sha(c['metadata_path'])==c['metadata_sha256']
  m=json.loads(Path(c['metadata_path']).read_text());layers=[]
  with np.load(c['captures_path'],allow_pickle=False) as a:
   for key in [f'encoder_{i:02}' for i in range(9)]+['embedding']:
    f=a['fp32_'+key][0].astype('float64');q=a['int8_'+key][0];meta=m['int8'][key];s,=meta['scales'];z,=meta['zeros'];lo=(-128-z)*s;hi=(127-z)*s
    v=(q.astype('float64')-z)*s
    assert f.shape==v.shape and np.isfinite(f).all()
    # Geometric best scalar code approximation of each FP32 activation, not a realizable trajectory.
    best=np.clip(np.rint(f/s)+z,-128,127);best=(best-z)*s
    layers.append(dict(key=key,elements=f.size,steps=len(f),scale=s,representable_min=lo,representable_max=hi,float_min=float(f.min()),float_max=float(f.max()),float_below_range=int((f<lo).sum()),float_above_range=int((f>hi).sum()),actual_low_code=int((q==-128).sum()),actual_high_code=int((q==127).sum()),actual_error_mean=float(abs(v-f).mean()),nearest_code_error_mean=float(abs(best-f).mean()),nearest_code_error_max=float(abs(best-f).max()),per_step_above_range=[int(x) for x in (f>hi).reshape(len(f),-1).sum(axis=1)],per_step_actual_high=[int(x) for x in (q==127).reshape(len(f),-1).sum(axis=1)]))
  rows.append(dict(target=c['target'],condition=c['condition'],request_order=0,layers=layers))
 out=dict(source_report_sha256=sha(source),script_sha256=sha(__file__),model_invocations=0,cases=rows,limits='Four fixed full startup witnesses only. Low ReLU codes include legitimate zero activations; endpoint occupancy is not proof of clipping. Nearest-code geometry is not realizable inference or causal attribution. No calibration, model, threshold or data changes.')
 (R/'4090-matched-saved-range-diagnostic-v1.json').write_text(json.dumps(out,indent=2)+'\n')
 for c in rows:
  print(c['target'],c['condition'])
  for x in c['layers']:
   print(x['key'], 'range',round(x['representable_max'],3),'fmax',round(x['float_max'],3),'above',x['float_above_range'],'high',x['actual_high_code'],'mae',round(x['actual_error_mean'],5),'nearest',round(x['nearest_code_error_mean'],5))
if __name__=='__main__':main()
