"""Join existing raw and bounded captures by immutable model/request identity."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 oldp=R/'4090-hey-broad-paired-qat-screen-result-v1.json'
 newp=R/'4090-fp32-internal-fullstream-original-result-v1.json'
 old=json.loads(oldp.read_text());new=json.loads(newp.read_text())
 ol=R/'4090-hey-broad-paired-qat-screen-execution-v1.json';nl=R/'4090-fp32-internal-fullstream-execution-v1.json'
 assert old['execution_sha256']==sha(ol) and new['execution_sha256']==sha(nl)
 x=json.loads(ol.read_text());y=json.loads(nl.read_text())
 identity=lambda m:(m['target'],m['condition'],m['path'],m['sha256'])
 assert list(map(identity,x['weights']))==list(map(identity,y['models']))
 for m in y['models']:assert sha(m['path'])==m['sha256']
 key=lambda c:(c['target'],c['condition'],c['request_order'])
 originals={key(c):c for c in old['cases']};assert len(originals)==len(new['cases'])==540
 floor=1/(1+np.exp(8.));ceiling=1-floor;rows=[]
 for c in new['cases']:
  o=originals.pop(key(c));assert sha(o['arrays_path'])==o['arrays_sha256'] and sha(c['path'])==c['sha256']
  with np.load(o['arrays_path'],allow_pickle=False) as a,np.load(c['path'],allow_pickle=False) as b:
   for name in ['raw','support_unclamped','sample_end']:
    assert a[name].dtype==b[name].dtype and a[name].shape==b[name].shape and a[name].tobytes()==b[name].tobytes()
   raw=b['tf'].astype(np.float64);bounded=a['float'];assert raw.shape==bounded.shape
   expected=np.clip(raw,floor,ceiling)
   transform_error=float(np.max(np.abs(expected-bounded)))
   gap=float(np.max(np.abs(raw-bounded)))
   scale,zero=o['output_quantization'];integer=(a['uint8'].astype(float)-zero)*scale
   stats={}
   for name,ref in [('raw',raw),('bounded',bounded)]:
    smooth=np.abs(np.convolve(integer-ref,np.ones(5)/5,mode='valid'))
    stats[name]=dict(mean=float(smooth.mean()),p99=float(np.percentile(smooth,99)),maximum=float(smooth.max()))
   assert all(stats['bounded'][k]==o['smoothed_abs_error'][k] for k in stats['bounded'])
   rows.append(dict(target=c['target'],condition=c['condition'],request_order=c['request_order'],steps=len(raw),raw_bounded_max_abs=gap,clip_identity_max_abs=transform_error,clip_identity_within_1e4=transform_error<=1e-4,integer_five_step_error=stats,original_passed=o['passed'],old_arrays_sha256=o['arrays_sha256'],raw_arrays_sha256=c['sha256']))
 assert not originals
 summaries=[]
 for target,condition in sorted({(r['target'],r['condition']) for r in rows}):
  selected=[r for r in rows if (r['target'],r['condition'])==(target,condition)]
  fail=lambda d:d['mean']>.02 or d['p99']>.1 or d['maximum']>.25
  summaries.append(dict(target=target,condition=condition,cases=len(selected),steps=sum(r['steps'] for r in selected),max_raw_bounded=max(r['raw_bounded_max_abs'] for r in selected),max_clip_identity_error=max(r['clip_identity_max_abs'] for r in selected),original_failed=sum(not r['original_passed'] for r in selected),descriptive_raw_error_failed=sum(fail(r['integer_five_step_error']['raw']) for r in selected),descriptive_error_decision_changes=sum(fail(r['integer_five_step_error']['raw'])!=fail(r['integer_five_step_error']['bounded']) for r in selected)))
 report=dict(schema='raw-bounded-paired-saved-v1',source_hashes={str(p):sha(p) for p in [oldp,newp,ol,nl,Path(__file__)]},summaries=summaries,cases=rows,model_invocations=0,limits='Descriptive saved-array comparison only. Original bounded-reference INT8 gates/decisions unchanged. Raw-reference error is diagnostic, not candidate requalification or proof of internal quantization cause. No threshold selection, new data, export or inference.')
 with (R/'4090-raw-bounded-paired-saved-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 print(json.dumps(summaries))
if __name__=='__main__':main()
