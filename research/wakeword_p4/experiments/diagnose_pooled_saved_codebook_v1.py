"""Saved-output alphabet geometry only; no model invoke, export or gate changes."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(error):
 e=np.abs(np.convolve(error,np.ones(5)/5,'valid'));return dict(mean=float(e.mean()),p99=float(np.percentile(e,99)),maximum=float(e.max()))
def failed(d):return d['mean']>.02 or d['p99']>.1 or d['maximum']>.25
def main():
 source=R/'4090-pooled-matched-screen-result-v1.json';r=json.loads(source.read_text());rows=[];summary=[]
 for target,condition in sorted({(c['target'],c['condition']) for c in r['cases']}):
  cases=[c for c in r['cases'] if (c['target'],c['condition'])==(target,condition)];assert len(cases)==135
  codebook=set();arrays=[];quant=None
  for c in cases:
   assert sha(c['arrays_path'])==c['arrays_sha256']
   with np.load(c['arrays_path'],allow_pickle=False) as a:
    codebook.update(map(int,a['uint8']));arrays.append((c,a['float'].copy(),a['uint8'].copy()))
   q=tuple(c['output_quantization'])
   if quant is None:quant=q
   else:assert quant==q
  codes=np.array(sorted(codebook),np.uint8);scale,zero=quant;values=(codes.astype(float)-zero)*scale;group=[]
  for c,reference,actual in arrays:
   # Choose individually observed output symbols; sequence is hypothetical,
   # with no claim that the actual internal state trajectory can realize it.
   nearest=values[np.argmin(np.abs(reference[:,None]-values[None,:]),axis=1)]
   projected=stats(nearest-reference);observed=stats((actual.astype(float)-zero)*scale-reference)
   assert observed==c['smoothed_abs_error']
   row=dict(target=target,condition=condition,request_order=c['request_order'],actual=observed,hypothetical_nearest_observed_code=projected,actual_error_failed=failed(observed),hypothetical_error_failed=failed(projected),arrays_sha256=c['arrays_sha256']);rows.append(row);group.append(row)
  summary.append(dict(target=target,condition=condition,observed_codes=codes.tolist(),code_count=len(codes),output_quantization=list(quant),actual_error_failed=sum(x['actual_error_failed'] for x in group),hypothetical_error_failed=sum(x['hypothetical_error_failed'] for x in group),worst_hypothetical={k:max(x['hypothetical_nearest_observed_code'][k] for x in group) for k in ['mean','p99','maximum']}))
 with (R/'4090-pooled-saved-codebook-diagnostic-v1.json').open('x') as f:f.write(json.dumps(dict(source_sha256=sha(source),script_sha256=sha(__file__),summary=summary,cases=rows,model_invocations=0,limits='Descriptive geometry of actually observed UINT8 alphabet. Per-step nearest-code hypothetical sequence is NOT actual model inference, a realizable state trajectory, candidate requalification or a unique cause. Original actual540 gates/rejections unchanged. No head/threshold change, training, export or newdata.'),indent=2)+'\n')
 print(json.dumps([{k:v for k,v in x.items() if k!='observed_codes'} for x in summary]))
if __name__=='__main__':main()
