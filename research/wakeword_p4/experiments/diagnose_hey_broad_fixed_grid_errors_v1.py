"""Descriptive saved-array error localization; never changes any gate or scores new data."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(x):
 return dict(count=int(x.size),mean=float(np.mean(x)),p99=float(np.quantile(x,.99)),maximum=float(np.max(x))) if x.size else None
def main():
 p=R/'4090-hey-broad-fixed-grid-screen-result-v1.json';old=R/'4090-hey-broad-paired-qat-screen-result-v1.json'
 result=json.loads(p.read_text());prior=json.loads(old.read_text());oldcases={(c['target'],c['condition'],c['request_order']):c for c in prior['cases']};cases=[];summary={}
 for c in result['cases']:
  key=(c['target'],c['condition'],c['request_order']);o=oldcases[key]
  assert sha(c['arrays_path'])==c['arrays_sha256']; assert sha(o['arrays_path'])==o['arrays_sha256']
  with np.load(c['arrays_path'],allow_pickle=False) as a,np.load(o['arrays_path'],allow_pickle=False) as b:
   assert np.array_equal(a['float'],b['float']) and np.array_equal(a['raw'],b['raw'])
   scale,zero=c['output_quantization'];error=(a['uint8'].astype(np.float64)-zero)*scale-a['float'];five=np.abs(np.convolve(error,np.ones(5)/5,'valid'));at=int(np.argmax(five))
   # Classify by the last constituent step. First fully supported phase is index74.
   last=np.arange(len(five))+4;startup=five[last<74];supported=five[last>=74]
   first_bad=np.flatnonzero(five>.1)
   row=dict(target=c['target'],condition=c['condition'],request_order=c['request_order'],passed=c['passed'],prior_ptq_passed=o['passed'],startup_last_step_lt74=stats(startup),last_step_ge74=stats(supported),worst_five_start_step=at,worst_five_end_step=at+4,worst_five_sample_end=int(a['sample_end'][at+4]),worst_float=a['float'][at:at+5].tolist(),worst_uint8=a['uint8'][at:at+5].tolist(),first_five_error_above_point1_end_step=int(first_bad[0]+4) if len(first_bad) else None,output_zero_codes=int(np.sum(a['uint8']==0)),output_255_codes=int(np.sum(a['uint8']==255)))
  cases.append(row);name=c['condition']+'/'+c['target'];s=summary.setdefault(name,dict(cases=0,failed=0,prior_ptq_failed=0,failed_with_supported_five_above_point1=0,failed_with_startup_five_above_point1=0,prior_failed_now_passed=0,prior_passed_now_failed=0));s['cases']+=1;s['failed']+=not c['passed'];s['prior_ptq_failed']+=not o['passed'];s['failed_with_supported_five_above_point1']+=bool(not c['passed'] and supported.max()>.1);s['failed_with_startup_five_above_point1']+=bool(not c['passed'] and startup.max()>.1);s['prior_failed_now_passed']+=bool(not o['passed'] and c['passed']);s['prior_passed_now_failed']+=bool(o['passed'] and not c['passed'])
 assert len(cases)==540
 dest=R/'4090-hey-broad-fixed-grid-error-localization-v1.json'
 with dest.open('x') as f:json.dump(dict(result_sha256=sha(p),prior_ptq_sha256=sha(old),summary=summary,cases=cases,limits='Descriptive retrospective localization, no gate or startup exclusion. Every fullwave gate unchanged. Partition at existing first full-support index74 is diagnostic only. Exact float/raw identity verified between original PTQ and fixedgrid. Does not establish cause, recognition/FAR or newdev admission.'),f,indent=2);f.write('\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
