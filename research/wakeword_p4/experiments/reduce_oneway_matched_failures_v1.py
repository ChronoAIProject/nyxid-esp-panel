"""Descriptive all-request matched reduction; no selection or model invocation."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 source=R/'4090-hey-broad-oneway-consistency-fixed-grid-screen-result-v1.json'
 data=json.loads(source.read_text());assert len(data['cases'])==540
 rows={(c['target'],c['condition'],c['request_order']):c for c in data['cases']};assert len(rows)==540
 output=[]
 for target in ['hey_willow','hey_nyx']:
  control={k[2]:v for k,v in rows.items() if k[:2]==(target,'twoway_consistency')}
  treatment={k[2]:v for k,v in rows.items() if k[:2]==(target,'oneway_consistency')}
  assert set(control)==set(treatment) and len(control)==135
  transitions={};differences={};failures=[]
  for order,a in control.items():
   b=treatment[order];key=('pass' if a['passed'] else 'fail')+'->'+('pass' if b['passed'] else 'fail');transitions.setdefault(key,[]).append(order)
   for metric in ['mean','p99','maximum']:
    differences.setdefault(metric,[]).append(b['smoothed_abs_error'][metric]-a['smoothed_abs_error'][metric])
   for condition,c in [('twoway_consistency',a),('oneway_consistency',b)]:
    assert sha(c['arrays_path'])==c['arrays_sha256']
    if c['passed']:continue
    with np.load(c['arrays_path'],allow_pickle=False) as z:
     scale,zero=c['output_quantization'];signed=(z['uint8'].astype(np.float64)-zero)*scale-z['float'];five=np.abs(np.convolve(signed,np.ones(5)/5,'valid'))
     assert float(np.quantile(five,.99))==c['smoothed_abs_error']['p99']
     # Window index denotes its first frame; >=74 means all five are after startup.
     indices=np.flatnonzero(five>.1)
     failures.append(dict(condition=condition,request_order=order,phase_max=c['phase_max_error'],five_stats=c['smoothed_abs_error'],windows_above_point1=int(len(indices)),windows_above_point1_after_startup=int(np.sum(indices>=74)),largest_window_start=int(np.argmax(five))))
  output.append(dict(target=target,all135_transitions=transitions,all135_signed_treatment_minus_control={k:dict(improved=int(np.sum(np.array(v)<0)),equal=int(np.sum(np.array(v)==0)),worsened=int(np.sum(np.array(v)>0)),minimum=float(min(v)),median=float(np.median(v)),maximum=float(max(v))) for k,v in differences.items()},all_failed_cases=failures))
 result=dict(source_sha256=sha(source),targets=output,limits='Descriptive complete matched135 per target, repeat1 only; no statistical efficacy/causal mechanism claim. All startup frames retained. >.1 window counts describe tails, not a replacement gate. Consumed numerical witnesses never train/threshold select. Both conditions remain rejected; no fresh development or promotion.')
 p=R/'4090-oneway-matched-failure-reduction-v1.json'
 with p.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 for t in output:print(json.dumps({k:v for k,v in t.items() if k!='all_failed_cases'}))
if __name__=='__main__':main()
