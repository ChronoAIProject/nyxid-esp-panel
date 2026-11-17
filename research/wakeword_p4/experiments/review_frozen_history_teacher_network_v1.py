"""Independent saved-probability/gradient and exit receipt review."""
import hashlib,json,gzip,collections
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=R/'4090-frozen-history-teacher-network-original-v1.json';d=json.loads(p.read_text());m=json.loads((R/'4090-frozen-history-teacher-metadata-math-preflight-v1.json').read_text())['inventory'];assert sha(d['trace']['path'])==d['trace']['sha256'];summary={};false=[]
 with np.load(d['trace']['path']) as z:
  for r in d['census']:
   row=m[r['index']];assert r['source_id']==row['source']['id'];support=(3*np.arange(1,row['frames']//3+1)+2)*160;end=row['prefix_samples']+row['reference_samples'];label=row['source']['label'];mask=(support>=end+480)&(support<=end+4320) if label else support>row['prefix_samples'];valid=np.convolve(mask.astype(int),np.ones(5,dtype=int),'valid')==5;g=summary.setdefault(str(label),collections.Counter());g['rows']+=1
   for h,v in r['arms'].items():
    a=z[f'p_{r["index"]}_{h}'];five=np.convolve(a.astype(np.float64),np.ones(5)/5,'valid');assert float(five[valid].max())==v['max_five'];assert bool(np.any(five[valid]>=.91796875))==v['crossing'];assert float(a[mask].mean())==v['mean_probability'];g[h+'_crossings']+=v['crossing'];g[h+'_mean_contradictions']+=v['mean_contradicts_label']
    if not label and v['crossing']:false.append(dict(index=r['index'],source=row['source'],history=h,max_five=v['max_five']))
  for r in d['probes']:
   s=z[f'gS_{r["step"]}'].astype(np.float64);k=z[f'gD_{r["step"]}'].astype(np.float64);assert np.isclose(np.linalg.norm(s),r['supervised_norm'],rtol=1e-12);assert np.isclose(np.linalg.norm(k),r['teacher_kl_norm'],rtol=1e-12);assert np.isclose(np.dot(s,k),r['dot'],rtol=1e-12,atol=1e-15)
 step=R/'4090-frozen-history-teacher-step-original-v1.json';st=json.loads(step.read_text());assert st['all_passed'] and st['teacher_unchanged'] and st['teacher_gradient_disconnected'] and [r['optimizer_updates'] for r in st['records']]==[1,1] and st['saved_model_checkpoints']==0
 receipts=[]
 for name,session in [('network',83789),('step',91460)]:
  log=Path(f'/tmp/frozen-history-teacher-{name}-v1.log');dest=R/f'4090-frozen-history-teacher-{name}-execution-v1.log.gz';dest.write_bytes(gzip.compress(log.read_bytes(),mtime=0));receipts.append(dict(stage=name,actual_exit_code=0,session_id=session,log_sha256=sha(log),compressed_log_sha256=sha(dest)))
 out=dict(all_passed=True,summary=summary,teacher_negative_crossings=false,opposing_gradient_batches=sum(r['dot']<0 for r in d['probes']),gradient_batches=50,original_network_sha256=sha(p),original_step_sha256=sha(step),receipts=receipts,limits='Saved4320source probabilities and50component gradients authenticated/recomputed. Clone steps prove wiring only, not formal fit or quality. Six teacher false-crossings remain, no source filtering or coefficient rescue.')
 (R/'4090-frozen-history-teacher-network-review-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(summary=summary,opposing_gradient_batches=out['opposing_gradient_batches'],all_passed=True)))
if __name__=='__main__':main()
