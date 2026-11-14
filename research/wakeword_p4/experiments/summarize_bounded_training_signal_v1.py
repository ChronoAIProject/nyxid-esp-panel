"""Complete fixed-run training signal accounting; descriptive, no inference/tuning."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');P='4090-hey-broad-bounded-consistency-'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summary(rows):
 out={'steps':len(rows)}
 for k in ['supervised_loss','raw_consistency_loss','bounded_consistency_loss','supervised_gradient_l2','raw_consistency_gradient_l2','bounded_consistency_gradient_l2']:
  x=np.array([r[k] for r in rows],np.float64);assert np.isfinite(x).all() and (x>=0).all();out[k]=dict(zero_steps=int(np.sum(x==0)),mean=float(x.mean()),median=float(np.median(x)),maximum=float(x.max()))
 c=np.array([r['bounded_consistency_loss'] for r in rows]);g=np.array([r['bounded_consistency_gradient_l2'] for r in rows]);out['bounded_positive_loss_zero_gradient_steps']=int(np.sum((c>0)&(g==0)));out['bounded_zero_loss_nonzero_gradient_steps']=int(np.sum((c==0)&(g>0)));return out
def main():
 rp=R/(P+'formal-result-v1.json');ap=R/(P+'formal-independent-audit-v1.json');result=json.loads(rp.read_text());audit=json.loads(ap.read_text());assert audit['result_sha256']==sha(rp) and len(audit['cases'])==len(result['cases'])==4;cases=[];files={str(p):sha(p) for p in [Path(__file__).resolve().relative_to(Path.cwd()),rp,ap]}
 for c in result['cases']:
  receipt=c['final_repeat'];p=Path(receipt['weights_path']).parent/'actual-batch-schedule.jsonl';assert sha(p)==receipt['batch_trace_sha256'];files[str(p)]=sha(p);rows=[json.loads(l) for l in p.read_text().splitlines()];assert len(rows)==10000 and [r['step'] for r in rows]==list(range(1,10001))
  intervals=[dict(first_step=i+1,last_step=i+1000,**summary(rows[i:i+1000])) for i in range(0,10000,1000)];cases.append(dict(target=c['target'],condition=c['condition'],all10000=summary(rows),fixed1000step_intervals=intervals))
 out=dict(files=files,cases=cases,model_invocations=0,optimizer_updates=0,limits='All40000 original logged pre-update batches, both C losses/backward norms were explicitly observed. Ten fixed1000step intervals and whole-run statistics are descriptive only; not fixed-probe final-state measurements, model repeat proof, parameter selection, recognition/FAR or causal gradient efficacy. Preserve all zero/sparse signals and original numerical rejection; do not adapt lambda/sampler/checkpoint or reuse fresh data to rescue this run.')
 with (R/'4090-bounded-consistency-complete-training-signal-v1.json').open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
 print(json.dumps([dict(target=c['target'],condition=c['condition'],zeroC=c['all10000']['bounded_consistency_loss']['zero_steps'],zeroGradient=c['all10000']['bounded_consistency_gradient_l2']['zero_steps'],positiveCzeroGradient=c['all10000']['bounded_positive_loss_zero_gradient_steps']) for c in cases]))
if __name__=='__main__':main()
