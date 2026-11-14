"""Compare preserved full-coordinate backend probes; no new inference."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 paths=[R/f'4090-command-crop-phase{s}-result-v1.json' for s in ('','-cpu')]
 gpu,cpu=[json.loads(p.read_text()) for p in paths]
 assert gpu['requests']==cpu['requests']==12510 and gpu['weights_unchanged'] and cpu['weights_unchanged'] and cpu['visible_gpus']==[]
 gaps={k:[] for k in ('nominal','phase')}
 for a,b in zip(gpu['rows'],cpu['rows'],strict=True):
  assert all(a[k]==b[k] for k in ('step','slot','new_order','new_end','stream'))
  for r in (a,b):
   for k in gaps:assert abs(abs(r[k]-r['stream'])-r[k+'_abs'])<1e-12
  for k in gaps:gaps[k].append(abs(a[k]-b[k]))
 report=dict(files={str(p):sha(p) for p in paths},coordinates=12510,same_saved_stream_exact=True,weights_unchanged=True,cpu_only=True,gpu_minus_cpu_abs={k:{str(q):float(np.quantile(v,q)) for q in (.5,.95,.99,1)} for k,v in gaps.items()},cpu_nominal_abs=cpu['nominal_abs'],cpu_phase_abs=cpu['phase_abs'],gpu_nominal_abs=gpu['nominal_abs'],gpu_phase_abs=gpu['phase_abs'],cpu_phase_within_original_1e_4=max(r['phase_abs'] for r in cpu['rows'])<=1e-4,cpu_nominal_low_stream_high=sum(r['nominal']<=.5 and r['stream']>=.9 for r in cpu['rows']),threshold_selected=False,limits='Fixedfinal sampled-coordinate numerical diagnosis only. CPU TF vs CPU LiteRT remains different software kernels; GPU differences are backend arithmetic evidence, not proof of a unique TF32 cause. Does not establish new training efficacy, recognition/FAR or INT8 qualification; freshdev/acceptance closed.')
 with (R/'4090-command-crop-phase-backend-comparison-v1.json').open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
