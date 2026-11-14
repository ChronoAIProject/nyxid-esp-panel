"""Read-only complete trace/manifests/clock/input identity audit; no inference."""
import hashlib,json
from pathlib import Path
import numpy as np
from collect_fp32_reused_development_v1 import R,LOCK,OUT,sha
def main():
 p=json.loads(LOCK.read_text());result=json.loads((OUT/'result.json').read_text())
 assert result['all_passed'] and result['execution_sha256']==sha(LOCK) and len(result['models'])==4
 manifests={phase:json.loads(Path(m['path']).read_text())['records'] for phase,m in p['materials'].items()}
 shared={};models=[]
 for model in p['models']:
  name=model['target']+'-'+model['condition'];folder=OUT/name;rp=folder/'report.json'
  expected=next(m for m in result['models'] if m['model']==name);assert sha(rp)==expected['report_sha256']
  report=json.loads(rp.read_text());records=[json.loads(x) for x in (folder/'records.jsonl').read_text().splitlines()]
  assert records==report['records'] and len(records)==372 and report['model']==model and report['execution_sha256']==sha(LOCK)
  position=0;steps=0;hours=0.
  for phase,rows in manifests.items():
   for i,row in enumerate(rows):
    rec=records[position];position+=1;assert rec['model']==name and rec['phase']==phase and rec['index']==i and rec['record']==row and rec['repeat_exact']
    trace=folder/phase/f'{i:06}.npz';assert Path(rec['trace_path'])==trace and sha(trace)==rec['trace_sha256']
    with np.load(trace,allow_pickle=False) as a:
     assert set(a.files)=={'probability','probability_repeat','sample_end','support_unclamped'}
     raw=a['probability'];n=(max(0,(row['samples']+159)//160-2)+2)//3
     assert raw.shape==(n,) and raw.dtype==np.float32 and raw.tobytes()==a['probability_repeat'].tobytes()
     assert np.isfinite(raw).all() and np.all((raw>=0)&(raw<=1)) and n==rec['steps']
     support=(3*np.arange(1,n+1,dtype=np.int64)+2)*160
     assert np.array_equal(a['support_unclamped'],support) and np.array_equal(a['sample_end'],np.minimum(support,row['samples']))
    identity=(rec['input_sha256'],rec['raw_frontend_sha256'],n,row['samples'],row['pcm_sha256'])
    key=(phase,i)
    if key in shared:assert shared[key]==identity
    else:shared[key]=identity
    steps+=n
    if phase=='background':hours+=row['samples']/16000/3600
  models.append(dict(model=name,complete_inputs=position,steps_per_repeat=steps,background_hours=hours,report_sha256=sha(rp),records_sha256=sha(folder/'records.jsonl')))
 assert len(shared)==372
 with (R/'4090-fp32-reused-collection-independent-audit-v1.json').open('x') as f:f.write(json.dumps(dict(execution_sha256=sha(LOCK),models=models,shared_inputs=372,all_passed=True,limits='Saved original1488 traces/manifest records/hash/clocks/repeats/probability-domain audit only. Shared frontend/input identity hashes checked across models, no PCM/frontend recomputation or model invocation. No recognition scores or newdata.'),indent=2)+'\n')
 print('All1488 complete raw FP32 traces independently audited')
if __name__=='__main__':main()
