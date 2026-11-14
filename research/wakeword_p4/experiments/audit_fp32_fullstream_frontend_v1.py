"""Recompute only original consumed PCM/frontend, never model inference."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import soundfile as sf

R=Path('research/wakeword_p4/results')
ROOT=Path('.local/wakeword/fp32-internal-fullstream-v1')
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 lock=R/'4090-fp32-internal-fullstream-execution-v1.json';p=json.loads(lock.read_text())
 source=Path('research/wakeword_p4/training/features.py')
 assert sha(source.read_bytes())==p['files'][str(source)]
 sys.path.insert(0,str(source.parent.resolve()))
 from features import features
 result=json.loads((ROOT/'result.json').read_text())
 assert result['execution_sha256']==sha(lock.read_bytes()) and len(result['cases'])==540
 rows=[]
 for i,request in enumerate(p['requests']):
  path=Path(request['path']);assert sha(path.read_bytes())==request['sha256']
  pcm,rate=sf.read(path,dtype='int16')
  assert rate==16000 and pcm.ndim==1 and len(pcm)==request['samples']
  assert sha(pcm.astype('<i2').tobytes())==request['pcm_sha256']
  raw=features(pcm);matches=[c for c in result['cases'] if c['index']==i]
  assert len(matches)==4
  for c in matches:
   path=Path(c['path']);assert sha(path.read_bytes())==c['sha256']
   with np.load(path,allow_pickle=False) as a:
    assert a['raw'].dtype==raw.dtype and a['raw'].shape==raw.shape and a['raw'].tobytes()==raw.tobytes()
  rows.append(dict(index=i,request_order=request['order'],raw_sha256=sha(raw.tobytes()),shape=list(raw.shape),models_matched=4))
 assert len(rows)==135
 (R/'4090-fp32-fullstream-original-frontend-audit-v1.json').write_text(json.dumps(dict(execution_sha256=sha(lock.read_bytes()),rows=rows,all_passed=True,limits='Original135 consumed media redecoded and frontend recomputed; all540 saved raw arrays exact. Same locked frontend implementation, not an independent frontend implementation. No model inference or recognition scoring.'),indent=2)+'\n')
if __name__=='__main__':main()
