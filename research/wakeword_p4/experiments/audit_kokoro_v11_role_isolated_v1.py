import argparse,json,hashlib
from pathlib import Path
import numpy as np,soundfile as sf
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
r=json.loads((a.root/'records.json').read_text()); checks=[]
for x in r:
 p=a.root/x['pcm_path']; n=a.root/x['native_path']; pcm,sr=sf.read(p,dtype='int16'); nat=np.load(n)
 ok=(sr==16000 and pcm.ndim==1 and len(pcm)>0 and np.isfinite(nat).all() and nat.size>0 and np.max(np.abs(pcm))<32767 and .2<=len(pcm)/sr<=5 and hashlib.sha256(p.read_bytes()).hexdigest()==x['pcm_sha256'])
 checks.append({'request_id':x['request_id'],'pass':bool(ok),'seconds':len(pcm)/sr,'pcm_sha256':x['pcm_sha256']})
out={'schema':'kokoro-v11-role-isolated-audit-v1','records':len(r),'passed':sum(c['pass'] for c in checks),'failed':sum(not c['pass'] for c in checks),'checks':checks,'training_eligible':sum(c['pass'] for c in checks)==len(r),'acceptance_sealed':True}
a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('records','passed','failed','training_eligible')}))
