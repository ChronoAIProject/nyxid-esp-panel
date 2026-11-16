import json,time,hashlib
from pathlib import Path
import numpy as np,torch
root=Path('.local/wakeword/kokoro-v11-embeddings-v2'); rows=json.loads((root/'manifest.json').read_text())['records']; X=[];y=[]
for r in rows:
 z=np.load(root/r['path']); X.append(z[-1]); y.append(r['label'])
X=torch.tensor(np.asarray(X),dtype=torch.float32,device='cuda'); y=torch.tensor(y,dtype=torch.long,device='cuda')
torch.manual_seed(726); m=torch.nn.Sequential(torch.nn.Linear(96,32),torch.nn.ReLU(),torch.nn.Linear(32,2)).cuda(); opt=torch.optim.Adam(m.parameters(),lr=1e-3); start=time.time(); hist=[]
for i in range(500):
 opt.zero_grad(); loss=torch.nn.functional.cross_entropy(m(X),y); loss.backward(); opt.step()
 if i in (0,99,499): hist.append({'step':i+1,'loss':float(loss),'accuracy':float((m(X).argmax(1)==y).float().mean()),'device':torch.cuda.get_device_name()})
out=Path('.local/wakeword/kokoro-v11-embedding-diagnostic-v1');out.mkdir(exist_ok=False);torch.save(m.state_dict(),out/'weights.pt');(out/'report.json').write_text(json.dumps({'schema':'kokoro-v11-embedding-diagnostic-v1','records':len(rows),'steps':500,'history':hist,'elapsed_seconds':time.time()-start,'gpu':torch.cuda.get_device_name(),'qualification':False,'reason':'last-embedding classifier; not continuous waveform streaming or INT8 model'},indent=2)+'\n');print(json.dumps(hist))
