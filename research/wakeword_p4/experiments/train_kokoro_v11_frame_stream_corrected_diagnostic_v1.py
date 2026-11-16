import json,time
from pathlib import Path
import numpy as np,torch
root=Path('.local/wakeword/kokoro-v11-embeddings-corrected-v1'); rows=json.loads((root/'manifest.json').read_text())['records']; X=[];Y=[]
for r in rows:
 z=np.load(root/r['path']).astype('float32'); X.append(z); Y.extend([r['label']]*len(z))
X=torch.tensor(np.concatenate(X),device='cuda'); Y=torch.tensor(Y,device='cuda',dtype=torch.long)
torch.manual_seed(726); m=torch.nn.Linear(96,2).cuda(); opt=torch.optim.Adam(m.parameters(),lr=2e-3); h=[];st=time.time()
for i in range(1000):
 opt.zero_grad(); loss=torch.nn.functional.cross_entropy(m(X),Y);loss.backward();opt.step()
 if i in (0,199,999): h.append({'step':i+1,'loss':float(loss.detach()),'frame_accuracy':float((m(X).argmax(1)==Y).float().mean())})
# symmetric per-tensor int8 weight replay
w=m.weight.detach().cpu().numpy(); scale=float(np.max(np.abs(w))/127); q=np.round(w/scale).clip(-127,127).astype(np.int8); b=m.bias.detach().cpu().numpy(); scores=(X.detach().cpu().numpy()@((q.astype('float32')*scale).T))+b; acc=float((scores.argmax(1)==Y.cpu().numpy()).mean())
out=Path('.local/wakeword/kokoro-v11-frame-stream-corrected-diagnostic-v1');out.mkdir(exist_ok=False);np.save(out/'weights-int8.npy',q);np.save(out/'weight-scale.npy',np.array(scale));np.save(out/'bias.npy',b);report={'schema':'kokoro-v11-frame-stream-corrected-diagnostic-v1','clips':len(rows),'frames':len(Y),'history':h,'int8_frame_accuracy':acc,'weight_scale':scale,'gpu':torch.cuda.get_device_name(),'elapsed_seconds':time.time()-st,'qualification':False,'reason':'diagnostic linear frame head; not MatchboxNet/TFLite streaming candidate'};(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
