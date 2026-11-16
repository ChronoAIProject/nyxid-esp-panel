import json,time
from pathlib import Path
import numpy as np,torch
from openwakeword.utils import AudioFeatures
import soundfile as sf
sysroot=Path('.local/wakeword');src=sysroot/'kokoro-v11-role-isolated-development-v2';aug=sysroot/'kokoro-v11-muse-augmentation-v1';af=AudioFeatures(melspec_model_path=str(sysroot/'oww-models/melspectrogram.onnx'),embedding_model_path=str(sysroot/'oww-models/embedding_model.onnx'),inference_framework='onnx',device='gpu',ncpu=2)
X=[];Y=[]
for r in json.loads((src/'records.json').read_text()):
 w,s=sf.read(src/r['pcm_path'],dtype='int16');X.append(af._get_embeddings(w));Y.extend([int(r['phrase_id'] in ('nyx','official'))]*X[-1].shape[0])
for r in json.loads((aug/'manifest.json').read_text())['records']:
 w,s=sf.read(r['path'],dtype='int16');z=af._get_embeddings(w);X.append(z);Y.extend([0]*len(z))
xx=np.concatenate(X).astype('float32');yy=np.array(Y);torch.manual_seed(726);tx=torch.tensor(xx,device='cuda');ty=torch.tensor(yy,device='cuda');m=torch.nn.Linear(96,2).cuda();o=torch.optim.Adam(m.parameters(),lr=2e-3);st=time.time()
for _ in range(1000):o.zero_grad();l=torch.nn.functional.cross_entropy(m(tx),ty);l.backward();o.step()
acc=float((m(tx).argmax(1)==ty).float().mean());out=Path('research/wakeword_p4/results/4090-kokoro-v11-frame-stream-augmented-diagnostic-v1.json');out.write_text(json.dumps({'schema':'kokoro-v11-frame-stream-augmented-diagnostic-v1','clips':168,'frames':len(yy),'augmentation_negative_clips':24,'frame_accuracy':acc,'loss':float(l.detach()),'gpu':torch.cuda.get_device_name(),'elapsed_seconds':time.time()-st,'qualification':False,'scope':'training-fit diagnostic only'},indent=2)+'\n');print(acc,float(l.detach()))
