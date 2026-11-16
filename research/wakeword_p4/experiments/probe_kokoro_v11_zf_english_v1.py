"""Compatibility-only probe; excluded from training and acceptance."""
import hashlib,json
from pathlib import Path
import numpy as np, torch
from huggingface_hub import hf_hub_download
from kokoro import KModel,KPipeline
repo='hexgrad/Kokoro-82M-v1.1-zh';rev='01e7505bd6a7a2ac4975463114c3a7650a9f7218'
paths={n:hf_hub_download(repo_id=repo,filename=n,revision=rev) for n in ['config.json','kokoro-v1_1-zh.pth','voices/zf_001.pt']}
m=KModel(repo_id=repo,config=paths['config.json'],model=paths['kokoro-v1_1-zh.pth']).to('cuda').eval()
p=KPipeline(lang_code='a',repo_id=repo,model=m);torch.manual_seed(903001)
c=list(p('Hey Nicks.',voice=paths['voices/zf_001.pt'],speed=.9));assert len(c)==1
x=np.asarray(c[0].audio,dtype=np.float32);assert x.ndim==1 and len(x)>0 and np.isfinite(x).all()
out=Path('.local/wakeword/kokoro-v11-zf-english-probe-v1');out.mkdir(exist_ok=True);np.save(out/'native.npy',x,allow_pickle=False)
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
r={'schema':'kokoro-v11-zf-english-probe-v1','repo':repo,'revision':rev,'source_sha256':{n:sha(p) for n,p in paths.items()},'device':torch.cuda.get_device_name(),'text':'Hey Nicks.','pipeline_language':'a','voice':'zf_001','speed':.9,'seed':903001,'chunks':len(c),'native_samples':len(x),'native_sha256':sha(out/'native.npy'),'peak':float(abs(x).max()),'phonemes':c[0].phonemes,'pred_dur':c[0].pred_dur.cpu().tolist(),'qualification':False,'excluded_from_training_and_acceptance':True,'scope':'Cross-language voice compatibility only; no pronunciation or independent speaker claim.'}
Path('research/wakeword_p4/results/4090-kokoro-v11-zf-english-probe-v1.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
