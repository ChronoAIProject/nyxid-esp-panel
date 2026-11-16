import json,hashlib
from pathlib import Path
import numpy as np,soundfile as sf
from openwakeword.utils import AudioFeatures
root=Path('.local/wakeword/kokoro-v11-role-isolated-development-v2-corrected'); out=Path('.local/wakeword/kokoro-v11-embeddings-corrected-v1'); out.mkdir(exist_ok=False)
models=Path('.local/wakeword/oww-models'); af=AudioFeatures(melspec_model_path=str(models/'melspectrogram.onnx'),embedding_model_path=str(models/'embedding_model.onnx'),inference_framework='onnx',device='gpu',ncpu=2)
rows=json.loads((root/'records.json').read_text()); rec=[]
for i,r in enumerate(rows):
 w,s=sf.read(root/r['pcm_path'],dtype='int16'); assert s==16000 and w.ndim==1
 z=af._get_embeddings(w); np.save(out/f'{i:04d}.npy',z,allow_pickle=False)
 rec.append({'request_id':r['request_id'],'label':int(r['phrase_id'] in ('nyx','official')),'path':f'{i:04d}.npy','sha256':hashlib.sha256((out/f'{i:04d}.npy').read_bytes()).hexdigest(),'frames':int(z.shape[0]),'features':int(z.shape[1]),'source_pcm_sha256':r['pcm_sha256']})
(out/'manifest.json').write_text(json.dumps({'schema':'kokoro-v11-embedding-cache-v1','provider':af.embedding_model.get_providers(),'records':rec},indent=2)+'\n'); print('cached',len(rec),'shape',z.shape,'providers',af.embedding_model.get_providers())
