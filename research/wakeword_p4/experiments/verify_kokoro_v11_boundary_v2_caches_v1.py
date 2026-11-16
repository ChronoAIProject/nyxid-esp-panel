import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('.local/wakeword/kokoro-v11-development-boundary-v2'); rows=json.loads((ROOT/'records.json').read_text())
train=[json.loads(x) for x in Path('research/wakeword_p4/results/4090-kokoro-v11-training-manifest-v5.jsonl').read_text().splitlines()]
mm=json.loads(Path('.local/wakeword/kokoro-v11-mfcc-boundary-v3/manifest.json').read_text()); ee=json.loads(Path('.local/wakeword/kokoro-v11-embeddings-boundary-v3/manifest.json').read_text())
assert len(rows)==len(train)==len(mm['records'])==len(ee['records'])==144
for i,(r,t,m,e) in enumerate(zip(rows,train,mm['records'],ee['records'])):
 assert t['path']==str(ROOT/r['pcm_path']) and t['pcm_sha256']==r['pcm_sha256'] and t['keyword_end_samples']==r['keyword_end_samples']
 assert m['pcm_sha256']==r['pcm_sha256'] and e['source_pcm_sha256']==r['pcm_sha256']
 for d,rec in [('mfcc',m),('embeddings',e)]:
  p=Path(f'.local/wakeword/kokoro-v11-{d}-boundary-v3/{i:04d}.npy'); assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['feature_sha256'] if 'feature_sha256' in rec else rec['sha256']
res={'schema':'kokoro-v11-boundary-v2-cache-verification-v1','source_root':str(ROOT),'source_records':144,'positive':sum(t['label'] for t in train),'negative':sum(not t['label'] for t in train),'mfcc_cache':'.local/wakeword/kokoro-v11-mfcc-boundary-v3','embedding_cache':'.local/wakeword/kokoro-v11-embeddings-boundary-v3','mfcc_frames':sum(np.load(f'.local/wakeword/kokoro-v11-mfcc-boundary-v3/{i:04d}.npy',mmap_mode='r').shape[0] for i in range(144)),'embedding_provider':ee['provider'],'acceptance_excluded':True,'threshold_selection_excluded':True,'training_manifest':'research/wakeword_p4/results/4090-kokoro-v11-training-manifest-v5.jsonl'}
Path('research/wakeword_p4/results/4090-kokoro-v11-boundary-v2-cache-verification-v1.json').write_text(json.dumps(res,indent=2)+'\n'); print(json.dumps(res))
