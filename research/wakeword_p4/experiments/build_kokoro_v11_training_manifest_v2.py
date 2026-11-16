import json,hashlib
from pathlib import Path
base=Path('research/wakeword_p4/results/4090-kokoro-v11-training-manifest-v1.jsonl'); out=Path('research/wakeword_p4/results/4090-kokoro-v11-training-manifest-v2.jsonl'); rows=[json.loads(x) for x in base.read_text().splitlines()]
aug=json.loads(Path('.local/wakeword/kokoro-v11-muse-augmentation-v1/manifest.json').read_text())['records']
for r in aug: rows.append({'path':r['path'],'label':0,'role':'development-training-augmentation','request_id':r['source_request_id']+'-aug','voice':None,'speed':None,'phrase_id':'muse','carrier':None,'pcm_sha256':r['sha256'],'native_sha256':None,'sample_rate':16000,'trimmed':False,'augmentation':r['augmentation'],'seed':r['seed'],'source_plan':'research/wakeword_p4/results/4090-kokoro-v11-training-plan-v1.json'})
out.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows));print(json.dumps({'records':len(rows),'positive':sum(r['label'] for r in rows),'negative':sum(not r['label'] for r in rows),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
