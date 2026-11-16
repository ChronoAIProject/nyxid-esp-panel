import json,hashlib
from pathlib import Path
root=Path('.local/wakeword/kokoro-v11-role-isolated-development-v2'); rows=json.loads((root/'records.json').read_text()); out=Path('research/wakeword_p4/results/4090-kokoro-v11-training-manifest-v1.jsonl')
with out.open('w') as f:
 for r in rows:
  label=1 if r['phrase_id'] in ('nyx','official') else 0
  x={'path':str(root/r['pcm_path']),'label':label,'role':'development-training','request_id':r['request_id'],'voice':r['voice'],'speed':r['speed'],'phrase_id':r['phrase_id'],'carrier':r['carrier'],'pcm_sha256':r['pcm_sha256'],'native_sha256':r['native_sha256'],'sample_rate':16000,'trimmed':False,'source_plan':'research/wakeword_p4/results/4090-kokoro-v11-role-isolated-request-plan-v1.json'}
  f.write(json.dumps(x,sort_keys=True)+'\n')
print(out, len(rows), hashlib.sha256(out.read_bytes()).hexdigest())
