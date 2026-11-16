import json,hashlib
from pathlib import Path
root=Path('.local/wakeword/kokoro-v11-development-boundary-v2'); rows=json.loads((root/'records.json').read_text());out=Path('research/wakeword_p4/results/4090-kokoro-v11-training-manifest-v5.jsonl')
with out.open('w') as f:
 for r in rows:
  f.write(json.dumps({'path':str(root/r['pcm_path']),'label':int(r['phrase_id'] in ('nyx','official')),'role':'boundary-complete-development-training','request_id':r['request_id'],'voice':r['voice'],'speed':r['speed'],'phrase_id':r['phrase_id'],'carrier':r['carrier'],'text':r['text'],'pcm_sha256':r['pcm_sha256'],'native_sha256':r['native_sha256'],'sample_rate':16000,'samples':r['pcm_samples'],'keyword_end_samples':r['keyword_end_samples'],'phoneme_metadata':r['phoneme_metadata'],'duration_metadata':r['duration_metadata'],'trimmed':False,'source_plan':'research/wakeword_p4/results/4090-kokoro-v11-role-isolated-request-plan-v2.json'},sort_keys=True)+'\n')
print(json.dumps({'records':len(rows),'positive':sum(r['phrase_id'] in ('nyx','official') for r in rows),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
