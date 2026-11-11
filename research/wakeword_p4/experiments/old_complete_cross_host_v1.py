"""Fixed Mac witnesses: report all cross-host differences, no detector scoring."""
import hashlib, importlib.util, json, platform, sys, wave
from pathlib import Path
import numpy as np
base=Path('.local/wakeword/mac-evidence-import-v1')
pack=base/'old-complete-container-pack-v1'
sys.path.insert(0,str(pack/'source'))
from audit_old_complete_sources import reference_stream
from features import features
sha=lambda b:hashlib.sha256(b).hexdigest()
sources=json.loads((pack/'study/sources.json').read_text())
records=json.loads((pack/'study/records.json').read_text())
originals=base/'old-complete-original-wavs'
for group in sources:
 row=group['aliases'][0];p=originals/row['path'];assert sha(p.read_bytes())==group['source_sha256']
 with wave.open(str(p)) as w:
  assert (w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getnframes())==(1,2,16000,row['samples'])
  assert sha(w.readframes(w.getnframes()))==row['pcm_sha256']
first={}
for i,g in enumerate(sources):
 r=g['aliases'][0];first.setdefault((r['engine'],r['label']),i)
results=[]
import soundfile as sf
for i in first.values():
 row=sources[i]['aliases'][0];pcm,rate=sf.read(originals/row['path'],dtype='float64');assert rate==16000
 for v in range(3):
  r=records[3*i+v];rendered=reference_stream(pcm,row['path'],v);raw=features(rendered);expected=np.load(pack/'study'/r['raw']);assert raw.shape==expected.shape
  diff=raw.astype(np.int32)-expected.astype(np.int32);where=np.argwhere(diff)
  results.append(dict(pack_index=3*i+v,engine=row['engine'],label=row['label'],variant=v,pcm_exact=sha(rendered.tobytes())==r['rendered_pcm_sha256'],frontend_exact=not len(where),different_elements=len(where),max_abs_delta=int(np.abs(diff).max()),first_difference=None if not len(where) else dict(index=where[0].tolist(),linux=int(raw[tuple(where[0])]),mac=int(expected[tuple(where[0])]))))
report=dict(scope='Training-only cross-host witnesses; no model inference or training',canonical_wav_headers_hashes_and_pcm_verified=len(sources),witnesses=len(results),pcm_exact=sum(r['pcm_exact'] for r in results),frontend_exact=sum(r['frontend_exact'] for r in results),platform=platform.platform(),python=sys.version,script_sha256=sha(Path(__file__).read_bytes()),details=results,limits='Same pinned frontend source, different host/library environment. Differences retained, not acceptance or pronunciation evidence. No validation/new/final WAV opened.')
Path('research/wakeword_p4/results/4090-old-complete-cross-host-v1.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='details'}))
