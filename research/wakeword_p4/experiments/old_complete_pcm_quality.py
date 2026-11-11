"""Mechanical audit of immutable old training PCM; never filters/relabels data."""
import hashlib,json,wave
from collections import Counter
from pathlib import Path
import numpy as np
base=Path('.local/wakeword/mac-evidence-import-v1')
source=base/'old-complete-container-pack-v1/study/sources.json'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='0f2b008c2525ee9c093580008af1029618823f92d2718622acd513e682cbc981'
rows=json.loads(source.read_text());details=[];counts=Counter()
for item in rows:
 row=item['aliases'][0];path=base/'old-complete-original-wavs'/row['path']
 assert not Path(row['path']).is_absolute() and '..' not in Path(row['path']).parts
 assert hashlib.sha256(path.read_bytes()).hexdigest()==item['source_sha256']
 with wave.open(str(path),'rb') as wav:
  rate=wav.getframerate();channels=wav.getnchannels();width=wav.getsampwidth()
  assert channels==1 and width==2 and wav.getcomptype()=='NONE'
  pcm=wav.readframes(wav.getnframes())
 assert hashlib.sha256(pcm).hexdigest()==row['pcm_sha256']
 x=np.frombuffer(pcm,dtype='<i2');assert len(x)==row['samples']
 rms=float(np.sqrt(np.mean((x.astype(np.float64)/32768)**2)))
 duration=len(x)/rate;saturated=int(np.count_nonzero((x==-32768)|(x==32767)))
 checks=dict(mono_pcm16_16k=rate==16000,nonempty_nonzero=bool(len(x) and np.any(x)),
             duration_02_to_5_seconds=.2<=duration<=5,rms_at_least_0001=rms>=.001,
             no_saturated_pcm16=saturated==0)
 for k,v in checks.items():counts[k+'/pass' if v else k+'/fail']+=1
 details.append(dict(source_sha256=item['source_sha256'],path=row['path'],engine=row['engine'],
  label=row['label'],phrase=row['phrase'],sample_rate=rate,samples=len(x),duration_seconds=duration,
  rms=rms,saturated_samples=saturated,checks=checks))
r=dict(scope='All2594 canonical original old training WAVs, including sources excluded by v40/v41 task sampler',
 source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 counts=dict(counts),all_mechanical_checks_passed=all(all(d['checks'].values()) for d in details),
 details=details,training_data_changed=False,detector_scored=False,
 limits='Retrospective mechanical audit only; thresholds mirror SLR83 specification, not new filtering or training inclusion rules. Original container integrity is not proof of uncropped generated phrase, phonetic correctness, human quality, TTS version provenance or native float no-clipping. Includes training sources only, no development/final PCM access.')
Path('research/wakeword_p4/results/4090-old-complete-pcm-quality-v1.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(dict(counts=r['counts'],all_mechanical_checks_passed=r['all_mechanical_checks_passed'])))
