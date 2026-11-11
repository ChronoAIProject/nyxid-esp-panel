"""Explain what an endpoint PCM code can and cannot establish about clipping."""
import ast,hashlib,json,wave
from collections import Counter
from pathlib import Path
import numpy as np
source=Path('.local/wakeword/piper/generate_samples.py')
digest=hashlib.sha256(source.read_bytes()).hexdigest()
assert digest=='b6068e38ca986fc3c1cd55b631878aeb619ebe5a65b32fd9f8a870c7e8e694d8'
tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audio_float_to_int16')
namespace={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
convert=namespace['audio_float_to_int16'];examples=[]
for sign in [1,-1]:
 x=np.array([0,.1,.2,.4,.2,.1,0],dtype=np.float32)*sign
 y=convert(x)
 examples.append(dict(input=x.tolist(),input_peak=float(np.abs(x).max()),output=y.tolist(),
                      endpoint_codes=int(np.count_nonzero((y==-32768)|(y==32767)))))
base=Path('.local/wakeword/mac-evidence-import-v1');manifest=base/'old-complete-container-pack-v1/study/sources.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='0f2b008c2525ee9c093580008af1029618823f92d2718622acd513e682cbc981'
peaks=Counter();signs=Counter();hashes=[]
for item in json.loads(manifest.read_text()):
 row=item['aliases'][0]
 if row['engine']!='piper':continue
 p=base/'old-complete-original-wavs'/row['path']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==item['source_sha256']
 with wave.open(str(p),'rb') as w:pcm=w.readframes(w.getnframes())
 assert hashlib.sha256(pcm).hexdigest()==row['pcm_sha256']
 x=np.frombuffer(pcm,dtype='<i2').astype(np.int32);peaks[str(int(np.abs(x).max()))]+=1
 signs['min_-32767']+=int(np.any(x==-32767));signs['max_32767']+=int(np.any(x==32767));hashes.append(item['source_sha256'])
r=dict(scope='Old training Piper endpoint-code diagnostic, not a source reconstruction',
 converter_source_sha256=digest,converter=ast.get_source_segment(source.read_text(),node),
 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 examples=examples,piper_files=len(hashes),absolute_pcm_peak_counts=dict(peaks),signed_endpoint_file_counts=dict(signs),
 original_failed_zero_endpoint_check_retracted=False,training_data_changed=False,
 limits='Pinned local converter peak-normalizes to32767 before cast; bounded nonclipped float fixture can produce32767. Actual Mac archive files not matched to local GPU generation manifest (0/40 earlier failures), so this is mechanism plausibility, not proof of original native waveform integrity or exact generation provenance. -32767 is within PCM range and is not counted by asymmetric -32768/32767 endpoint gate. Native float and generation provenance remain required; do not relax gate retrospectively.')
Path('research/wakeword_p4/results/4090-old-piper-quantization-diagnostic-v1.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:r[k] for k in ['piper_files','absolute_pcm_peak_counts','signed_endpoint_file_counts','examples']}))
