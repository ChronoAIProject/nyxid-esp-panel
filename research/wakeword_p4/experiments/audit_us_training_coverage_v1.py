"""Read-only independent byte/conversion/grid audit; no model inference."""
import hashlib,itertools,json,math,io
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from piper_phonemize import phonemize_espeak
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
root=Path('.local/wakeword/us-training-coverage-v1');source=Path('.local/wakeword/piper-us-training-source-v1')
run=json.loads((root/'run.json').read_text());report=json.loads((root/'report.json').read_text());src=json.loads((source/'source.json').read_text())
assert sha(source/'source.json')==run['source_sha256']==report['source_sha256']=='76f36ee23be647b674c5450548c90eebec1bde51bba3d9d9dab1273e904903de'
for name,info in src['files'].items():assert sha(source/name)==info['sha256'] and (source/name).stat().st_size==info['size_bytes']
assert sha(root/'run.json')==report['run_sha256'] and sha(root/'manifest.jsonl')==report['manifest_sha256']
assert sha(Path('research/wakeword_p4/experiments/generate_us_training_coverage_v1.py'))==run['script_sha256']==report['script_sha256']
rows=[json.loads(x) for x in (root/'manifest.jsonl').read_text().splitlines()]
texts={'hey_nyx':'Hey Nicks.','hey_nick':'Hey Nick.','hey_next':'Hey Next.','hey_mix':'Hey Mix.'}
grid=list(itertools.product(['lessac','ryan','ljspeech'],[.9,1.,1.1],texts,range(3)))
assert len(rows)==len(grid)==108
checks=[]
for i,(r,g) in enumerate(zip(rows,grid)):
 assert (r['voice'],r['length_multiplier'],r['target'],r['draw'])==g and r['order']==i
 assert r['split']=='training' and r['role']=='us-coverage-training-only' and r['speaker_id']==0 and r['text']==texts[r['target']]
 model=source/r['voice']/f"en_US-{r['voice']}-medium.onnx";cfgpath=Path(str(model)+'.json');cfg=json.loads(cfgpath.read_text())
 assert sha(model)==r['model_sha256'] and sha(cfgpath)==r['config_sha256'] and r['run_sha256']==sha(root/'run.json')
 phones=phonemize_espeak(r['text'],cfg['espeak']['voice']);assert len(phones)==1 and phones[0]==r['phonemes']
 ids=cfg['phoneme_id_map'];tokens=ids['^']+ids['_']
 for phone in phones[0]:tokens=tokens+ids[phone]+ids['_']
 assert tokens+ids['$']==r['token_ids']
 scales=np.asarray([cfg['inference']['noise_scale'],cfg['inference']['length_scale']*g[1],cfg['inference']['noise_w']],dtype=np.float32)
 np.testing.assert_array_equal(scales,r['scales'])
 for key,digest in [('native_path','native_sha256'),('float16k_path','float16k_sha256'),('path','sha256')]:assert sha(root/r[key])==r[digest]
 native=np.load(root/r['native_path'],allow_pickle=False);x=np.load(root/r['float16k_path'],allow_pickle=False)
 assert native.dtype==x.dtype==np.dtype('float32')
 rate=cfg['audio']['sample_rate'];div=math.gcd(rate,16000);expected=resample_poly(native,16000//div,rate//div) if rate!=16000 else native.copy()
 np.testing.assert_array_equal(expected,x)
 pcm,sr=sf.read(root/r['path'],dtype='int16');info=sf.info(root/r['path'])
 # Repeat the pinned libsndfile conversion in memory; do not assume idealized rounding.
 buffer=io.BytesIO();sf.write(buffer,x,16000,subtype="PCM_16",format="WAV");buffer.seek(0)
 quant,_=sf.read(buffer,dtype="int16")
 np.testing.assert_array_equal(pcm,quant)
 assert native.ndim==x.ndim==pcm.ndim==1 and np.isfinite(native).all() and np.isfinite(x).all()
 assert len(native)==r['native_samples'] and len(x)==len(pcm)==r['samples']==r['generated_reference_end_sample']
 assert rate==r['native_rate'] and len(x)==math.ceil(len(native)*16000/rate)
 assert sr==16000 and info.subtype=='PCM_16' and info.channels==1
 assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==r['pcm_sha256']
 rms=float(np.sqrt(np.mean(x.astype(np.float64)**2)));assert rms==r['rms'] and rms>=.001
 assert .2<=len(pcm)/16000<=5 and np.any(pcm) and np.max(np.abs(native))<1 and np.max(np.abs(x))<1
 assert not np.any((pcm==-32768)|(pcm==32767)) and r['trimmed'] is False and r['normalized'] is False
 assert r['native_durations'] is None and all(r['checks'].values()) and r['quality_pass'] is True
 checks.append(dict(path=r['path'],samples=len(pcm),rms=rms,verified=True))
assert len(list(root.glob('*.wav')))==108 and len(list(root.glob('*.npy')))==216
out=dict(scope='Independent read-only audit of every retained training-only output; no ONNX/wake-model inference',clips=108,verified=108,float_dtype='float32',pcm_dtype='int16',original_report_sha256=sha(root/'report.json'),manifest_sha256=sha(root/'manifest.jsonl'),run_sha256=sha(root/'run.json'),source_sha256=sha(source/'source.json'),audit_script_sha256=sha(Path(__file__)),checks=['all source/output/PCM hashes','exact preregistered grid/order/training split','config phonemes/token IDs/scales','native-to-resampled array equality','PCM16 exactly reproduces pinned libsndfile conversion (same library, not independent rounding implementation)','full lengths and independent mechanical quality gates'],details=checks,limits='Mechanical synthesis integrity only; no phonetic truth, physical speaker independence, historical corpus repair, model acceptance or deterministic generation claim. New development/finals not accessed.')
Path('research/wakeword_p4/results/4090-us-training-coverage-audit-v1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='details'}))
