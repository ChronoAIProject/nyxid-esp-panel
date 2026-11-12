"""Read-only full8064 full audit; conversion libraries shared, no generation."""
import hashlib,io,json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from piper_phonemize import phonemize_espeak
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=Path('.local/wakeword/piper-balanced-training-full-v1');run=json.loads((b/'run.json').read_text());report=json.loads((b/'report.json').read_text());rows=[json.loads(x) for x in (b/'manifest.jsonl').read_text().splitlines()]
source=Path('.local/wakeword/piper');configpath=source/'models/en_US-libritts_r-medium.pt.json';c=json.loads(configpath.read_text())
assert sha(configpath)==run['config_sha256']=='119118e510d0b8a7a0c8649a0668640d6db9b00239e874169d964853a8d15848'
assert sha(source/'models/en_US-libritts_r-medium.pt')==run['model_sha256']=='e95ee53770bf598c354a6e6dbfc95ccb259aeeb501d35a86be8a767429ab0ff6'
assert sha(Path('research/wakeword_p4/experiments/generate_piper_balanced_training_full_v1.py'))==run['script_sha256']==report['script_sha256']
assert sha(Path('research/wakeword_p4/experiments/piper_balanced_training_full_v1.md'))==run['plan_sha256']
coverage=Path('research/wakeword_p4/results/4090-piper-training-phrase-coverage-v1.json');assert sha(coverage)==run['coverage_sha256']
for p,h in run['source_files'].items():assert sha(source/p)==h
ids=np.random.default_rng(726).permutation(904)[:672].tolist();assert ids==run['ids']==json.loads(coverage.read_text())['training_ids']
phrases=[('hey_nyx','Hey Nicks.'),('hey_nick','Hey Nick.'),('hey_next','Hey Next.'),('hey_mix','Hey Mix.')];grid=[(i,s,t,x) for i in ids for s in [.9,1.,1.1] for t,x in phrases]
assert len(rows)==report['clips']==len(grid)==8064 and run['seed']==197726 and run['draws']==1
assert sha(b/'run.json')==report['run_sha256'] and sha(b/'manifest.jsonl')==report['manifest_sha256']
failures=[];ledger=[]
for order,(r,(sid,speed,target,text)) in enumerate(zip(rows,grid)):
 assert (r['order'],r['speaker_id'],r['length_scale'],r['target'],r['text'])==(order,sid,speed,target,text)
 assert r['paired_speaker_ids']==[sid,sid] and r['split']=='training' and not r['trimmed'] and not r['normalized'] and r['draw']==0 and r['run_sha256']==report['run_sha256']
 assert r['noise_scale']==r['noise_scale_w']==.333 and r['slerp_weight']==.5
 for info in r['files'].values():
  p=b/info['path'];assert p.parent==b and sha(p)==info['sha256'] and p.stat().st_size==info['size_bytes']
 a=np.load(b/r['files']['native']['path'],allow_pickle=False);v=np.load(b/r['files']['resampled']['path'],allow_pickle=False);d=np.load(b/r['files']['durations']['path'],allow_pickle=False)
 assert a.dtype==v.dtype==d.dtype==np.float32 and a.ndim==v.ndim==d.ndim==1
 phones=phonemize_espeak(text,c['espeak']['voice']);assert len(phones)==1 and phones[0]==r['phonemes']
 token=list(c['phoneme_id_map']['^'])+list(c['phoneme_id_map']['_'])
 for phone in phones[0]:token+=c['phoneme_id_map'][phone]+c['phoneme_id_map']['_']
 token+=c['phoneme_id_map']['$'];assert token==r['token_ids'] and len(d)==len(token) and d.tolist()==r['duration_samples']
 assert np.array_equal(resample_poly(a,320,441),v)
 p=b/r['files']['wav']['path'];pcm,rate=sf.read(p,dtype='int16');buf=io.BytesIO();sf.write(buf,v,16000,format='WAV',subtype='PCM_16');buf.seek(0);q,_=sf.read(buf,dtype='int16');assert np.array_equal(pcm,q)
 assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==r['pcm_sha256']
 rms=float(np.sqrt(np.mean(v.astype(np.float64)**2))) if len(v) else 0.;assert rms==r['rms']
 checks=dict(finite=bool(np.isfinite(a).all() and np.isfinite(v).all()),format_mono_pcm16_16k=pcm.ndim==1 and rate==16000 and sf.info(p).subtype=='PCM_16',nonempty_nonzero=bool(len(a) and len(pcm) and np.any(pcm)),duration=.2<=len(pcm)/16000<=5,rms=rms>=.001,native_no_fullscale=bool(np.all(np.abs(a)<1)),resampled_no_fullscale=bool(np.all(np.abs(v)<1)),pcm_no_endpoint=not bool(np.any((pcm==-32768)|(pcm==32767))),complete_length=len(pcm)==len(v),duration_full_length=bool(np.isfinite(d).all() and np.all(d>=0) and d.sum()==len(a)),resample_length=len(v)==(len(a)*320+440)//441)
 assert checks==r['checks'] and all(checks.values())==r['quality_pass']
 assert r['native_samples']==len(a) and r['samples']==r['generated_reference_end_sample']==len(pcm) and r['native_rate']==22050
 if not all(checks.values()):failures.append(dict(order=order,checks={k:v for k,v in checks.items() if not v}))
 ledger.append(dict(order=order,speaker_id=sid,target=target,native_samples=len(a),samples=len(pcm),duration_total_samples=int(d.sum()),quality_pass=all(checks.values())))
assert failures==report['failures'] and report['quality_pass']==8064-len(failures)
result=dict(scope='Independent read-only8064attempt artifact/ledger/conversion replay',attempts=8064,quality_pass=8064-len(failures),failures=failures,all_file_hashes_match=True,all_duration_arrays_match=True,all_resample_arrays_exact=True,all_pcm_arrays_exact=True,run_sha256=sha(b/'run.json'),manifest_sha256=sha(b/'manifest.jsonl'),original_report_sha256=sha(b/'report.json'),script_sha256=sha(Path(__file__)),ledger=ledger,limits='Shares scipy/libsndfile/phonemizer with generator; not independent numerical implementation or CUDA generation replay. Duration boundaries are generation references, not acoustic truth. No wake inference, newdev/final data, model training or promotion.')
Path('research/wakeword_p4/results/4090-piper-balanced-training-full-audit-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='ledger'}))
