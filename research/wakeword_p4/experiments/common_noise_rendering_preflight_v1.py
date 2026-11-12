"""First50 original paired schedules, component-audited common-noise renderings."""
import hashlib,io,json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from features import features
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=Path('.local/wakeword');source=base/'piper-balanced-training-full-v1';pack=base/'piper-balanced-training-features-v1';probe=base/'paired-consistency-init-gradient-probe-v1/gradient-probe.json';out=base/'common-noise-rendering-preflight-v1'
assert not out.exists();assert sha(probe)=='0cb06dcb3276125d56b0bc260ef7c6be7e57921fed03b4b799a482e3d675e42f'
assert sha(source/'manifest.jsonl')=='e3a69b86294bf91db29777428e785252411710834d22ba92a037d595c963cdc0'
assert sha(pack/'report.json')=='93733755fec50b41cf71fcc8e2efa9fdfc14956a61e8b50e98e4bd110d5248b3'
records=json.loads((pack/'report.json').read_text())['records'];lookup={(int(r['voice']),r['length_multiplier'],r['target'],r['variant']):r for r in records}
traces=json.loads(probe.read_text())['pair_trace'];assert len(traces)==50
out.mkdir();ir=np.zeros(961,np.float64);ir[[0,160,480,960]]=[1,.25,-.12,.06];np.save(out/'rir.npy',ir)
receipts=[];windows=[];support=[]
for b,tr in enumerate(traces):
 batch=[]
 for voice in tr['voices']:
  phrases=['hey_nyx','hey_nick','hey_next','hey_mix'];dry=[];metadata=[]
  for phrase in phrases:
   r=lookup[voice,tr['speed'],phrase,0];p=source/r['source'];assert sha(p)==r['source_sha256'];s,rate=sf.read(p,dtype='float64');assert rate==16000 and s.ndim==1 and len(s)==r['source_samples'];dry.append(s);metadata.append(r)
  size=32000+max(len(s)+960 for s in dry)+16000
  seed=int(hashlib.sha256(f'common-noise-v1:{voice}:{tr["speed"]}'.encode()).hexdigest()[:16],16)
  noise=np.random.default_rng(seed).normal(0,1,size);noise/=np.sqrt(np.mean(noise**2))
  noise25=noise*.08*10**(-25/20);noise15=noise*.08*10**(-15/20)
  signals={};unscaled={};waves={};raws={};gains={};components={'noise25':noise25,'noise15':noise15,'rir':ir}
  for j,s in enumerate(dry):
   wet=fftconvolve(s,ir,mode='full')
   for arm,signal in [('quiet',s),('noise25',s),('combined',wet),('noise15',s),('rir',wet)]:
    normalized=signal*(.08/np.sqrt(np.mean(signal**2)));padded=np.pad(normalized,(32000,size-32000-len(normalized)));signals[j,arm]=padded
    n=noise25 if arm=='noise25' else noise15 if arm in ['noise15','combined'] else np.zeros(size)
    unscaled[j,arm]=padded+n
  for arm in ['quiet','noise25','combined','noise15','rir']:
   groupgain=min(1.,.95/max(np.max(np.abs(unscaled[j,arm])) for j in range(4)))
   gains[arm]=groupgain
   for j in range(4):
    # Quiet baseline preserves original individual headroom. Other arms
    # share one group headroom scalar so additive noise remains identical.
    gain=min(1.,.95/np.max(np.abs(unscaled[j,arm]))) if arm=='quiet' else groupgain
    w=unscaled[j,arm]*gain;assert np.isfinite(w).all() and np.max(np.abs(w))<1
    buf=io.BytesIO();sf.write(buf,w,16000,subtype='PCM_16',format='WAV');buf.seek(0);pcm,rate=sf.read(buf,dtype='int16');assert np.any(pcm) and not np.any((pcm==-32768)|(pcm==32767));raw=features(pcm)
    waves[j,arm]=w;raws[j,arm]=raw;components[f'signal_{j}_{arm}']=signals[j,arm];components[f'wave_{j}_{arm}']=w;components[f'raw_{j}_{arm}']=raw;components[f'pcm_{j}_{arm}']=pcm
    if arm=='quiet':
     old=metadata[j];assert sha(pack/old['raw'])==old['raw_file_sha256'];prior=np.load(pack/old['raw'],allow_pickle=False);np.testing.assert_array_equal(raw[:len(prior)],prior)
  # Noise component before shared headroom is exactly shared across phrases.
  for arm in ['noise25','combined','noise15']:
   n=noise25 if arm=='noise25' else noise15
   for j in range(4):np.testing.assert_array_equal(unscaled[j,arm],signals[j,arm]+n)
  low=max((32000+len(s)+960+159)//160-1 for s in dry);high=min(350,min(len(raw) for raw in raws.values()))
  assert low<=high and (high-150)*160<=32000
  endpoint=tr['endpoint_inclusive'];valid=low<=endpoint<=high
  # Preserve scheduled endpoint for this diagnostic; explicit fallback only.
  for j in range(4):
   for arm in ['quiet','noise25','combined','noise15','rir']:
    raw=raws[j,arm];batch.append(np.clip(raw[endpoint-149:endpoint+1].astype(np.float32)/25.6,0,26))
  path=out/f'b{b}-voice{voice}.npz';np.savez_compressed(path,**components)
  receipts.append({'batch':b,'voice':voice,'speed':tr['speed'],'source_sha256':[r['source_sha256'] for r in metadata],'seed':seed,'buffer_samples':size,'group_headroom':gains,'scheduled_endpoint_inclusive':endpoint,'conservative_common_endpoint_min':low,'max':high,'scheduled_endpoint_supported':valid,'arrays':path.name,'arrays_sha256':sha(path)})
  support.append(valid)
 windows.append(np.stack(batch))
 if b%10==0:print(json.dumps({'batch':b,'groups':len(receipts)}),flush=True)
np.savez_compressed(out/'windows.npz',x=np.stack(windows))
report={'script_sha256':sha(Path(__file__)),'source_manifest_sha256':sha(source/'manifest.jsonl'),'probe_sha256':sha(probe),'rir_sha256':sha(out/'rir.npy'),'windows_sha256':sha(out/'windows.npz'),'batches':50,'groups':100,'windows':2000,'quiet_baseline_raw_prefix_exact':True,'scheduled_supported_groups':sum(support),'groups_receipts':receipts,'optimizer_updates':0,'limits':'Common rendering construction only; no matched independent-noise control yet. Shared finite buffers extend diagnostic silence, source PCM never trimmed. Float noise components shared before group headroom; integer PCM rounding may change additive residual. Generation/container references only, not phonetic truth. No model inference, newdev/final or promotion.'}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');Path('research/wakeword_p4/results/4090-common-noise-rendering-preflight-v1.json').write_bytes((out/'report.json').read_bytes());print(json.dumps({k:v for k,v in report.items() if k!='groups_receipts'}),flush=True)
