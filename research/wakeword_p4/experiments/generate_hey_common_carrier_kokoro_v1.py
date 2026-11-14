"""Prospective CPU-only training common-confusion command synthesis."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import numpy as np

R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword')
PLAN=R/'4090-hey-common-carrier-kokoro-generation-execution-v1.json'
OUT=BASE/'hey-common-carrier-kokoro-native-v1'
DERIVED=BASE/'hey-common-carrier-kokoro-seeded-v1.onnx'
MODEL=BASE/'kokoro-models/kokoro-v1.0.onnx';BANK=MODEL.with_name('voices-v1.0.bin')
CARRIERS={'light':'turn on the light.','time':'what time is it?','timer':'set a timer.'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def versions():return {n:importlib.metadata.version(n) for n in ('kokoro-onnx','onnxruntime','onnx','numpy','scipy','soundfile','phonemizer-fork','espeakng-loader')}
def session(path):
    import onnxruntime as ort
    o=ort.SessionOptions();o.intra_op_num_threads=1;o.inter_op_num_threads=1;o.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path),sess_options=o,providers=['CPUExecutionProvider'])
def voice_data(k,req):
    phones=k.tokenizer.phonemize(req['text'],req['lang']);tokens=k.tokenizer.tokenize(phones)
    assert 0<len(tokens)<500
    style=np.asarray(k.get_voice_style(req['conditioning_id'])[len(tokens)],dtype=np.float32)
    return dict(phonemes=phones,token_ids=[0,*tokens,0],style_sha256=hashlib.sha256(style.tobytes()).hexdigest()),style

def plan():
    import onnx,espeakng_loader
    import kokoro_onnx.tokenizer,kokoro_onnx.config
    from kokoro_onnx import Kokoro
    oldplan=R/'4090-hey-broad-common-negative-generation-plan-v1.json';old=json.loads(oldplan.read_text())
    assert sha(MODEL)==old['kokoro_model_sha256'] and sha(BANK)==old['kokoro_bank_sha256']
    original=onnx.load(MODEL);derived=onnx.ModelProto();derived.CopyFrom(original)
    nodes=[n for n in derived.graph.node if 'Random' in n.op_type]
    assert [n.op_type for n in nodes]==['RandomUniformLike','RandomNormalLike']
    originals=[]
    for i,n in enumerate(nodes):
        originals.append(onnx.NodeProto());originals[-1].CopyFrom(n)
        attrs=[a for a in n.attribute if a.name!='seed'];del n.attribute[:];n.attribute.extend(attrs)
        n.attribute.append(onnx.helper.make_attribute('seed',float(704926+100000*i)))
    data=derived.SerializeToString()
    for n,oldnode in zip(nodes,originals,strict=True):n.CopyFrom(oldnode)
    assert derived.SerializeToString()==original.SerializeToString()
    if DERIVED.exists():assert DERIVED.read_bytes()==data
    else:DERIVED.write_bytes(data)
    k=Kokoro.from_session(session(DERIVED),str(BANK))
    records=BASE/'hey-broad-common-kokoro-native-v1/records.jsonl'
    anchors=[json.loads(l) for l in records.read_text().splitlines()];assert len(anchors)==540
    requests=[]
    for anchor in anchors:
        a=anchor['request'];assert a['split']=='training' and anchor['quality_pass']
        for f in anchor['files'].values():assert sha(f['path'])==f['sha256']
        for carrier,suffix in CARRIERS.items():
            req=dict(order=len(requests),anchor_order=anchor['order'],anchor_text=a['text'],conditioning_id=a['conditioning_id'],lang=a['lang'],speed=a['scale'],carrier=carrier,
                     text=a['text'].removesuffix('.')+', '+suffix,label=0,split='training',target_id='common_negative')
            meta,_=voice_data(k,req);requests.append(dict(request=req,metadata=meta,anchor_files=anchor['files']))
    assert len(requests)==1620
    files=[Path(__file__),Path(__file__).with_name('common_carrier_slot_policy_v1.py'),R/'4090-hey-common-carrier-slot-policy-check-v1.json',MODEL,BANK,DERIVED,oldplan,records,R/'4090-hey-broad-surviving-window-census-v2.json',R/'4090-hey-broad-surviving-window-independent-audit-v1.json',Path(inspect.getfile(Kokoro)),Path(kokoro_onnx.tokenizer.__file__),Path(kokoro_onnx.config.__file__),Path(espeakng_loader.get_library_path())]
    files += sorted(p for p in Path(espeakng_loader.get_data_path()).rglob('*') if p.is_file())
    p=dict(schema='hey-common-carrier-kokoro-generation-execution-v1',files={str(f):sha(f) for f in files},versions=versions(),requests=requests,
           provider='CPUExecutionProvider',threads=1,derived_random_seed=[704926,804926],learned_graph_unchanged_except_random_seeds=True,
           synthesis_policy='One native inference per ordered request, no retries/substitution/trim/normalization. Save original native tensor, flattened float, complete float resample, PCM16, durations/tokens/style/speed and every failure. Entire1620pack rejected on any failure. Fixed seed state advances in exact request order; no claim old unseeded anchors share noise.',
           quality='Finite float32 singleton leading axes, nonempty, monoPCM16k,.2..10seconds,RMS>=.001,native/resampled abs<1,noPCMendpoints,ceil2/3full resample length,one nonnegative integer duration per padded token,sum(duration)*600==native_samples. No phonetic boundary inferred for negatives.',
           replacement_policy='Scan batch then shuffled input index, only surviving Kokoro common-negative slots; independent RNGseed90726 consumes selection integer[0,2) and carrier integer[0,3) on EVERY eligible slot, even unselected. Reset per target/arm, not perbatch; freeze decisions before training. Preserve original positive/reciprocal/prefix/background/otherengine tensors. For original end e, oldlo=max(225,lead-25), oldN=oldframes-oldlo+1, newlo=max(225,newlead-25), newN=newframes-newlo+1: newend=newlo+((2*(e-oldlo)+1)*newN)//(2*oldN). No additional baseline RNG draws. Same source voice/speed/render variant/environment seed; source length/crop distribution changes jointly.',
           staging='This lock admits training-only native synthesis, not training. Actual renderer/quality/source overlap/features/replacement tensor schedule and model/optimizer execution locks must be pushed after material admission and before model training. No fresh development/acceptance, human listening or board required.',
           limits='Synthetic-process specification only, not human pronunciation truth or demonstrated recognition improvement. Old fourcarrier two-voice failure retained. New seeded generator changes stochastic waveform as well as text/duration.')
    with PLAN.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    print(json.dumps(dict(requests=1620,voices=20,derived_sha256=sha(DERIVED),model_inferences=0)))

def run():
    import soundfile as sf
    from scipy.signal import resample_poly
    from kokoro_onnx import Kokoro
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes()
    p=json.loads(PLAN.read_text())
    for path,h in p['files'].items():assert sha(path)==h
    assert versions()==p['versions'];OUT.mkdir(exist_ok=False)
    k=Kokoro.from_session(session(DERIVED),str(BANK));failures=[]
    with (OUT/'records.jsonl').open('x') as log:
        for item in p['requests']:
            req=item['request'];i=req['order'];record=dict(request=req,trimmed=False,normalized=False)
            try:
                meta,style=voice_data(k,req);assert meta==item['metadata']
                tokens=np.asarray([meta['token_ids']],dtype=np.int64);speed=np.asarray([req['speed']],dtype=np.float32)
                tensor,duration=k.sess.run(None,dict(input_ids=tokens,style=style,speed=speed))
                raw_outputs={}
                for name,a in [('engine_tensor',tensor),('duration',duration)]:
                    dest=OUT/f'{i:04}-{name}.npy';np.save(dest,a,allow_pickle=False);raw_outputs[name]=dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest))
                record['raw_outputs']=raw_outputs
                assert tensor.dtype==np.float32 and all(n==1 for n in tensor.shape[:-1])
                native=tensor.reshape(-1);resampled=resample_poly(native,2,3);dur=duration.ravel()
                arrays=dict(engine_tensor=tensor,native=native,resampled=resampled,duration=duration,tokens=tokens,style=style,speed=speed);files={}
                for name,a in arrays.items():
                    dest=OUT/f'{i:04}-{name}.npy';np.save(dest,a,allow_pickle=False);files[name]=dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest))
                wav=OUT/f'{i:04}.wav';sf.write(wav,resampled,16000,subtype='PCM_16');pcm,rate=sf.read(wav,dtype='int16');files['wav']=dict(path=str(wav),bytes=wav.stat().st_size,sha256=sha(wav))
                rms=float(np.sqrt(np.mean(resampled.astype(np.float64)**2)))
                checks=dict(finite=all(bool(np.isfinite(a).all()) for a in arrays.values()),nonempty=bool(native.size and pcm.size and np.any(pcm)),
                            format=pcm.ndim==1 and rate==16000 and sf.info(wav).subtype=='PCM_16',duration=.2<=len(pcm)/16000<=10,rms=rms>=.001,
                            native_no_fullscale=bool(np.all(np.abs(native)<1)),resampled_no_fullscale=bool(np.all(np.abs(resampled)<1)),
                            pcm_no_endpoint=not bool(np.any((pcm==-32768)|(pcm==32767))),full_length=len(pcm)==len(resampled)==(len(native)*2+2)//3,
                            token_duration=dur.size==tokens.size and bool(np.all(dur>=0) and np.array_equal(dur,np.rint(dur))),native_duration=float(np.sum(dur,dtype=np.float64))*600==len(native))
                record.update(metadata=meta,files=files,native_samples=len(native),samples=len(pcm),rms=rms,pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),checks=checks,quality_pass=all(checks.values()))
            except Exception as e:record.update(error=type(e).__name__+': '+str(e),quality_pass=False)
            if not record['quality_pass']:failures.append(dict(order=i,checks=record.get('checks'),error=record.get('error')))
            log.write(json.dumps(record)+'\n');log.flush()
            if (i+1)%20==0:print(json.dumps(dict(completed=i+1,total=1620,failures=len(failures))),flush=True)
    result=dict(execution_sha256=sha(PLAN),cases=1620,quality_pass=1620-len(failures),failures=failures,whole_pack_qualified=not failures,records_sha256=sha(OUT/'records.jsonl'),training_launched=False,
                limits=p['limits']+' Independent artifact audit still required before render/features/training.')
    with (R/'4090-hey-common-carrier-kokoro-generation-result-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['plan','run']);a=ap.parse_args();plan() if a.action=='plan' else run()
