"""C161 fixed-profile consumed Mac12 synthetic waveform diagnostic."""
import hashlib,json,sys,subprocess
from pathlib import Path
import numpy as np
from collect_multispeaker_formal_v1 import prepare,predict
from uint8_event_policy_v1 import event_steps
R=Path('research/wakeword_p4/results')
PLAN=R/'4090-c161-mac12-plan-v1.json'
PACK=Path('.local/wakeword/mac-consumed-speech-grid-witness-v1/fixtures')
OUT=Path('.local/wakeword/c161-mac12-v1')
MODEL=Path('.local/wakeword/causal-source-balanced-int8-v1/int8.tflite')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
    manifest=PACK/'manifest.jsonl'
    assert sha(manifest)=='f4645c372050c2f1e19a09ee41312d11675025985e609c296e1309f155640541'
    rows=[json.loads(x) for x in manifest.read_text().splitlines()][:12]
    assert all(r['point']==0 for r in rows)
    assert MODEL.stat().st_size==60952 and sha(MODEL)=='c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81'
    paths=[Path(__file__).resolve(),MODEL,manifest,Path(__file__).with_name('collect_multispeaker_formal_v1.py'),Path(__file__).with_name('uint8_event_policy_v1.py'),Path('research/wakeword_p4/training/features.py')]
    for r in rows:
        p=PACK/r['path'];assert sha(p)==r['sha256'];paths.append(p)
    with PLAN.open('x') as f:f.write(json.dumps(dict(files={str(p):sha(p) for p in paths},rows=rows,model=str(MODEL),threshold=.91796875,smoothing=5,prime=81600,policy='12 existing consumed Mac synthetic PCM, Daniel/Samantha, Nyx/Willow/HiESP bare+command. Fresh frontend/model per whole input, continuous81600zero prime, native480return clock, final finish only at end. Fixed original C161 profile, no calibration/training/final access. Historical official native Willow positive control4/4 uses32256prime and is diagnostic, not exact matched runtime.'),indent=2)+'\n')
def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes()
    p=json.loads(PLAN.read_text())
    for f,h in p['files'].items():assert sha(f)==h,f
    import soundfile as sf
    from ai_edge_litert.interpreter import Interpreter,OpResolverType
    sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
    from features import features
    def runtime(**kw):return Interpreter(num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,**kw)
    OUT.mkdir(exist_ok=False)
    records=[]
    for i,row in enumerate(p['rows']):
        pcm,rate=sf.read(PACK/row['path'],dtype='int16')
        assert rate==16000 and pcm.ndim==1 and len(pcm)==row['samples'] and hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
        combined=np.concatenate((np.zeros(81600,np.int16),pcm))
        raw=features(combined);values,support,clock=prepare(raw,len(combined))
        scores,codes,q=predict(values,p['model'],runtime)
        blocks=np.minimum(((support+479)//480)*480,len(combined))-81600
        steps=event_steps(codes,.91796875)
        events=[dict(step=s,relative_sample=int(blocks[s-1]),finish=bool(support[s-1]>len(combined))) for s in steps]
        trace=OUT/f'{i:03d}.npz';np.savez_compressed(trace,uint8=codes,relative_block_return=blocks,support=support)
        rec=dict(index=i,source=row,events=events,real_events=[e for e in events if e['relative_sample']>0],prime_events=[e for e in events if e['relative_sample']<=0],trace_sha256=sha(trace),quantization=q)
        records.append(rec);print(i,row['voice'],row['target'],row['form'],len(rec['real_events']),flush=True)
    result=dict(schema='c161-mac12-result-v1',plan_sha256=sha(PLAN),model_sha256=sha(MODEL),threshold=.91796875,records=records,limits='Consumed small synthetic diagnostic only, correlated voices and no background/FAR/P4 or human claim; other-target activations are diagnostic, not daily occurrence rates.')
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
