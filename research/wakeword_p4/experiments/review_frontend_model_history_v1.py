"""Locked 2x2 feature-splice mechanism diagnostic; not a deployable policy."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from collect_multispeaker_formal_v1 import prepare,predict
from uint8_event_policy_v1 import event_steps
from replay_continuous_chatterbox_v1 import PLAN as SOURCE_PLAN,sha
R=Path('research/wakeword_p4/results');PLAN=R/'4090-history-factorial-plan-v1.json';OUT=Path('.local/wakeword/history-factorial-v1')
SOURCE=R/'4090-continuous-chatterbox-original-v1.json'

def lock():
    old=json.loads(SOURCE_PLAN.read_text());files=dict(old['files'])
    for f in [SOURCE_PLAN,SOURCE,Path(__file__)]:files[str(f)]=sha(f)
    p=dict(files=files,versions=old['versions'],conditions=['continuous','silence','continuous_prefix_silence_target','silence_prefix_continuous_target'],policy='All12 Nyx x2 gap schedules x2 fixed models x4 feature conditions. Reconstruct exact whole session PCM and validate hash. For each target create same-length PCM with all other audio zero. Compute complete continuous and isolated frontend streams with same phase. Splice at first model call whose unrounded support exceeds target start. Prefix conditions change model input history; suffix conditions change frontend history represented in target features. Call group may straddle onset, hence boundary contamination remains. Whole-session event policy fixed; all codes retained. Continuous control must reproduce original codes. Synthetic feature interventions diagnose mechanisms only, not physical frontend resets, not additive causal effects. No tuning/training/acceptance/fresh exposure.')
    with PLAN.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')

def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes()
    p=json.loads(PLAN.read_text())
    for f,h in p['files'].items():assert sha(f)==h,f
    import importlib.metadata
    for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
    import soundfile as sf
    from ai_edge_litert.interpreter import Interpreter,OpResolverType
    sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
    def runtime(**kw):return Interpreter(num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,**kw)
    source=json.loads(SOURCE_PLAN.read_text());sessions=json.loads(SOURCE.read_text())['results'];OUT.mkdir(exist_ok=False);records=[]
    for session in sessions:
        full=np.zeros(session['samples'],np.int16)
        for t in session['timeline']:
            if t['kind']!='clip':continue
            row=source['rows'][t['index']];wav,=[i['path'] for i in row['files'] if i['path'].endswith('.wav')]
            pcm,rate=sf.read(wav,dtype='int16');assert rate==16000
            full[81600+t['start']:81600+t['end']]=pcm
        import hashlib
        assert hashlib.sha256(full.astype('<i2').tobytes()).hexdigest()==session['pcm_sha256']
        continuous,support,_=prepare(features(full),len(full));clocks=np.minimum(((support+479)//480)*480,len(full))-81600
        controls={}
        for model in source['models']:
            _,codes,_=predict(continuous,model['path'],runtime)
            with np.load(session['models'][model['name']]['trace_path']) as z:assert np.array_equal(codes,z['uint8'])
            controls[model['name']]=codes
        for t in session['timeline']:
            if t['kind']!='clip' or t['group']!='nyx':continue
            isolated=np.zeros_like(full);isolated[81600+t['start']:81600+t['end']]=full[81600+t['start']:81600+t['end']]
            quiet,qs,_=prepare(features(isolated),len(full));assert np.array_equal(support,qs)
            call=int(np.searchsorted(support,81600+t['start'],side='right'));frame=call*3
            conditions=dict(continuous=continuous,silence=quiet,continuous_prefix_silence_target=np.concatenate([continuous[:frame],quiet[frame:]]),silence_prefix_continuous_target=np.concatenate([quiet[:frame],continuous[frame:]]))
            record=dict(gap_samples=session['gap_samples'],index=t['index'],splice_call_zero_based=call,splice_support=int(support[call]),target=t,models={})
            for model in source['models']:
                name=model['name'];record['models'][name]={}
                for condition,values in conditions.items():
                    if condition=='continuous':codes=controls[name];q=None
                    else:_,codes,q=predict(values,model['path'],runtime)
                    events=[dict(step=s,relative_sample=int(clocks[s-1])) for s in event_steps(codes,.91796875)]
                    means=np.concatenate([np.full(4,np.nan),np.convolve(codes.astype(np.int64),np.ones(5,np.int64),'valid')/1280.0]);mask=(clocks>t['start'])&(clocks<=t['end'])
                    f=OUT/f'{session["gap_samples"]}-{t["index"]:03d}-{name}-{condition}.npz'
                    np.savez_compressed(f,uint8=codes,relative_block_return=clocks)
                    record['models'][name][condition]=dict(target_events=sum(t['start']<e['relative_sample']<=t['end'] for e in events),target_max_mean=float(np.nanmax(means[mask])),events=events,trace_path=str(f),trace_sha256=sha(f),quantization=q)
            records.append(record);print(json.dumps(dict(gap=session['gap_samples'],index=t['index'])),flush=True)
    (OUT/'result.json').write_text(json.dumps(dict(completed=True,continuous_controls_byte_exact=True,plan_sha256=sha(PLAN),records=records),indent=2)+'\n')
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
