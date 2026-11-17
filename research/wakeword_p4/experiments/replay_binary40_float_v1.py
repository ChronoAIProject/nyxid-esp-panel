"""Complete consumed float replay against saved INT8, no classifier fitting."""
import hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from export_binary40_trained_v1 import sha
from replay_binary40_v1 import output as introot,lock as intlock
from collect_multispeaker_formal_v1 import prepare
from uint8_event_policy_v1 import event_steps
R=Path('research/wakeword_p4/results');LOCK=R/'4090-binary40-float-replay-plan-v1.json'
B=Path('.local/wakeword/binary40-float-replay-v1')
COHORTS=['builtin','daily','continuous','primed','jenny','joe','melo','mac']
def read(p):return json.loads(Path(p).read_text())
def plan():
    files={}
    for c in COHORTS:
        files.update(read(intlock(c))['files'])
        for p in [intlock(c),introot(c)/'result.json',R/f'4090-binary40-{c}-comparison-v1.json']:files[str(p)]=sha(p)
    paths=[Path(__file__),Path('.local/wakeword/binary40-training-v1/final.pt'),Path('research/wakeword_p4/training/causal_binary40_v1.py'),
           Path('research/wakeword_p4/training/features.py'),Path(__file__).with_name('collect_multispeaker_formal_v1.py'),Path(__file__).with_name('uint8_event_policy_v1.py')]
    files.update({str(p):sha(p) for p in paths})
    p=dict(files=files,cohorts=COHORTS,chunk_feature_frames=3000,
           policy='Unchanged final20000 float weights, all original8cohorts complete incl all backgrounds, identical original PCM/continuous81600zero prime/native480support/whole-stream finish. CUDAfloat inference with caches carried across3000feature chunks, original frontend and source hashes. Compare saved INT8 codes only; no INT8 rerun, new fitting/threshold/calibration/checkpoint rescue or acceptance. Direct float5score event policy, no fake UINT8 rounding. Correlated consumed diagnostics, no new independent exposure/P4 claim.')
    with LOCK.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
def wave(source,mac=False):
    import soundfile as sf
    path=Path('.local/wakeword/mac-consumed-speech-grid-witness-v1/fixtures')/source['path'] if mac else Path(source.get('wav_path',source.get('path','')))
    expected=source.get('wav_sha256',source.get('sha256'));assert sha(path)==expected
    pcm,rate=sf.read(path,dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==source['samples']
    assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==source['pcm_sha256']
    return pcm
def inputs(c):
    root=introot(c);receipt=read(root/'result.json');p=read(intlock(c))
    if c in ['primed','jenny','joe','melo']:
        rows=[json.loads(s) for s in (root/'binary40/records.jsonl').read_text().splitlines()] if c=='melo' else read(receipt['models'][0]['path'])['records']
        for row in rows:
            s=row['record'];yield row['index'],s,wave(s),Path(row['trace_path']),row['trace_sha256'],row['quantization'],row,None
    elif c=='mac':
        for row in receipt['records']:yield row['index'],row['source'],wave(row['source'],True),root/f'{row["index"]:03d}.npz',row['trace_sha256'],row['quantization'],None,None
    elif c in ['builtin','daily']:
        import soundfile as sf
        for row,s in zip(receipt['records'],p['rows'],strict=True):
            assert row['index']==s['index'];item,=[x for x in s['files'] if x['path'].endswith('.wav')];assert sha(item['path'])==item['sha256']
            pcm,rate=sf.read(item['path'],dtype='int16');assert rate==16000 and pcm.ndim==1
            v=row['models']['binary40'];yield row['index'],s['request'],pcm,root/f'{row["index"]:03d}-binary40.npz',v['trace_sha256'],v['quantization'],None,None
    else:
        import soundfile as sf
        for row in receipt['results']:
            pieces=[];gap=row['gap_samples']
            for i in p['order']:
                item,=[x for x in p['rows'][i]['files'] if x['path'].endswith('.wav')];assert sha(item['path'])==item['sha256']
                pcm,rate=sf.read(item['path'],dtype='int16');assert rate==16000 and pcm.ndim==1;pieces.extend([pcm,np.zeros(gap,np.int16)])
            pieces.append(np.zeros(p['tail_samples'],np.int16));pcm=np.concatenate(pieces)
            combined=np.concatenate((np.zeros(81600,np.int16),pcm));assert hashlib.sha256(combined.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
            v=row['models']['binary40'];yield gap,dict(gap_samples=gap),pcm,Path(v['trace_path']),v['trace_sha256'],v['quantization'],None,row['timeline']
def float_events(values):
    means=np.convolve(np.asarray(values,np.float64),np.ones(5,np.float64)/5,'valid');last=-1000;armed=True;events=[]
    for step,mean in enumerate(means,5):
        if mean<.91796875*.5 and step-last>=67:armed=True
        if armed and mean>=.91796875 and step-last>=67:events.append(step);last=step;armed=False
    return events
def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
    p=read(LOCK)
    for f,h in p['files'].items():assert sha(f)==h,f
    subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','1'],check=True)
    B.mkdir(exist_ok=False)
    try:
        import torch
        sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
        from causal_binary40_v1 import Binary40
        from features import features
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
        cp=torch.load('.local/wakeword/binary40-training-v1/final.pt',map_location='cpu',weights_only=False);assert cp['step']==20000
        model=Binary40().cuda().eval();start=time.monotonic();reports={}
        for c in p['cohorts']:
            folder=B/c;folder.mkdir();records=[];groups={}
            for index,source,pcm,path,expected,q,original,timeline in inputs(c):
                assert sha(path)==expected
                combined=np.concatenate((np.zeros(81600,np.int16),pcm));raw=features(combined);values,support,_=prepare(raw,len(combined))
                clock=np.minimum(((support+479)//480)*480,len(combined))-81600
                if original:
                    assert hashlib.sha256(raw.tobytes()).hexdigest()==original['raw_frontend_sha256']
                    assert hashlib.sha256(values.tobytes()).hexdigest()==original['input_sha256']
                scale,zero=q['input'];quant=np.clip(np.rint(values/scale)+zero,-128,127).astype(np.int8)
                assert hashlib.sha256(quant.tobytes()).hexdigest()==q['quantized_input_sha256']
                with np.load(path) as z:
                    codes=z['uint8'].copy();assert codes.dtype==np.uint8
                    oldclock=z['block_return_sample']-81600 if c=='melo' else z['real_block_return_sample'] if c in ['primed','jenny','joe'] else z['relative_block_return']
                    assert np.array_equal(oldclock,clock) and len(codes)==len(values)//3
                states=model.initial_state(device='cuda');probabilities=[]
                with torch.inference_mode():
                    for offset in range(0,len(values),p['chunk_feature_frames']):
                        chunk=torch.from_numpy(values[offset:offset+p['chunk_feature_frames']].T.copy()[None]).cuda()
                        logits,states=model(chunk,states);probabilities.append(torch.sigmoid(logits[0,0]).cpu().numpy())
                probability=np.concatenate(probabilities);assert np.isfinite(probability).all()
                es={name:[dict(step=s,relative_sample=int(clock[s-1])) for s in steps] for name,steps in [('float',float_events(probability)),('int8',event_steps(codes,.91796875))]}
                onset=source.get('lead_samples',0) if c in ['primed','jenny','joe'] and source.get('target_id')=='hey_nyx' else 0
                hits={name:any(e['relative_sample']>0 and e['relative_sample']>=onset for e in ev) for name,ev in es.items()}
                phase=original['phase'] if original else 'synthetic'
                if c in ['builtin','daily']:key=source['group']
                elif c=='continuous':key='continuous'
                elif c=='mac':key=source['voice']+'/'+source['target']
                elif phase=='background':key='background'
                else:
                    target=source.get('phrase_id')=='nyx' if c=='melo' else source.get('target_id')=='hey_nyx'
                    key=(source['voice_id'].split(':')[-1]+'/' if c=='melo' else '')+('target/' if target else 'other/')+source.get('carrier','whole-utterance-unclassified')
                g=groups.setdefault(key,dict(rows=0,hits=dict(float=0,int8=0),float_only=0,int8_only=0,events=dict(float=0,int8=0),hours=0.));g['rows']+=1
                for name in hits:g['hits'][name]+=hits[name];g['events'][name]+=sum(e['relative_sample']>0 for e in es[name])
                g['float_only']+=hits['float'] and not hits['int8'];g['int8_only']+=hits['int8'] and not hits['float']
                if phase=='background':g['hours']+=len(pcm)/16000/3600
                dest=folder/f'{index:06d}.npz';np.savez_compressed(dest,float_probability=probability,relative_block_return=clock,support=support)
                row=dict(index=index,source=source,phase=phase,events=es,hits=hits,trace=dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest)),saved_int8_trace=dict(path=str(path),sha256=expected),max_probability_difference=float(np.abs(probability.astype(np.float64)-codes.astype(np.float64)/256).max()))
                if timeline:
                    row['timeline']=timeline;row['containers']={}
                    for name in es:
                        groups_hit={}
                        for e in es[name]:
                            owner=next(t for t in timeline if t['start']<e['relative_sample']<=t['end'])
                            if owner['kind']=='clip':groups_hit.setdefault(owner['group'],set()).add(owner['index'])
                        row['containers'][name]={k:sorted(v) for k,v in groups_hit.items()}
                records.append(row)
                with (folder/'records.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(dict(cohort=c,index=index,phase=phase,hits=hits)),flush=True)
            report=folder/'result.json';report.write_text(json.dumps(dict(cohort=c,plan_sha256=sha(LOCK),groups=groups,records=records,completed=True),indent=2)+'\n')
            reports[c]=dict(path=str(report),sha256=sha(report),groups=groups)
        out=dict(schema='binary40-float-replay-original-v1',plan_sha256=sha(LOCK),cohorts=reports,elapsed_seconds=time.monotonic()-start,gpu=torch.cuda.get_device_name(),policy=p['policy'],optimizer_updates=0)
        with (R/'4090-binary40-float-replay-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    except Exception:
        (B/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
