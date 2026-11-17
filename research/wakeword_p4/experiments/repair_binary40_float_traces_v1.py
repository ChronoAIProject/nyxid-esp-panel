"""Recover overwritten synthetic traces, preserving original float events exactly."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from export_binary40_trained_v1 import sha
R=Path('research/wakeword_p4/results');LOCK=R/'4090-binary40-float-trace-repair-plan-v1.json'
B=Path('.local/wakeword/binary40-float-replay-v3')
OLD=R/'4090-binary40-float-replay-original-v2.json'
def read(p):return json.loads(Path(p).read_text())
def plan():
    d=read(OLD);files={str(OLD):sha(OLD)};missing=[]
    for c,v in d['cohorts'].items():
        p=Path(v['path']);files[str(p)]=sha(p)
        for row in read(p)['records']:
            trace=Path(row['trace']['path']);files[str(trace)]=sha(trace)
            if sha(trace)!=row['trace']['sha256']:missing.append(dict(cohort=c,index=row['index'],phase=row['phase']))
    assert len(missing)==24 and all(x['phase']=='synthetic' for x in missing)
    for p in [Path(__file__),Path(__file__).with_name('replay_binary40_float_v2.py'),Path('.local/wakeword/binary40-training-v1/final.pt')]:files[str(p)]=sha(p)
    p=dict(files=files,missing=missing,policy='Repair only24 overwritten synthetic float traces (12Legacy+12Jenny); no background or INT8 reinference, no model/training/calibration/profile changes. Strict finalweights loaded; recompute identical physical stream, require exact original float event list, preserve original v2 report and existing files, hardlink verified surviving traces to unique phase-index names. Every repaired waveform/hash/clock/oldINT8trace authenticated. Original allscope event statistics unchanged.')
    with LOCK.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
    p=read(LOCK)
    for path,h in p['files'].items():assert sha(path)==h,path
    import torch
    from replay_binary40_float_v2 import wave,float_events
    from collect_multispeaker_formal_v1 import prepare
    sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
    from causal_binary40_v1 import Binary40
    from features import features
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    cp=torch.load('.local/wakeword/binary40-training-v1/final.pt',map_location='cpu',weights_only=False);assert cp['step']==20000
    model=Binary40().cuda().eval();model.load_state_dict(cp['model'],strict=True)
    assert all(torch.equal(v.cpu(),cp['model'][k]) for k,v in model.state_dict().items())
    B.mkdir(exist_ok=False);d=read(OLD);reports={};recovered=[];start=time.monotonic()
    for c,v in d['cohorts'].items():
        old=read(v['path']);folder=B/c;folder.mkdir();keys=set()
        for row in old['records']:
            key=(row['phase'],row['index']);assert key not in keys;keys.add(key)
            src=Path(row['trace']['path']);dest=folder/f'{row["phase"]}-{row["index"]:06d}.npz';assert not dest.exists()
            if sha(src)==row['trace']['sha256']:os.link(src,dest)
            else:
                assert dict(cohort=c,index=row['index'],phase=row['phase']) in p['missing']
                pcm=wave(row['source']);combined=np.concatenate((np.zeros(81600,np.int16),pcm))
                values,support,_=prepare(features(combined),len(combined));clock=np.minimum(((support+479)//480)*480,len(combined))-81600
                ip=Path(row['saved_int8_trace']['path']);assert sha(ip)==row['saved_int8_trace']['sha256']
                with np.load(ip) as z:
                    expected=z['block_return_sample']-81600 if c=='melo' else z['real_block_return_sample'];assert np.array_equal(clock,expected)
                state=model.initial_state(device='cuda');parts=[]
                with torch.inference_mode():
                    for offset in range(0,len(values),3000):
                        chunk=torch.from_numpy(values[offset:offset+3000].T.copy()[None]).cuda();y,state=model(chunk,state);parts.append(torch.sigmoid(y[0,0]).cpu().numpy())
                probability=np.concatenate(parts)
                es=[dict(step=s,relative_sample=int(clock[s-1])) for s in float_events(probability)]
                assert es==row['events']['float'],(c,row['index'])
                np.savez_compressed(dest,float_probability=probability,relative_block_return=clock,support=support)
                assert sha(dest)==row['trace']['sha256'],(c,row['index'],'recomputed trace differs')
                recovered.append(dict(cohort=c,index=row['index'],phase=row['phase']))
            row['trace']=dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest))
        path=folder/'result.json';path.write_text(json.dumps(old,indent=2)+'\n')
        reports[c]=dict(path=str(path),sha256=sha(path),groups=old['groups'])
    assert recovered==p['missing']
    out=dict(d,schema='binary40-float-replay-original-v3',cohorts=reports,trace_repair=dict(plan_sha256=sha(LOCK),rows=len(recovered),elapsed_seconds=time.monotonic()-start,original_report_sha256=sha(OLD),all_recomputed_traces_and_events_exact=True,background_reruns=0))
    with (R/'4090-binary40-float-replay-original-v3.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out['trace_repair']))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
