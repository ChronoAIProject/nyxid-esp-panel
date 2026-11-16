"""Small deterministic training-only waveform augmentation feasibility pilot."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');PLAN=R/'4090-c161-pitch-pilot-plan-v1.json';OUT=Path('.local/wakeword/c161-pitch-pilot-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
    meta=Path('.local/wakeword/hey-broad-training-features-v1/hey_nyx/features.json');rows=json.loads(meta.read_text());selected=[]
    for engine in ['piper','kokoro','qwen']:
        for text in ['Hey Nicks.','Hey Nick.']:
            row=next(r for r in rows if r['engine']==engine and r['text']==text and r['source_pair_key']['carrier']=='bare' and r['variant']==0)
            assert row['split']=='training' and sha(row['path'])==row['sha256'];selected.append(row)
    sox=Path(shutil.which('sox')).resolve();files={str(p):sha(p) for p in [Path(__file__).resolve(),meta,sox,Path('research/wakeword_p4/training/features.py')]}
    files.update({r['path']:r['sha256'] for r in selected})
    d=dict(schema='c161-pitch-pilot-plan-v1',files=files,sox=str(sox),rows=selected,cents=[-200,200],gain=.8,policy='Feasibility only: first training-only quiet bare target/Nick per3engines, same +/-200cents transform for both labels; SoX pitch (duration preserving) then PCM16 output, fixed .8 headroom before transform. Preserve complete rendered wave, never crop/stretch after transform. No inference/training/heldout access; sample count equality is not phonetic timing/semantic correctness or proof of useful coverage. Max12derived WAV+features, bounded sequential CPU.')
    with PLAN.open('x') as f:f.write(json.dumps(d,indent=2)+'\n')
def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes()
    p=json.loads(PLAN.read_text())
    for f,h in p['files'].items():assert sha(f)==h,f
    assert shutil.disk_usage(OUT.parent).free>100*1024**3
    import soundfile as sf
    sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
    from features import features
    OUT.mkdir(exist_ok=False);details=[]
    for i,row in enumerate(p['rows']):
        source,rate=sf.read(row['path'],dtype='int16');assert rate==16000 and source.ndim==1 and len(source)==row['samples']
        assert hashlib.sha256(source.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
        baseline=features(source)
        for cents in p['cents']:
            dest=OUT/f'{i:02d}-{cents:+d}.wav'
            command=[p['sox'],'--single-threaded','-v','0.8',row['path'],'-b','16',str(dest),'pitch',str(cents)]
            result=subprocess.run(command,capture_output=True,text=True,check=True)
            x,sr=sf.read(dest,dtype='int16');assert sr==16000 and x.ndim==1 and len(x)==len(source)
            assert np.isfinite(x).all() and np.max(np.abs(x.astype(np.int32)))<32767
            raw=features(x);assert raw.shape==baseline.shape and not np.array_equal(raw,baseline)
            f=dest.with_suffix('.npy');np.save(f,raw,allow_pickle=False)
            details.append(dict(source_index=i,engine=row['engine'],label=row['label'],text=row['text'],source_sha256=row['sha256'],cents=cents,command=command,stderr=result.stderr,samples=len(x),wav_sha256=sha(dest),feature_sha256=sha(f),feature_shape=list(raw.shape),changed_feature_fraction=float(np.mean(raw!=baseline)),peak=int(np.max(np.abs(x.astype(np.int32)))),wav=str(dest),features=str(f)))
    out=dict(schema='c161-pitch-pilot-result-v1',plan_sha256=sha(PLAN),details=details,derived_files=12,bytes=sum(f.stat().st_size for f in OUT.iterdir()),checks_passed=True,limits='Mechanical feasibility only. Pitch shift alters timbre and potential local timing, not new independent speakers or validated pronunciation. Feature change alone does not isolate pitch from gain. No recognition/model improvement claim or labels admitted to training.')
    (R/'4090-c161-pitch-pilot-result-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='details'}))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
