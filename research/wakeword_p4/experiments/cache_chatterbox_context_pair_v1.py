"""Training-only paired waveform-history caches, no model inference."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from causal_sequence_supervision_v1 import masks
R=Path('research/wakeword_p4/results');ADMISSION=R/'4090-chatterbox-training-admission-v1.json';PLAN=R/'4090-chatterbox-context-cache-plan-v1.json';OUT=Path('.local/wakeword/chatterbox-context-cache-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lock():
    rows=json.loads(ADMISSION.read_text())['admitted'];assert len(rows)==84
    assert all(r['request']['role']=='training-only' for r in rows)
    paths=[Path(__file__),ADMISSION,Path(__file__).with_name('causal_sequence_supervision_v1.py'),Path('research/wakeword_p4/training/features.py')]
    schedule=[]
    for row in rows:
        assert sha(row['wav']['path'])==row['wav']['sha256'];paths.append(Path(row['wav']['path']))
        pool=sorted([r for r in rows if r['request']['voice']==row['request']['voice'] and r['request']['requested_label']==0 and r['index']!=row['index']],key=lambda r:r['index'])
        prefix=pool[row['index']%len(pool)]
        for gap in [8000,32000]:schedule.append(dict(target=row,prefix=prefix,gap_samples=gap))
    p=dict(files={str(f):sha(f) for f in paths},schedule=schedule,versions={n:importlib.metadata.version(n) for n in ['numpy','soundfile','pymicro-features']},prime=81600,tail=8000,retained_lead_frames=360,max_frames=846,policy='All84 admitted training-only targets x2 fixed gaps, same-voice admitted daily negative prefix, deterministic index-based choice independent of model scores. Compare preceding speech versus zeros with identical total length/target bytes/timing. Extract entire waveform frontend before slicing at a multiple-of3 frame boundary leaving at least360 feature frames before target onset. Whole target and tail retained. This is a declared training-history crop with >=309-frame model context before positive supervision, not full-wave training; startup negative supervision remains. Preserve original prefix-end positive masks; negative targets supervise all retained calls. No development input/training invocation/threshold/model scoring. Phase/masks exactly paired. No claim independent speakers, physical uninterrupted model state at crop, or generalization.')
    with PLAN.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes()
    p=json.loads(PLAN.read_text())
    for f,h in p['files'].items():assert sha(f)==h,f
    for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
    import soundfile as sf
    sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
    OUT.mkdir(exist_ok=False);records=[];offset=0;changed=0
    streams={c:(OUT/f'{c}-features.bin').open('xb') for c in ['speech','silence']}
    try:
        for i,s in enumerate(p['schedule']):
            row=s['target'];prefix=s['prefix'];target,rate=sf.read(row['wav']['path'],dtype='int16');previous,prate=sf.read(prefix['wav']['path'],dtype='int16');assert rate==prate==16000
            onset=81600+len(previous)+s['gap_samples'];end=onset+len(target)
            start_frame=3*((onset//160-2-360)//3);start_sample=start_frame*160
            lead=onset-start_sample;endpoint=lead+round(row['automatic_prefix_end_seconds']*16000) if row['request']['requested_label'] else None
            variants={};hashes={};wavehashes={}
            for c in streams:
                wave=np.concatenate([np.zeros(81600,np.int16),previous if c=='speech' else np.zeros_like(previous),np.zeros(s['gap_samples'],np.int16),target,np.zeros(8000,np.int16)])
                assert np.array_equal(wave[onset:end],target)
                raw=features(wave)[start_frame:];assert raw.dtype==np.uint16 and raw.shape[1]==40 and len(raw)<=846
                variants[c]=raw;hashes[c]=hashlib.sha256(raw.astype('<u2').tobytes()).hexdigest();wavehashes[c]=hashlib.sha256(wave.astype('<i2').tobytes()).hexdigest()
                raw.astype('<u2').tofile(streams[c])
            assert variants['speech'].shape==variants['silence'].shape
            frames=len(variants['speech']);label=row['request']['requested_label'];support,pw,nw=masks(frames,label,int(lead) if label else None,int(endpoint) if label else None)
            assert np.isclose(pw.sum(),label) and np.isclose(nw.sum(),1) and not np.any((pw>0)&(nw>0))
            if label:assert (support[pw>0].min()//160-309)>=0
            # Same continuous frontend framing after crop; pad/model groups preserve original phase.
            assert start_frame%3==0 and lead>=360*160
            target_frames=np.arange(frames)*160+320>=lead
            different=bool(np.any(variants['speech'][target_frames]!=variants['silence'][target_frames]));changed+=different
            records.append(dict(pair=i,source=row,prefix=prefix,gap_samples=s['gap_samples'],offset=offset,frames=frames,label=label,voice=row['request']['voice'],lead=int(lead) if label else None,endpoint=int(endpoint) if label else None,crop_start_frame=start_frame,retained_target_onset=lead,full_waveform_sha256=wavehashes,feature_sha256=hashes,target_region_features_differ=different,positive_calls=np.flatnonzero(pw).tolist(),negative_calls=int(np.count_nonzero(nw))))
            offset+=frames
    finally:
        for stream in streams.values():stream.close()
    (OUT/'records.json').write_text(json.dumps(records,indent=2)+'\n')
    result=dict(completed=True,plan_sha256=sha(PLAN),pairs=len(records),positive_pairs=sum(r['label'] for r in records),frames_per_condition=offset,pairs_target_features_differ=changed,files={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in OUT.iterdir()},all_masks_checked=True,all_targets_byte_exact=True,paired_geometry_exact=True,training_invoked=False)
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
