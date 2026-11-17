"""Bounded fullwave derived features on existing31644 training-only sources."""
import hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from pilot_bare_envelope_warp_v1 import warp
from build_c161_pitch_cache_v2 import selected,META
R=Path('research/wakeword_p4/results');P=R/'4090-c161-envelope-cache-plan-v1.json';O=Path('.local/wakeword/c161-envelope-cache-v1')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def alpha(row):
    key=json.dumps(dict(engine=row['engine'],pair=row['source_pair_key'],variant=row['variant']),sort_keys=True)
    return (.9,1.1)[int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%2]
def main():
    rows=selected();assert len(rows)==31644 and all(r['split']=='training' for r in rows)
    if sys.argv[1]=='lock':
        paths=[META,Path(__file__),Path(__file__).with_name('pilot_bare_envelope_warp_v1.py'),Path(__file__).with_name('build_c161_pitch_cache_v2.py'),Path('research/wakeword_p4/training/features.py'),R/'4090-envelope-harmonics-result-original-v1.json']
        p=dict(files={str(f):sha(f) for f in paths},records=len(rows),frames=sum(r['frames'] for r in rows),expected_feature_bytes=sum(r['frames'] for r in rows)*80,reserve_gib=100,max_new_gib=2,policy='All31644 original broad training-only bare target/Nick/Next/Mix/Muse,3engines/allvariants. Fixed .9/1.1 hash by engine/sourcepair/variant independent of label/text; fullwave transform then continuous MicroFrontend. Source schedules/masks unchanged. Same local envelope algorithm, not canonical VTLP or phonetic preservation claim. CPU2 single process; one derivative per source, no WAV copies, no scoring/training/heldout. Failure terminal and retained; no restart/rescue.')
        with P.open('x') as f:json.dump(p,f,indent=2)
        return
    assert sys.argv[1]=='run' and subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(P)])==P.read_bytes()
    p=json.loads(P.read_text())
    for name,h in p['files'].items():assert sha(name)==h,name
    assert shutil.disk_usage(O.parent).free>(p['reserve_gib']+p['max_new_gib'])*1024**3
    assert not O.exists();O.mkdir()
    import soundfile as sf
    sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
    from features import features
    started=time.monotonic();offset=0
    (O/'launch.json').write_text(json.dumps(dict(pid=__import__('os').getpid(),plan_sha256=sha(P)))+'\n')
    try:
        with (O/'features.bin').open('xb') as binary,(O/'records.jsonl').open('x') as ledger:
            for i,row in enumerate(rows):
                assert sha(row['path'])==row['sha256']
                pcm,sr=sf.read(row['path'],dtype='int16');assert sr==16000 and pcm.ndim==1 and len(pcm)==row['samples']
                assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
                y,meta=warp(pcm.astype(np.float32)/32768,alpha(row))
                assert len(y)==len(pcm) and np.isfinite(y).all() and np.max(np.abs(y))<1
                q=np.rint(y*32768).astype('<i2');assert np.max(np.abs(q.astype(np.int32)))<32767
                raw=features(q);assert raw.dtype==np.uint16 and raw.shape==(row['frames'],40)
                data=raw.astype('<u2').tobytes();binary.write(data)
                rec=dict(index=i,original_offset=row['offset'],offset=offset,frames=len(raw),alpha=alpha(row),source_sha256=row['sha256'],pcm_sha256=hashlib.sha256(q.tobytes()).hexdigest(),features_sha256=hashlib.sha256(data).hexdigest(),label=row['label'],engine=row['engine'],transform=meta)
                ledger.write(json.dumps(rec)+'\n');offset+=len(raw)
                if i==0 or (i+1)%100==0:
                    binary.flush();ledger.flush();progress=dict(completed=i+1,total=len(rows),frames=offset,elapsed_seconds=time.monotonic()-started)
                    (O/'progress.json').write_text(json.dumps(progress)+'\n');print(json.dumps(progress),flush=True)
                if (i+1)%1000==0:assert shutil.disk_usage(O).free>p['reserve_gib']*1024**3
        assert offset==p['frames'] and (O/'features.bin').stat().st_size==p['expected_feature_bytes']
        result=dict(records=len(rows),frames=offset,elapsed_seconds=time.monotonic()-started,plan_sha256=sha(P),files={n:sha(O/n) for n in ['features.bin','records.jsonl']},training_started=False)
        (O/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    except Exception:
        (O/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
