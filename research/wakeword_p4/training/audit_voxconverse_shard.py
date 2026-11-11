"""Audit embedded development recordings without wake scoring or resampling."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(8*1024*1024): h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): p.error('Fresh report required')
    source=json.loads((a.source/'source.json').read_text());path=a.source/'dev-00000-of-00005.parquet'
    entry=next(x for x in source['selected_shards'] if Path(x['path']).name==path.name)
    assert path.stat().st_size==entry['size'] and sha(path)==entry['lfs']['oid']
    table=pq.ParquetFile(path);rows=[]
    for batch in table.iter_batches(batch_size=1):
        r=batch.to_pylist()[0];embedded=r['audio']['bytes']
        assert embedded
        info=sf.info(io.BytesIO(embedded));pcm,rate=sf.read(io.BytesIO(embedded),dtype='int16')
        finite=bool(np.isfinite(pcm).all())
        end=max(r['timestamps_end'],default=0)
        starts=r['timestamps_start'];ends=r['timestamps_end']
        assert len(starts)==len(ends)==len(r['speakers'])
        assert all(0<=start<stop<=info.duration+.02 for start,stop in zip(starts,ends))
        rows.append(dict(index=len(rows),embedded_path=r['audio']['path'],encoded_bytes=len(embedded),
            encoded_sha256=hashlib.sha256(embedded).hexdigest(),rate=rate,channels=info.channels,
            frames=info.frames,seconds=info.duration,subtype=info.subtype,annotation_end_seconds=end,
            pcm16_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),
            locally_labeled_speakers=len(set(r['speakers'])),
            replay_format_eligible=rate==16000 and info.channels==1 and info.subtype=='PCM_16' and finite,
            samples_preserved=True,trimmed=False))
    result=dict(shard=path.name,shard_sha256=sha(path),source_sha256=sha(a.source/'source.json'),
        script_sha256=sha(Path(__file__)),recordings=len(rows),source_hours=sum(x['seconds'] for x in rows)/3600,
        all_replay_format_eligible=all(x['replay_format_eligible'] for x in rows),
        unique_encoded_hashes=len({x['encoded_sha256'] for x in rows}),records=rows,
        wake_models_scored=False,final_audio_accessed=False,
        limits='Full embedded mirror recordings audited, no trimming/resampling. Byte equality to original upstream archive not established. Diarization time validity does not prove no editing or no target phrases. Speaker IDs are local. Development only; this duration is not a scored FAR exposure or final acceptance.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__': main()
