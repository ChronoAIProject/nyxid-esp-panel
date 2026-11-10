"""Reconstruct existing dev-other feature-input PCM for Hi ESP validation only.

Keeps each speaker stream and exact200ms gaps/alignment. It is concatenated read
speech, not natural conversation. Gaps never add to original audio exposure.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(job):
    corpus,features,output,row=job
    corpus,features,output=Path(corpus),Path(features),Path(output)
    if row['split']!='validation' or sha(features/row['path'])!=row['feature_sha256']:
        raise ValueError('Need unchanged validation features')
    sources=[];digest=hashlib.sha256();original=0;total=0
    target=output/(row['speaker']+'.wav')
    with sf.SoundFile(target,'w',samplerate=16000,channels=1,subtype='PCM_16') as stream:
        for source in row['sources']:
            path=(corpus/source['path']).resolve()
            if not path.is_relative_to(corpus.resolve()) or sha(path)!=source['sha256']:
                raise ValueError('Changed or escaped source FLAC')
            pcm,rate=sf.read(path,dtype='int16')
            if (rate!=16000 or pcm.ndim!=1 or len(pcm)!=source['samples'] or
                    hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()!=source['pcm_sha256']):
                raise ValueError('Changed decoded validation PCM')
            original+=len(pcm)
            signal=np.pad(pcm,(0,3200+(-len(pcm)-3200)%160))
            stream.write(signal);digest.update(signal.astype('<i2').tobytes());total+=len(signal)
            sources.append(dict(path=source['path'],sha256=source['sha256'],pcm_sha256=source['pcm_sha256']))
    if (digest.hexdigest()!=row['stream_pcm_sha256'] or total/16000!=row['seconds'] or
            original/16000!=row['source_seconds']):
        raise ValueError('Reconstruction differs from original Nyx frontend input')
    return dict(path=target.name,role='development-read-speech-validation',split='validation',
                speaker=row['speaker'],samples=total,source_seconds=original/16000,
                inserted_silence_seconds=(total-original)/16000,sha256=sha(target),
                pcm_sha256=digest.hexdigest(),prior_feature_sha256=row['feature_sha256'],sources=sources)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','features','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.output.exists() or a.workers<1:p.error('Fresh output and positive workers required')
    cache=json.loads((a.features/'cache.json').read_text())
    if sha(a.features/'human.json')!=cache['index_sha256']:raise ValueError('Changed feature index')
    rows=json.loads((a.features/'human.json').read_text())
    if not rows or any(r['split']!='validation' for r in rows):
        raise ValueError('Require existing exclusively validation corpus')
    for transcript in cache['transcripts']:
        if sha(a.corpus/transcript['path'])!=transcript['sha256']:
            raise ValueError('Changed audited transcript')
    a.output.mkdir(parents=True)
    records=[]
    with ProcessPoolExecutor(a.workers) as pool:
        for row in pool.map(worker,[(str(a.corpus),str(a.features),str(a.output),r) for r in rows]):
            records.append(row);print(json.dumps(dict(completed=len(records),planned=len(rows))),flush=True)
    report=dict(role='Existing dev-other read-speech validation PCM for Hi ESP, not final',
                source_cache_sha256=sha(a.features/'cache.json'),source_audit_sha256=sha(a.features/'cache.json'),
                prior_feature_index_sha256=cache['index_sha256'],source_sha256=sha(Path(__file__)),
                continuous_recordings=False,utterance_concatenation=True,records=records,
                original_source_hours=sum(r['source_seconds'] for r in records)/3600,
                independent_final_audio_accessed=False,
                limits='Previously reused development read speech. Exact200ms utterance gaps plus alignment reproduce Nyx input but are excluded from exposure. Model resets between speakers only. Not natural conversation or independent acceptance.')
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'}))


if __name__=='__main__':main()
