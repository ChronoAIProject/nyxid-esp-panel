"""Mine fixed-model streaming INT8 high scores from audited training-only speech.

Never scans validation, development regression or independent-final audio.
High scores are candidates, not proof that the utterance is truly negative.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
from evaluate import Detector, smooth, event_indices


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def worker(job):
    model,features,output,row,context,minimum,maximum=job
    path=Path(features)/row['path']
    if row['split']!='training' or sha(path)!=row['feature_sha256']:
        raise ValueError('Changed or non-training source')
    data=np.load(path,mmap_mode='r',allow_pickle=False)
    if data.dtype!=np.uint16 or data.shape!=(row['frames'],40):
        raise ValueError('Invalid source features')
    detector=Detector(Path(model));first=detector.predict(data[:300]);detector.reset()
    np.testing.assert_array_equal(first,detector.predict(data[:300]));detector.reset()
    values=smooth(detector.predict(data),5)
    target=Path(output)/(row['speaker']+'-scores.npy');np.save(target,values)
    eligible=np.flatnonzero(values>=minimum)
    ordered=eligible[np.argsort(-values[eligible],kind='stable')]
    blocked=np.zeros(len(values),dtype=bool);candidates=[]
    for index in ordered:
        index=int(index);end=3*(index+5)+1
        if blocked[index] or end<context or end>len(data):continue
        candidates.append(dict(score_index=index,score=float(values[index]),
                               frame_start=end-context,frame_end=end,
                               stream_seconds=(3*(index+4)+5)*.01))
        blocked[max(0,index-66):min(len(values),index+67)]=True
        if len(candidates)>=maximum:break
    return dict(speaker=row['speaker'],path=row['path'],split='training',
                feature_sha256=row['feature_sha256'],score_sha256=sha(target),
                frames=row['frames'],source_seconds=row['source_seconds'],
                max_score=float(values.max()),events_at_point5=len(event_indices(values,.5)),
                reset_checked=True,candidates=candidates)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','training-report','features','corpus','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--min-score',type=float,default=.5)
    p.add_argument('--max-per-stream',type=int,default=64)
    a=p.parse_args()
    if a.output.exists() or not 0<a.min_score<=255/256 or a.workers<1 or a.max_per_stream<1:
        p.error('Use fresh output and valid mining parameters')
    training=json.loads(a.training_report.read_text())
    if sha(a.model)!=training['model_sha256'] or training['context_frames']!=150:
        raise ValueError('Need pinned150-frame model')
    cache=json.loads((a.features/'cache.json').read_text())
    index=sha(a.features/'human.json')
    rows=json.loads((a.features/'human.json').read_text())
    if (index!=cache['index_sha256'] or not rows or any(r['split']!='training' for r in rows)
            or not cache['speaker_disjoint']):
        raise ValueError('Require exclusively audited training cache')
    transcripts={};transcript_hashes=[]
    for path in sorted(a.corpus.rglob('*.trans.txt')):
        transcript_hashes.append(dict(path=str(path.relative_to(a.corpus)),sha256=sha(path)))
        for line in path.read_text().splitlines():
            key,text=line.split(maxsplit=1)
            if key in transcripts:raise ValueError('Duplicate transcript')
            transcripts[key]=text
    expected={(r['path'],r['sha256']) for r in cache['transcripts']}
    if {(r['path'],r['sha256']) for r in transcript_hashes}!=expected:
        raise ValueError('Transcripts differ from audited training corpus')
    a.output.mkdir(parents=True);started=time.monotonic();streams=[];selected=[]
    jobs=[(str(a.model),str(a.features),str(a.output),r,150,a.min_score,a.max_per_stream) for r in rows]
    with ProcessPoolExecutor(a.workers) as pool:
        for original,result in zip(rows,pool.map(worker,jobs)):
            sources=[];at=0
            for r in original['sources']:
                end=at+r['samples'];sources.append((at,end,r))
                at=end+3200+(-r['samples']-3200)%160
            if at/16000!=original['seconds']:raise ValueError('Source timeline mismatch')
            for candidate in result['candidates']:
                lo,hi=candidate['frame_start']*160,(candidate['frame_end']+2)*160
                overlap=[]
                for start,end,r in sources:
                    if start<hi and end>lo:
                        text=transcripts[Path(r['path']).stem]
                        overlap.append(dict(path=r['path'],sha256=r['sha256'],
                                            pcm_sha256=r['pcm_sha256'],text=text,
                                            source_start_seconds=start/16000,source_end_seconds=end/16000))
                if not overlap:raise ValueError('Candidate entirely in inserted silence')
                suspect=any(re.search(r'\bhey\s+(?:nyx|nix|nicks|knicks|nick\s+s)\b',x['text'],re.I) for x in overlap)
                selected.append(dict(**candidate,speaker=original['speaker'],split='training',
                                     path=original['path'],feature_sha256=original['feature_sha256'],
                                     sources=overlap,quarantined=suspect))
            streams.append({k:v for k,v in result.items() if k!='candidates'})
            print(json.dumps(dict(completed=len(streams),planned=len(rows),candidates=len(selected),
                                  seconds=round(time.monotonic()-started,1))),flush=True)
    report=dict(role='Training-only real-speech hard-negative candidates, not quality evidence',
                model_sha256=sha(a.model),training_report_sha256=sha(a.training_report),
                feature_cache_sha256=sha(a.features/'cache.json'),feature_index_sha256=index,
                source_sha256=sha(Path(__file__)),context_frames=150,window_phase=1,
                smoothing_window=5,min_score=a.min_score,separation_steps=67,max_per_stream=a.max_per_stream,
                source_hours=sum(r['source_seconds'] for r in rows)/3600,streams=streams,
                candidates=selected,transcripts_sha256=hashlib.sha256(json.dumps(transcript_hashes,sort_keys=True).encode()).hexdigest(),
                validation_audio_accessed=False,independent_final_audio_accessed=False,
                limits='Training model scores are deliberately reused for mining, never evaluation. Literal transcript screen only; candidate labels require review before training. Original corpus speaker isolation remains in effect.')
    (a.output/'candidates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete=True,streams=len(streams),candidates=len(selected),
                         source_hours=report['source_hours'],seconds=round(time.monotonic()-started,1))),flush=True)


if __name__=='__main__':main()
