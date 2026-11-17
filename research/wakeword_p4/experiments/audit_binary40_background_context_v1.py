"""One complete already-consumed background transcript audit, not training."""
import importlib.metadata as metadata,json,subprocess,sys,time
from pathlib import Path
import numpy as np
from export_binary40_trained_v1 import sha
R=Path('research/wakeword_p4/results');LOCK=R/'4090-binary40-background-context-plan-v1.json'
MODEL=Path('.local/wakeword/whisper-small-en')
WAV=Path('.local/wakeword/prospective-background-whole-waveform-v1/wav/ppexo.wav')
def plan():
    files=[Path(__file__),WAV,Path('.local/wakeword/whisper-small-en-source.json'),R/'4090-binary40-melo-comparison-v1.json',
           Path('.local/wakeword/prospective-background-development-source-v1/test4-diagnostic-v2.json')]
    files += [p for p in MODEL.iterdir() if p.is_file()]
    p=dict(files={str(f):sha(f) for f in files},versions={n:metadata.version(n) for n in ['faster-whisper','ctranslate2','numpy','soundfile']},
           event_relative_sample=1988160,rate=16000,display_context_seconds=10,
           policy='Single whole178.432s consumed development recording, existing pinned small.en CUDAfloat16 CPUthreads2 beam3 temperature0 English no prompt/VAD/no previous-text conditioning. Preserve all segments and nominal timestamps, report overlap with event+-10s and source diarization. No wake reinference, filtering/relabeling, training, acceptance or threshold/calibration change. ASR timestamps/text not phone truth, no listening claim.')
    with LOCK.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
    p=json.loads(LOCK.read_text())
    for path,h in p['files'].items():assert sha(path)==h
    for n,v in p['versions'].items():assert metadata.version(n)==v
    import soundfile as sf
    import pyarrow.parquet as pq
    from faster_whisper import WhisperModel
    import hashlib
    pcm,rate=sf.read(WAV,dtype='int16');assert rate==16000 and len(pcm)==2854912 and pcm.ndim==1
    assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()=='cc94e505f7a980d688d0004dafbb95104a1642752ab98f63d4926309159eb314'
    shard=Path('.local/wakeword/prospective-background-development-source-v1/test-00004-of-00011.parquet')
    audit=json.loads(Path('.local/wakeword/prospective-background-development-source-v1/test4-diagnostic-v2.json').read_text())
    assert sha(shard)==audit['shard_sha256']
    row=pq.ParquetFile(shard).read(columns=['timestamps_start','timestamps_end','speakers']).to_pylist()[15]
    event=p['event_relative_sample']/16000
    intervals=[dict(start=a,end=b,speaker=s) for a,b,s in zip(row['timestamps_start'],row['timestamps_end'],row['speakers']) if a<=event+10 and b>=event-10]
    start=time.monotonic();model=WhisperModel(str(MODEL),device='cuda',compute_type='float16',cpu_threads=2,num_workers=1)
    assert model.model.device=='cuda'
    segments,info=model.transcribe(pcm.astype(np.float32)/32768,language='en',beam_size=3,temperature=0,condition_on_previous_text=False,vad_filter=False,word_timestamps=True)
    rows=[]
    for s in segments:
        rows.append(dict(id=s.id,start=s.start,end=s.end,text=s.text,avg_logprob=s.avg_logprob,no_speech_prob=s.no_speech_prob,
                         words=[dict(start=w.start,end=w.end,word=w.word,probability=w.probability) for w in (s.words or [])]))
    result=dict(schema='binary40-background-context-original-v1',plan_sha256=sha(LOCK),wav_sha256=sha(WAV),pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),
                duration_seconds=len(pcm)/16000,device=model.model.device,compute_type='float16',elapsed_seconds=time.monotonic()-start,
                event_seconds=event,source_diarization_near_event=intervals,source_schema_has_transcripts=False,
                all_segments=rows,context_segments=[r for r in rows if r['start']<=event+10 and r['end']>=event-10],policy=p['policy'])
    out=R/'4090-binary40-background-context-original-v1.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='all_segments'}))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
