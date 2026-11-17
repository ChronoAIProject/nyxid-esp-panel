"""Whole96 synthesized clips, fixed CUDA ASR, no selection or relabeling."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import hashlib,importlib.metadata,json,subprocess,sys,time
from pathlib import Path
R=Path('research/wakeword_p4/results');LOCK=R/'4090-parallel-melo-context-asr-plan-v1.json';MODEL=Path('.local/wakeword/whisper-small-en')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 replay=json.loads((R/'4090-parallel-melo-context-replay-plan-v1.json').read_text());paths=[Path(__file__),Path('research/wakeword_p4/training/gpu_python.py'),R/'4090-parallel-melo-context-replay-plan-v1.json']+[p for p in MODEL.iterdir() if p.is_file()];rows=[]
 for r in replay['rows']:
  wav,=[x['path'] for x in r['files'] if x['path'].endswith('.wav')];paths.append(Path(wav));rows.append(dict(index=r['index'],request=r['request'],wav=wav))
 p=dict(schema='parallel-melo-context-asr-plan-v1',files={str(p):sha(p) for p in paths},versions={n:importlib.metadata.version(n) for n in ['faster-whisper','ctranslate2','numpy','soundfile']},rows=rows,policy='Existing pinned Whisper small.en CUDAfloat16,CPUthreads2,beam3,temperature0,English,no prompt/VAD/previous-text conditioning; full96waveforms once in original order, save all text/segments/word estimates. No rerender/relabel/filter/selection based on ASR or wake scores. Intended text and ASR are not phonetic truth. No new models, acceptance access or training.')
 with LOCK.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for path,h in p['files'].items():assert sha(path)==h,path
 for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
 import numpy as np,soundfile as sf
 from faster_whisper import WhisperModel
 model=WhisperModel(str(MODEL),device='cuda',compute_type='float16',cpu_threads=2,num_workers=1);assert model.model.device=='cuda';rows=[]
 for r in p['rows']:
  pcm,rate=sf.read(r['wav'],dtype='int16');assert rate==16000 and pcm.ndim==1;start=time.monotonic()
  segments,info=model.transcribe(pcm.astype(np.float32)/32768,language='en',beam_size=3,temperature=0,condition_on_previous_text=False,vad_filter=False,word_timestamps=True)
  segs=[dict(id=s.id,start=s.start,end=s.end,text=s.text,avg_logprob=s.avg_logprob,no_speech_prob=s.no_speech_prob,words=[dict(start=w.start,end=w.end,word=w.word,probability=w.probability) for w in (s.words or [])]) for s in segments]
  row=dict(**r,segments=segs,elapsed_seconds=time.monotonic()-start);rows.append(row);print(json.dumps(dict(index=r['index'],text=''.join(s['text'] for s in segs))),flush=True)
 out=dict(schema='parallel-melo-context-asr-original-v1',plan_sha256=sha(LOCK),device=model.model.device,rows=rows,policy=p['policy'])
 with (R/'4090-parallel-melo-context-asr-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
