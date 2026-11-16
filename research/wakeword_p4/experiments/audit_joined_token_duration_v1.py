"""Verify original generator-token sample intervals, not phonetic alignment."""
import collections,gzip,hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 admission=R/'4090-joined-command-pcm-admission-v1.json';p=json.loads(admission.read_text());records=[];vocab=collections.Counter();zero=0
 for row in p['native_metadata']:
  wav=Path(row['path']);d=row['duration'];path=Path(d['path']);path=path if str(path).startswith('.local/') else wav.parent/path;assert sha(path)==d['sha256'];dur=np.load(path,allow_pickle=False).reshape(-1);tokens=np.array(row['token_ids'],np.int64)
  assert dur.size==tokens.size and np.issubdtype(dur.dtype,np.integer) and (dur>=0).all()
  # Original generator output unit is600samples@24k; exact400samples@16k.
  import soundfile as sf
  info=sf.info(wav);assert info.samplerate==16000 and sum(map(int,dur))*400==info.frames
  edges=np.concatenate(([0],np.cumsum(dur,dtype=np.int64)*400));assert edges[-1]==info.frames
  intervals=[]
  for token,start,end in zip(tokens,edges[:-1],edges[1:],strict=True):
   token=int(token);start=int(start);end=int(end);vocab[token]+=1;zero+=int(end==start)
   # Frame's complete400sample observation must fit insideinterval, optional.
   first=((start+159)//160)*160;count=max(0,(end-400-first)//160+1)
   intervals.append(dict(token_id=token,start_sample=start,end_sample=end,fully_contained_400sample_frames=count))
  records.append(dict(path=str(wav),text=row['text'],conditioning=row['conditioning'],phonemes=row['phonemes'],samples=info.frames,intervals=intervals))
 payload=(json.dumps(dict(schema='joined-generator-token-intervals-v1',records=records),separators=(',',':'))+'\n').encode();dest=R/'4090-joined-generator-token-intervals-v1.json.gz'
 with dest.open('xb') as f:f.write(gzip.compress(payload,mtime=0))
 out=dict(schema='joined-token-duration-audit-v1',source_admission_sha256=sha(admission),script_sha256=sha(__file__),native_requests=len(records),tokens=sum(vocab.values()),zero_duration_tokens=zero,vocabulary=dict(sorted(vocab.items())),intervals=dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest)),all_duration_token_counts_integer_nonnegative_and_sample_totals_passed=True,neural_calls=0,policy='OriginalTTSduration unit600@24k=400@16k, exactgeneratorallocation intervals only. Includesboundary/punctuation/space tokens andzero durations, no claimatomicphoneorhumanacoustictruth. No texttospeech, ASR/forcedalignment, recognition/training/newdev. No labelsmappedtoRIR/noiseorfrontendframes fortrainingyet. Futuretargetdesignmustexplicitlyhandlepadding/nonacoustictokens, coarticulation/reverb andcausalobservability; framecontainmentcountsdescriptiveonly. DifferentfromunsupervisedCTCtiming butnotproofofbenefit; priorphoneCTCfailure retained.')
 with (R/'4090-joined-token-duration-audit-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in out.items() if k not in ['vocabulary','policy']}))
if __name__=='__main__':main()
