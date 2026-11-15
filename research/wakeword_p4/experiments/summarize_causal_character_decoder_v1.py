"""Publish original direct decoder traces losslessly and summarize actual events."""
import collections,gzip,hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=[]
 for group in ('primed','jenny','joe'):
  root=Path('.local/wakeword/causal-character-decoder-waveform-v1')/group
  original=root/'result.json';data=json.loads(original.read_text());assert data['all_passed']
  dest=R/f'4090-causal-character-decoder-{group}-original-v1.json.gz'
  with dest.open('xb') as f:f.write(gzip.compress(original.read_bytes(),mtime=0))
  traces={};counts=collections.Counter();events=collections.Counter();background_samples=0;index=[]
  for i,row in enumerate(data['records']):
   record=row['record'];trace=Path(row['trace_path']);assert sha(trace)==row['trace_sha256']
   with np.load(trace,allow_pickle=False) as z:
    for key in z.files:traces[f'r{i:04}_{key}']=z[key]
   target=record.get('target_id',record.get('target'))
   kind='background' if row['phase']!='synthetic' else ('positive' if target=='hey_nyx' else 'negative')
   counts[kind]+=1
   source=[e for e in row['events'] if e['sample']>0]
   events[kind+'_clips']+=bool(source);events[kind+'_events']+=len(source)
   events['prime_events']+=sum(e['sample']<=0 for e in row['events'])
   if kind=='background':background_samples+=record['samples']
   index.append(dict(index=i,phase=row['phase'],original_index=row['index'],original_trace_sha256=row['trace_sha256']))
  packed=R/f'4090-causal-character-decoder-{group}-traces-v1.npz'
  assert not packed.exists();np.savez_compressed(packed,**traces)
  out.append(dict(group=group,counts=dict(counts),events=dict(events),background_hours=background_samples/16000/3600,original=dict(path=str(dest),sha256=sha(dest),uncompressed_sha256=sha(original)),traces=dict(path=str(packed),sha256=sha(packed),index=index)))
 report=dict(schema='causal-character-decoder-behavior-v1',groups=out,threshold_selected=False,candidate_promoted=False,limits='Consumed development only. Direct greedy HEY NICKS character decoder, no binary threshold or smoothing. Source events have sample>0; prime retained. No right boundary test. Reused backgrounds nonadditive. No model rerun or numerical equality gate. Zero events is a nonfunctional wake detector, not successful rejection.')
 dest=R/'4090-causal-character-decoder-behavior-v1.json'
 with dest.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 for row in out:print(json.dumps({k:row[k] for k in ('group','counts','events','background_hours')}))
if __name__=='__main__':main()
