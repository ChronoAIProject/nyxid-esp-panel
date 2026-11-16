"""Descriptive best-substring token edits; never an inference decoder."""
import collections,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from ordered_generator_matcher_v1 import PATTERN
R=Path('research/wakeword_p4/results')
def alignment(reference,sequence):
 # Free prefix/suffix in sequence, deterministic tie order match/sub/delete/insert.
 m,n=len(reference),len(sequence);dp=[[0]*(n+1) for _ in range(m+1)];back={}
 for i in range(1,m+1):dp[i][0]=i;back[i,0]=(i-1,0,'delete')
 for i in range(1,m+1):
  for j in range(1,n+1):
   choices=[(dp[i-1][j-1]+(reference[i-1]!=sequence[j-1]),i-1,j-1,'match' if reference[i-1]==sequence[j-1] else 'substitute'),(dp[i-1][j]+1,i-1,j,'delete'),(dp[i][j-1]+1,i,j-1,'insert')];best=min(choices,key=lambda x:x[0]);dp[i][j]=best[0];back[i,j]=best[1:]
 end=min(range(n+1),key=lambda j:dp[m][j]);i,j=m,end;edits=[]
 while i:
  pi,pj,kind=back[i,j]
  if kind!='match':edits.append(dict(operation=kind,reference_position=i-1 if kind!='insert' else i,expected=reference[i-1] if kind!='insert' else None,predicted=sequence[j-1] if kind!='delete' else None))
  i,j=pi,pj
 return dict(distance=dp[m][end],start=j,end=end,edits=list(reversed(edits)))
def main():
 assert alignment((1,2,3),(9,1,2,3,8))['distance']==0
 assert alignment((1,2,3),(1,4,3))['edits']==[dict(operation='substitute',reference_position=1,expected=2,predicted=4)]
 assert alignment((1,2,3),(1,3))['distance']==1
 assert alignment((1,2,3),(1,9,2,3))['distance']==1
 files=[R/f'4090-causal-{k}-all-training-wave-diagnostic-v1.json' for k in ['unit','position']];out=dict(schema='causal-target-edit-diagnostic-v1',pattern=PATTERN,files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*files]},runs={},limitations='Best substring edit alignment of entire collapsed recording is descriptive only. Ties are deterministic but not acoustically unique. May align command content, ignores right boundary/confidence/timing, and cannot establish causal acoustic errors. None remains a mismatching symbol. No fuzzy decoder, threshold selection, optimizer update or heldout data.')
 for kind,path in zip(['unit','position'],files):
  records=json.load(path.open())['records'];details=[];counts=collections.Counter();operations=collections.Counter();bycarrier={}
  for r in records:
   if r['class_id']!=1:continue
   a=alignment(PATTERN,r['collapsed_tokens']);a.update(index=r['index'],carrier=r['carrier'],event=bool(r['events_samples']));details.append(a);counts[a['distance']]+=1
   group=bycarrier.setdefault(r['carrier'],collections.Counter());group[a['distance']]+=1
   if not a['event']:
    for e in a['edits']:operations[f"{e['operation']}:{e['reference_position']}:{e['expected']}->{e['predicted']}"]+=1
  out['runs'][kind]=dict(distance_histogram=dict(counts),carrier_distance_histograms={k:dict(v) for k,v in bycarrier.items()},miss_edit_counts=dict(operations.most_common()),records=details)
 with (R/'4090-causal-target-edit-diagnostic-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 for k,v in out['runs'].items():print(k,v['distance_histogram'],list(v['miss_edit_counts'].items())[:12])
if __name__=='__main__':main()
