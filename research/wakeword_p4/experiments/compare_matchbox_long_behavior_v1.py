"""Summarize two complete fixed-policy reports and paired saved events only."""
import collections,gzip,hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 reports={};records={};files={str(Path(__file__)):sha(Path(__file__))}
 for arm in ['control','pooled']:
  p=R/f'4090-matchbox-long-{arm}-consumed-melo-original-v1.json'
  if not p.exists():print(json.dumps(dict(status='not_ready',arm=arm)));return
  r=json.loads(p.read_text());assert r['all474streams_complete'] and r['metrics']['threshold']==.91796875
  source=B/f'matchbox-long-{arm}-consumed-melo-v1/records.jsonl';assert sha(source)==r['records_sha256']
  rows=[json.loads(l) for l in source.open()];assert len(rows)==474
  counts=collections.defaultdict(lambda:dict(clips=0,activated=0));states=[]
  for index,row in enumerate(rows):
   assert row['index']==index and sha(row['trace_path'])==row['trace_sha256']
   real=[e for e in row['events'] if not e['priming']];rec=row['record'];states.append(bool(real))
   if rec['phase']=='synthetic':
    k=rec['phrase_id']+'/'+rec['carrier'];counts[k]['clips']+=1;counts[k]['activated']+=int(bool(real))
  reports[arm]=dict(model=r['model'],metrics=r['metrics'],background_hours=r['background_hours'],background_far_upper95=r['background_far_upper95'],by_phrase_carrier=dict(counts),elapsed_seconds=r['elapsed_seconds'])
  records[arm]=rows;files[str(p)]=sha(p);files[str(source)]=sha(source)
  out=R/f'4090-matchbox-long-{arm}-consumed-melo-records-v1.jsonl.gz'
  if out.exists():
   with gzip.open(out,'rb') as f:assert f.read()==source.read_bytes()
  else:
   with gzip.open(out,'wb') as f:f.write(source.read_bytes())
  files[str(out)]=sha(out)
 delta=collections.defaultdict(collections.Counter)
 for a,b in zip(records['control'],records['pooled'],strict=True):
  assert a['record']==b['record']
  rec=a['record'];ka=bool([e for e in a['events'] if not e['priming']]);kb=bool([e for e in b['events'] if not e['priming']])
  key=rec.get('phrase_id','background')
  delta[key]['both' if ka and kb else 'control_only' if ka else 'pooled_only' if kb else 'neither']+=1
 result=dict(schema='matchbox-long-behavior-comparison-v1',files=files,arms=reports,paired_activation_delta={k:dict(v) for k,v in delta.items()},neural_calls=0,threshold_selected=False,acceptance_accessed=False,candidate_promoted=False,limits='Full consumed diagnostic at one fixed point. Paired event activation is not source-independent population inference; events before generated source onset remain visible in original receipts. Whole reused background hours not additive. No confidence claim from correlated synthetic variants, P4 resource/latency proof or HiESP parity. Original source and execution/restart limitations retained. No rescue sweep or protected acceptance opening.')
 with (R/'4090-matchbox-long-behavior-comparison-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(arms=reports,paired_activation_delta=result['paired_activation_delta'])))
if __name__=='__main__':main()
