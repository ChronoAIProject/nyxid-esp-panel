"""Compare complete paired consumed recordings, no new neural calls."""
import collections,gzip,hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 files={str(Path(__file__)):sha(__file__)};reports={};records={}
 for arm,stem in [('original','matchbox-long-pooled'),('aligned','matchbox-grid-aligned')]:
  report=R/f'4090-{stem}-consumed-melo-original-v1.json';r=json.loads(report.read_text());assert r['all474streams_complete'] and r['metrics']['threshold']==.91796875
  log=B/f'{stem}-consumed-melo-v1/records.jsonl';assert sha(log)==r['records_sha256'];rows=[json.loads(l) for l in log.open()];assert len(rows)==474
  counts=collections.defaultdict(lambda:dict(clips=0,activated=0));early=0
  for i,row in enumerate(rows):
   assert row['index']==i and sha(row['trace_path'])==row['trace_sha256'];rec=row['record'];real=[e for e in row['events'] if not e['priming']]
   if rec['phase']=='synthetic':
    key=rec['voice_id']+'/'+rec['phrase_id']+'/'+rec['carrier'];counts[key]['clips']+=1;counts[key]['activated']+=int(any(e['sample']>=rec['lead_samples'] for e in real));early+=sum(e['sample']<rec['lead_samples'] for e in real)
  reports[arm]=dict(model=r['model'],metrics=r['metrics'],background_hours=r['background_hours'],far_upper95=r['background_far_upper95'],by_voice_phrase_carrier=dict(counts),pre_source_events=early)
  records[arm]=rows;files.update({str(report):sha(report),str(log):sha(log)})
  if arm=='aligned':
   out=R/'4090-matchbox-grid-aligned-consumed-melo-records-v1.jsonl.gz'
   if out.exists():
    with gzip.open(out,'rb') as f:assert f.read()==log.read_bytes()
   else:
    with gzip.open(out,'wb') as f:f.write(log.read_bytes())
   files[str(out)]=sha(out)
 delta=collections.defaultdict(collections.Counter)
 for a,b in zip(records['original'],records['aligned'],strict=True):
  assert a['record']==b['record'];r=a['record'];lead=r.get('lead_samples',0)
  hit=lambda x:any(not e['priming'] and e['sample']>=lead for e in x['events'])
  x,y=hit(a),hit(b);delta[r.get('phrase_id','background')]['both' if x and y else 'original_only' if x else 'aligned_only' if y else 'neither']+=1
 result=dict(schema='matchbox-grid-behavior-comparison-v1',files=files,arms=reports,paired_delta={k:dict(v) for k,v in delta.items()},all948trace_hashes_verified=True,neural_calls=0,threshold_selected=False,promotion_qualified=False,limits='Consumed fixed-point diagnosis only. Same complete background recordings, hours not additive or fresh acceptance. Source+RIR generated boundaries not phonetic truth. No threshold rescue, phase causality, P4 feasibility or HiESP parity claim.')
 with (R/'4090-matchbox-grid-behavior-comparison-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(metrics={k:v['metrics'] for k,v in reports.items()},paired_delta=result['paired_delta'])))
if __name__=='__main__':main()
