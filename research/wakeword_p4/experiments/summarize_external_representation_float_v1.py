"""Summarize complete independently decoded float/INT8 evidence without selection."""
import hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 sources={}
 def load(p):
  p=Path(p);sources[str(p)]=sha(p);return json.loads(p.read_text())
 review=load(R/'4090-external-representation-float-review-v1.json');assert review['all_passed']
 terminal=load(R/'4090-external-representation-float-supervision-original-v1.json');assert terminal['status']=='completed-await-research-decision'
 assert [s['name'] for s in terminal['stages']]==['float-control','float-feature','independent-float-review']
 for stage in terminal['stages']:assert stage['actual_exit_code']==0 and sha(stage['log'])==stage['log_sha256']
 table={};background={};changes={}
 for arm in ['control','feature']:
  p=R/f'4090-external-representation-{arm}-float-replay-original-v1.json';report=load(p);assert sha(p)==review['arms'][arm]['report_sha256']
  assert review['arms'][arm]['rows']==review['arms'][arm]['unique_traces']==1760
  archive=load(R/f'4090-external-representation-{arm}-float-archive-review-v1.json');assert archive==terminal['archives'][arm]
  assert sha(archive['path'])==archive['sha256'] and Path(archive['path']).stat().st_size==archive['bytes']
  metrics={};background[arm]={};changes[arm]={}
  def add(key,g):
   row=metrics.setdefault(key,dict(rows=0,float=0,int8=0));row['rows']+=g['rows']
   for name in ['float','int8']:row[name]+=g['hits'][name]
  for cohort,spec in report['cohorts'].items():
   d=load(spec['path']);assert sources[spec['path']]==spec['sha256']
   changes[arm][cohort]=review['arms'][arm]['cohorts'][cohort]['float_int8_hit_differences']
   if cohort=='continuous':
    for row in d['records']:
     metrics[f'continuous_gap{row["source"]["gap_samples"]}']=dict(rows=12,**{name:len(row['containers'][name].get('nyx',[])) for name in ['float','int8']})
   for key,g in d['groups'].items():
    if cohort=='builtin' and key=='nyx':add('builtin',g)
    elif cohort=='daily':add('daily_conflict_clips',g)
    elif cohort=='mac' and key.endswith('/hey_nyx'):add('mac_host',g)
    elif cohort in ['primed','jenny','joe','melo'] and (key.startswith('target/') or '/target/' in key):
     parts=key.split('/');scope=cohort+('/'+parts[0] if cohort=='melo' else '')
     carrier=parts[-1];kind='unclassified' if carrier=='whole-utterance-unclassified' else 'bare' if carrier in ['', 'bare'] else 'commands'
     for suffix in ['all',kind]:add(scope+'/'+suffix,g)
    elif key=='background':background[arm][cohort]=g
  table[arm]=metrics
 limits='Complete consumed synthetic float attribution, no optimization or tuning. Direct float scores compared with saved integer codes under fixed policy; no independent exposure, acoustic-cause proof, human or P4 claim. Reused backgrounds separate/nonadditive; no automatic promotion.'
 out=dict(schema='external-representation-float-attribution-summary-v1',metrics=table,reused_background=background,hit_differences=changes,source_files=sources,limits=limits)
 p=R/'4090-external-representation-float-summary-v1.json'
 with p.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 lines=['# External feature complete float attribution','',limits,'','| Metric | Control float | Control INT8 | Feature float | Feature INT8 |','|---|---:|---:|---:|---:|']
 assert set(table['control'])==set(table['feature'])
 for key in sorted(table['control']):
  a,b=table['control'][key],table['feature'][key];assert a['rows']==b['rows']
  lines.append('| '+key+' | '+' | '.join(f'{m[n]}/{m["rows"]}' for m,n in [(a,'float'),(a,'int8'),(b,'float'),(b,'int8')])+' |')
 with (R/'4090-external-representation-float-table-v1.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 print('\n'.join(lines))
if __name__=='__main__':main()
