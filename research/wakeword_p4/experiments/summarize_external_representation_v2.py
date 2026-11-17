"""Complete reviewed cohorts only; preserve ordinary-hit trades, no selection."""
import hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results')
COHORTS=['builtin','daily','continuous','primed','jenny','joe','melo','mac']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 sources={}
 def load(p):
  p=Path(p);sources[str(p)]=sha(p);return json.loads(p.read_text())
 review=load(R/'4090-external-representation-consumed-review-v1.json');assert review['all_passed'] and set(review['arms'])=={'control','feature'}
 previous=load(R/'4090-frozen-history-teacher-consumed-review-v1.json');assert previous['all_passed']
 baseline=load(R/'4090-binary40-final-summary-v1.json')
 for path,h in baseline['source_files'].items():assert sha(path)==h,path
 reports={};models={}
 for name,prefix in [('original_binary40','4090-binary40'),('prior_teacher','4090-frozen-history-teacher-teacher'),('control','4090-external-representation-control'),('feature','4090-external-representation-feature')]:
  reports[name]={c:load(R/f'{prefix}-{c}-comparison-v1.json') for c in COHORTS}
  if name in ['control','feature']:
   for c,d in reports[name].items():assert review['arms'][name][c]['comparison_sha256']==sha(R/f'{prefix}-{c}-comparison-v1.json')
   export=load(R/f'{prefix}-export-original-v1.json');assert sha(export['model']['path'])==export['model']['sha256'];models[name]=export['model']
  if name=='prior_teacher':
   for c in COHORTS:assert previous['arms']['teacher'][c]['comparison_sha256']==sha(R/f'{prefix}-{c}-comparison-v1.json')
 def metrics(data,model='binary40'):
  out={}
  for c in ['primed','jenny','joe','melo']:
   for key,g in data[c]['groups'].items():
    if not (key.startswith('target/') or '/target/' in key):continue
    scope=c+('/'+key.split('/')[0] if c=='melo' else '')
    kind='unclassified' if key.endswith('whole-utterance-unclassified') else 'bare' if key.endswith('target/') or key.endswith('target/bare') else 'commands'
    for suffix in ['all',kind]:
     entry=out.setdefault(scope+'/'+suffix,dict(hits=0,clips=0));entry['hits']+=g['hits'][model];entry['clips']+=g['clips']
  return out
 table={'c161':metrics(reports['original_binary40'],'c161')}
 for name,data in reports.items():
  out=metrics(data)
  out['builtin']=dict(hits=data['builtin']['groups']['nyx']['hits'],clips=12)
  out['daily_conflict_clips']=dict(hits=sum(g['hits'] for g in data['daily']['groups'].values()),clips=96)
  for row in data['continuous']['records']:out[f'continuous_gap{row["gap_samples"]}']=dict(hits=len(row['containers'].get('nyx',[])),clips=12)
  out['mac_host']=dict(hits=sum(g['hits'] for key,g in data['mac']['groups'].items() if key.endswith('/hey_nyx')),clips=4)
  table[name]=out
 c=table['c161'];c['builtin']=dict(hits=baseline['baseline_builtin']['nyx']['activated'],clips=12)
 c['daily_conflict_clips']=dict(hits=sum(g['c161_activated'] for g in baseline['baseline_daily'].values()),clips=96)
 c['mac_host']=dict(hits=baseline['baseline_mac_target_hits'],clips=4)
 for row in baseline['baseline_continuous']:c[f'continuous_gap{row["gap_samples"]}']=dict(hits=row['target_containers'],clips=12)
 assert all(set(v)==set(c) and all(v[k]['clips']==c[k]['clips'] for k in c) for v in table.values())
 trades={};background={};daily={}
 for arm in ['control','feature']:
  data=reports[arm];trades[arm]={};background[arm]={};daily[arm]=data['daily']['records']
  for cohort in COHORTS:
   original=data[cohort]['records']
   for other in ['original_binary40','prior_teacher']:
    before=reports[other][cohort]['records'];assert len(original)==len(before)
    changes=[]
    for after,old in zip(original,before):
     if cohort in ['primed','jenny','joe','melo']:
      assert after['source']==old['source'] and after['phase']==old['phase'] and after['index']==old['index']
      if after['hits'] is not None and after['hits']['binary40']!=old['hits']['binary40']:changes.append(dict(index=after['index'],source=after['source'],before=old['hits']['binary40'],after=after['hits']['binary40']))
     elif cohort=='continuous':
      assert after['pcm_sha256']==old['pcm_sha256'] and after['gap_samples']==old['gap_samples']
      groups=set(after['containers'])|set(old['containers'])
      changes.append(dict(gap_samples=after['gap_samples'],groups={g:dict(gained=sorted(set(after['containers'].get(g,[]))-set(old['containers'].get(g,[]))),lost=sorted(set(old['containers'].get(g,[]))-set(after['containers'].get(g,[])))) for g in sorted(groups)},outside_clip_events=after['outside_clip_events']))
     else:
      identity='source' if cohort=='mac' else 'request';assert after['index']==old['index'] and after[identity]==old[identity]
      a=any(e['relative_sample']>0 for e in after['events']);b=any(e['relative_sample']>0 for e in old['events'])
      if a!=b:changes.append(dict(index=after['index'],source=after[identity],before=b,after=a))
    trades[arm][cohort+'/'+other]=changes
   if cohort in ['primed','jenny','joe','melo']:background[arm][cohort]=data[cohort]['reused_background']
 limits='Consumed synthetic evidence only; no automatic promotion, pooled reused background hours, independent acceptance, human-generalization, phonetic endpoint, first-command retention or P4 claim. Other/nearword groups remain in complete original reports and are not a zero-error gate. Historical teacher is a separate intervention; only fresh control versus feature is the matched causal external-feature comparison.'
 out=dict(schema='external-representation-final-summary-v1',review_status='complete-reviewed-evidence-awaits-research-decision',models=models,metrics=table,changes=trades,daily_records=daily,reused_background=background,source_files={**sources,str(Path(__file__)):sha(__file__)},limits=limits)
 dest=R/'4090-external-representation-final-summary-v1.json'
 with dest.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 lines=['# Matched external-feature complete comparison','',limits,'','| Metric | C161 | Original Binary40 | Prior teacher | Control | Feature |','|---|---:|---:|---:|---:|---:|']
 for metric in sorted(c):lines.append('| '+metric+' | '+' | '.join(f'{table[n][metric]["hits"]}/{table[n][metric]["clips"]}' for n in ['c161','original_binary40','prior_teacher','control','feature'])+' |')
 lines+=['','Exact clip gains/losses, daily utterances, each reused background set and source hashes are preserved in '+dest.name+'. No automatic model selection.']
 with (R/'4090-external-representation-final-table-v1.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 print('\n'.join(lines))
if __name__=='__main__':main()
