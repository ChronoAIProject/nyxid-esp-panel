"""One fixed shared-refractory merge of authenticated saved detector events."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
from uint8_event_policy_v1 import event_steps
R=Path('research/wakeword_p4/results');LOCK=R/'4090-c161-binary40-parallel-events-plan-v2.json';COHORTS=['builtin','daily','continuous','primed','jenny','joe','melo','mac']
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def report_rows(root):
 rec=read(root/'result.json');entry,=rec['models'];assert sha(entry['path'])==entry['sha256'];return read(entry['path'])['records']
def rows(c):
 root=Path(f'.local/wakeword/binary40-{c}-replay-v1');comparison=read(R/f'4090-binary40-{c}-comparison-v1.json');bresult=read(root/'result.json')
 if c in ['primed','jenny','joe','melo']:
  if c=='melo':
   a=[json.loads(s) for s in Path('.local/wakeword/melo-development-int8-v2/c161/records.jsonl').read_text().splitlines()];b=[json.loads(s) for s in (root/'binary40/records.jsonl').read_text().splitlines()]
  else:a=report_rows(Path(f'.local/wakeword/causal-source-balanced-{c}-v1'));b=report_rows(root)
  for old,aa,bb in zip(comparison['records'],a,b,strict=True):
   assert aa['index']==bb['index']==old['index'] and aa['record']==bb['record']==old['source'] and aa['phase']==bb['phase']==old['phase']
   clock='block_return_sample' if c=='melo' else 'real_block_return_sample';offset=81600 if c=='melo' else 0
   yield dict(index=old['index'],source=old['source'],phase=old['phase'],saved=old['events'],paths={n:dict(path=v['trace_path'],sha256=v['trace_sha256'],clock=clock,offset=offset) for n,v in [('c161',aa),('binary40',bb)]})
 elif c=='continuous':
  a=read(R/'4090-continuous-chatterbox-original-v1.json')['results'];b=bresult['results']
  for old,aa,bb in zip(comparison['records'],a,b,strict=True):
   assert aa['pcm_sha256']==bb['pcm_sha256']==old['pcm_sha256'] and aa['timeline']==bb['timeline']
   yield dict(index=aa['gap_samples'],source=dict(gap_samples=aa['gap_samples'],pcm_sha256=aa['pcm_sha256']),phase='continuous',timeline=aa['timeline'],saved={n:[{k:e[k] for k in ['step','relative_sample']} for e in row['models'][n]['events']] for n,row in [('c161',aa),('binary40',bb)]},paths={n:dict(path=row['models'][n]['trace_path'],sha256=row['models'][n]['trace_sha256'],clock='relative_block_return',offset=0) for n,row in [('c161',aa),('binary40',bb)]})
 else:
  a=read('.local/wakeword/c161-chatterbox-v1/result.json')['records'] if c=='builtin' else read(R/'4090-daily-language-pair-result-v1.json')['records'] if c=='daily' else read('.local/wakeword/c161-mac12-v1/result.json')['records']
  for old,aa,bb in zip(comparison['records'],a,bresult['records'],strict=True):
   assert old['index']==aa['index']==bb['index'];key='source' if c=='mac' else 'request';assert old[key]==aa[key]==bb[key]
   av=aa['models']['c161'] if c=='daily' else aa;bv=bb['models']['binary40'] if c!='mac' else bb
   ap=Path('.local/wakeword/c161-chatterbox-v1')/f'{aa["index"]:03d}.npz' if c=='builtin' else Path('.local/wakeword/daily-language-pair-v1')/f'{aa["index"]:03d}-c161.npz' if c=='daily' else Path('.local/wakeword/c161-mac12-v1')/f'{aa["index"]:03d}.npz'
   bp=root/(f'{aa["index"]:03d}.npz' if c=='mac' else f'{aa["index"]:03d}-binary40.npz')
   yield dict(index=old['index'],source=old[key],phase='synthetic',saved={'c161':[{k:e[k] for k in ['step','relative_sample']} for e in av['events']],'binary40':[{k:e[k] for k in ['step','relative_sample']} for e in old['events']]},paths={n:dict(path=str(path),sha256=v['trace_sha256'],clock='relative_block_return',offset=0) for n,path,v in [('c161',ap,av),('binary40',bp,bv)]})
def plan():
 files={};counts={}
 for c in COHORTS:
  items=list(rows(c));counts[c]=len(items)
  for row in items:
   for spec in row['paths'].values():assert sha(spec['path'])==spec['sha256'];files[spec['path']]=spec['sha256']
 for p in [Path(__file__),Path(__file__).with_name('uint8_event_policy_v1.py'),Path(__file__).with_name('c161_binary40_parallel_preflight_v1.md'),R/'4090-c161-binary40-parallel-input-original-v1.json',*[R/f'4090-binary40-{c}-comparison-v1.json' for c in COHORTS]]:files[str(p)]=sha(p)
 assert sum(counts.values())==1760
 p=dict(schema='c161-binary40-parallel-events-plan-v2',files=files,counts=counts,threshold=.91796875,smoothing_steps=5,shared_refractory_calls=67,policy='Fixed C161+originalBinary40 only, each original independent decoder/rearm unchanged; candidate events ordered by common native call step, C161 first on ties, shared emission requires>=67calls since last emitted event. Suppressed candidates do not reset shared cooldown or either decoder. Full complete consumed sources, authenticate both raw traces/clocks and reproduce original events before fusion. OR clip coverage diagnostic ceiling only; no parent/policy/threshold sweep, new model inference or optimization. Preserve C161 losses and everyday/background additions; no resource/P4 or promotion claim.')
 with LOCK.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
def merge(saved):
 candidates=sorted([dict(**e,model=n) for n,events in saved.items() for e in events],key=lambda e:(e['step'],e['model']!='c161'))
 emitted=[];suppressed=[];last=-1000
 for e in candidates:
  if e['step']-last>=67:emitted.append(e);last=e['step']
  else:suppressed.append(e)
 return emitted,suppressed
def main():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=read(LOCK)
 for f,h in p['files'].items():assert sha(f)==h,f
 cohorts={};started=time.monotonic()
 for c in COHORTS:
  records=[];groups={}
  for row in rows(c):
   clocks={}
   for n,spec in row['paths'].items():
    assert sha(spec['path'])==spec['sha256']
    with np.load(spec['path'],allow_pickle=False) as z:
     codes=z['uint8'];assert codes.dtype==np.uint8;clock=z[spec['clock']]-spec['offset'];clocks[n]=clock.copy();events=[dict(step=s,relative_sample=int(clock[s-1])) for s in event_steps(codes,.91796875)]
    assert events==row['saved'][n],(c,row['index'],n)
   assert np.array_equal(clocks['c161'],clocks['binary40']),(c,row['index'],'clock')
   emitted,suppressed=merge(row['saved']);out=dict(**row,emitted=emitted,suppressed=suppressed)
   if row['phase']=='continuous':
    containers={};outside={}
    for n,es in {**row['saved'],'parallel':emitted}.items():
     hit={};other=[]
     for e in es:
      owner=next(t for t in row['timeline'] if t['start']<e['relative_sample']<=t['end'])
      if owner['kind']=='clip':hit.setdefault(owner['group'],set()).add(owner['index'])
      else:other.append(dict(event=e,owner=owner))
     containers[n]={k:sorted(v) for k,v in hit.items()};outside[n]=other
    out.update(containers=containers,outside_clip_events=outside)
   else:
    source=row['source'];onset=source.get('lead_samples',0) if c in ['primed','jenny','joe'] and source.get('target_id')=='hey_nyx' else 0
    hits={n:any(e['relative_sample']>0 and e['relative_sample']>=onset for e in es) for n,es in {**row['saved'],'parallel':emitted}.items()};hits['or_ceiling']=hits['c161'] or hits['binary40'];out['hits']=hits
    key=source['group'] if c in ['builtin','daily'] else source['voice']+'/'+source['target'] if c=='mac' else 'background' if row['phase']=='background' else (source['voice_id'].split(':')[-1]+'/' if c=='melo' else '')+('target/' if (source.get('phrase_id')=='nyx' if c=='melo' else source.get('target_id')=='hey_nyx') else 'negative/'+source.get('phrase_id',source.get('target_id','unknown'))+'/')+source.get('carrier','whole-utterance-unclassified')
    g=groups.setdefault(key,dict(rows=0,hits={n:0 for n in hits},events={n:0 for n in ['c161','binary40','parallel']},gained_vs_c161=0,lost_vs_c161=0,hours=0.));g['rows']+=1
    for n,h in hits.items():g['hits'][n]+=h
    for n,es in {**row['saved'],'parallel':emitted}.items():g['events'][n]+=sum(e['relative_sample']>0 for e in es)
    g['gained_vs_c161']+=hits['parallel'] and not hits['c161'];g['lost_vs_c161']+=hits['c161'] and not hits['parallel']
    if key=='background':g['hours']+=source['samples']/16000/3600
   records.append(out)
  assert len(records)==p['counts'][c];cohorts[c]=dict(groups=groups,records=records)
 out=dict(schema='c161-binary40-parallel-events-original-v2',plan_sha256=sha(LOCK),cohorts=cohorts,total_rows=sum(len(v['records']) for v in cohorts.values()),elapsed_seconds=time.monotonic()-started,optimizer_updates=0,model_inference_calls=0,policy=p['policy'],limits='Saved detector-event simulation with exact paired native clocks. No measured dual-model streaming runtime/P4, independent acceptance or automatic promotion. Coverage OR is diagnostic ceiling, shared refractory may lose C161 hits. Reused backgrounds nonadditive.')
 with (R/'4090-c161-binary40-parallel-events-original-v2.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(dict(completed=True,total_rows=out['total_rows'],elapsed_seconds=out['elapsed_seconds'])))
if __name__=='__main__':{'plan':plan,'run':main}[sys.argv[1]]()
