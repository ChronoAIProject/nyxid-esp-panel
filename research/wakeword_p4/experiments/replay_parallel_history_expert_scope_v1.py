"""All1760 consumed raw-stream identities, one fixed third-parent event policy."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import hashlib,io,json,subprocess,sys,tarfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from uint8_event_policy_v1 import event_steps
from preflight_parallel_history_expert_v1 import merge
R=Path('research/wakeword_p4/results');PLAN=R/'4090-parallel-history-expert-scope-plan-v1.json';BASE=R/'4090-c161-binary40-parallel-events-original-v2.json';COHORTS=['builtin','daily','continuous','primed','jenny','joe','melo','mac'];PRIORITY=['c161','binary40','history_expert']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def projection(es):return [{k:e[k] for k in ['step','relative_sample']} for e in es]
def lock():
 assert read(R/'4090-parallel-history-expert-preflight-review-v1.json')['bounded_gate_passed'];files=[Path(__file__),Path(__file__).with_name('uint8_event_policy_v1.py'),Path(__file__).with_name('preflight_parallel_history_expert_v1.py'),BASE,R/'4090-parallel-history-expert-preflight-plan-v1.json',R/'4090-parallel-history-expert-preflight-review-v1.json',R/'4090-c161-binary40-parallel-runtime-review-v1.json',R/'4090-frozen-history-teacher-consumed-review-v1.json'];export=R/'4090-frozen-history-teacher-teacher-export-original-v1.json';files.extend([export,Path(read(export)['model']['path'])])
 for c in COHORTS:
  cp=R/f'4090-frozen-history-teacher-teacher-{c}-comparison-v1.json';rp=R/f'4090-frozen-history-teacher-teacher-{c}-replay-plan-v1.json';d=read(cp);a=d['archive'];assert sha(a['path'])==a['sha256'];assert read(rp)['files'][read(export)['model']['path']]==read(export)['model']['sha256'];files.extend([cp,rp,Path(a['path'])])
 p=dict(schema='parallel-history-expert-scope-plan-v1',files={str(p):sha(p) for p in files},priority=PRIORITY,rows=1760,policy='Exactly original C161/originalB40/frozen-history-teacher-treatment; own235/256/5/67/half-rearm, native-step fixed priority C161,B40,history, shared emitted67call cooldown without suppression reset/extension. All8cohorts/1760 consumed raw traces, no inference, optimizer, model/parent/threshold/order/gap sweep or selection. Authenticate original source identities, uint8 clocks/own events and original dual arbitration before triple merge. Preserve all target gains/losses and everyday/other-target/constructed/background events with separate reused hours. No new exposure/P4/resources/promotion or actual triple execution. No fourth expert selected.')
 with PLAN.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
def member(c,row):
 i=row['index'];return f'{i}-binary40.npz' if c=='continuous' else f'binary40/{i:04d}.npz' if c=='melo' else f'hey_nyx-binary40/{row["phase"]}/{i:06d}.npz' if c in ['primed','jenny','joe'] else f'{i:03d}.npz' if c=='mac' else f'{i:03d}-binary40.npz'
def positive(c,s):return s.get('group')=='nyx' if c=='builtin' else s.get('phrase_id')=='nyx' if c=='melo' else s.get('target',s.get('target_id'))=='hey_nyx'
def key(c,row):
 s=row['source']
 if row['phase']=='background':return 'background'
 if c in ['builtin','daily']:return s['group']
 role='target' if positive(c,s) else 'negative/'+str(s.get('phrase_id',s.get('target_id',s.get('target','unknown'))))
 voice=s['voice_id'].split(':')[-1]+'/' if c=='melo' else s['voice']+'/' if c=='mac' else ''
 return voice+role+'/'+(s.get('carrier',s.get('form','unclassified')) or 'bare')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=read(PLAN)
 for f,h in p['files'].items():assert sha(f)==h,f
 baseline=read(BASE);cohorts={};total=0
 for c in COHORTS:
  comparison=read(R/f'4090-frozen-history-teacher-teacher-{c}-comparison-v1.json');archive=comparison['archive'];rows=[];groups=defaultdict(Counter)
  with tarfile.open(archive['path'],'r:gz') as tf:
   assert len(tf.getnames())==len(set(tf.getnames())) and set(tf.getnames())==set(archive['members']);result=json.load(tf.extractfile('result.json'))
   for row,original in zip(baseline['cohorts'][c]['records'],comparison['records'],strict=True):
    if c=='continuous':assert row['index']==original['gap_samples'] and row['source']['pcm_sha256']==original['pcm_sha256']
    else:
     assert row['index']==original['index'];sourcekey='request' if c in ['builtin','daily'] else 'source';assert row['source']==original[sourcekey]
     if c in ['primed','jenny','joe','melo']:assert row['phase']==original['phase']
    path=member(c,row);raw=tf.extractfile(path).read();spec=archive['members'][path];assert len(raw)==spec['bytes'] and hashlib.sha256(raw).hexdigest()==spec['sha256']
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:
     codes=z['uint8'].copy();clockkey='block_return_sample' if c=='melo' else 'real_block_return_sample' if c in ['primed','jenny','joe'] else 'relative_block_return';clock=z[clockkey]-(81600 if c=='melo' else 0)
    assert codes.dtype==np.uint8;teacher=[dict(step=int(s),relative_sample=int(clock[s-1])) for s in event_steps(codes,.91796875)]
    if c=='continuous':
     ref=next(r for r in result['results'] if r['gap_samples']==row['index']);assert ref['timeline']==row['timeline'];saved=projection(ref['models']['binary40']['events'])
    else:saved=projection(original['events']['binary40'] if c in ['primed','jenny','joe','melo'] else original['events'])
    assert teacher==saved,(c,row['index'],'teacher events')
    for name,sp in row['paths'].items():
     assert sha(sp['path'])==sp['sha256']
     with np.load(sp['path'],allow_pickle=False) as z:
      assert np.array_equal(z[sp['clock']]-sp['offset'],clock);es=[dict(step=int(s),relative_sample=int(clock[s-1])) for s in event_steps(z['uint8'],.91796875)];assert es==row['saved'][name]
    assert merge(row['saved'],PRIORITY[:2])==(row['emitted'],row['suppressed'])
    own=dict(**row['saved'],history_expert=teacher);emitted,suppressed=merge(own,PRIORITY);out=dict(index=row['index'],source=row['source'],phase=row['phase'],saved=own,dual_emitted=row['emitted'],emitted=emitted,suppressed=suppressed,teacher_trace=dict(archive=archive['path'],member=path,sha256=spec['sha256']))
    all_events={**own,'dual':row['emitted'],'triple':emitted}
    if c=='continuous':
     containers={};outside={}
     for n,events in all_events.items():
      hits=defaultdict(set);extra=[]
      for e in events:
       owner=next(t for t in row['timeline'] if t['start']<e['relative_sample']<=t['end'])
       if owner['kind']=='clip':hits[owner['group']].add(owner['index'])
       else:extra.append(dict(event=e,owner=owner))
      containers[n]={g:sorted(v) for g,v in hits.items()};outside[n]=extra
     out.update(timeline=row['timeline'],containers=containers,outside_clip_events=outside)
    else:
     s=row['source'];onset=s.get('lead_samples',0) if c in ['primed','jenny','joe'] and s.get('target_id')=='hey_nyx' else 0;hits={n:any(e['relative_sample']>0 and e['relative_sample']>=onset for e in es) for n,es in all_events.items()};assert all(hits[n]==row['hits'][n] for n in ['c161','binary40']) and hits['dual']==row['hits']['parallel'];out['hits']=hits
     g=groups[key(c,row)];g['clips']+=1
     for n,h in hits.items():g[n+'_hits']+=h;g[n+'_events']+=sum(e['relative_sample']>0 for e in all_events[n])
     g['gains_vs_dual']+=hits['triple'] and not hits['dual'];g['losses_vs_dual']+=hits['dual'] and not hits['triple']
     if row['phase']=='background':g['reused_seconds']+=s['samples']/16000
    rows.append(out);total+=1
  cohorts[c]=dict(groups=groups,records=rows);print(json.dumps(dict(cohort=c,rows=len(rows),total=total)),flush=True)
 assert total==1760
 out=dict(schema='parallel-history-expert-scope-original-v1',plan_sha256=sha(PLAN),cohorts=cohorts,total_rows=total,optimizer_updates=0,model_inference_calls=0,policy=p['policy'])
 with (R/'4090-parallel-history-expert-scope-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
