"""Complete36target saved-byte diagnosis, with unfiltered original ASR context."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');PLAN=R/'4090-parallel-history-context-gaps-plan-v1.json';NAMES=['c161','binary40','history_expert']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lock():
 names=['parallel-history-expert-runtime-original','parallel-history-expert-runtime-review','parallel-context-streams-plan','parallel-prospective-asr-original','parallel-melo-context-asr-original'];files=[Path(__file__)]+[R/f'4090-{n}-v1.json' for n in names];runtime=read(files[1]);spec=runtime['cohorts']['context'];files.append(Path(spec['path']))
 for row in read(spec['path'])['records']:files.append(Path(row['trace']['path']))
 for n in ['parallel-prospective','parallel-melo-context']:
  p=R/f'4090-{n}-asr-plan-v1.json';files.append(p)
  for row in read(R/f'4090-{n}-asr-original-v1.json')['rows']:files.append(Path(row['wav']))
 with PLAN.open('x') as f:json.dump(dict(schema='parallel-history-context-gaps-plan-v1',files={str(p):sha(p) for p in files},scope='All36target containers in all3existing physical continuous context streams. Exact trailing-five sums independently cumsum/convolve checked; count per-detector crossings/events and shared emitted/suppressed, include complete original isolated dual events and unfiltered source ASR text/words. No inference, crop, resynthesis, new labels, threshold/policy/source selection or training. ASR is corroboration only, not acoustic truth or first-command/phone timing. Consumeddiagnostic, no FAR/P4/independent qualification.'),f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=read(PLAN)
 for path,h in p['files'].items():assert sha(path)==h,path
 assert read(R/'4090-parallel-history-expert-runtime-review-v1.json')['all_passed'];runtime=read(R/'4090-parallel-history-expert-runtime-original-v1.json');rows=read(runtime['cohorts']['context']['path'])['records'];source=read(R/'4090-parallel-context-streams-plan-v1.json');asr={n:read(R/f'4090-{n}-asr-original-v1.json')['rows'] for n in ['parallel-prospective','parallel-melo-context']};out=[];summary=[]
 for row,stream in zip(rows,source['streams'],strict=True):
  assert row['name']==stream['name'];name=row['name'];condition=[]
  with np.load(row['trace']['path'],allow_pickle=False) as z:
   clock=z['relative_block_return'];sums={}
   for n in NAMES:
    codes=z[n+'_uint8'];prefix=np.r_[0,np.cumsum(codes,dtype=np.int64)];sums[n]=prefix[5:]-prefix[:-5];assert np.array_equal(sums[n],np.convolve(codes.astype(np.int64),np.ones(5,np.int64),mode='valid'))
   for t in row['timeline']:
    if t['kind']!='clip' or t['group']!='nyx_connected_command':continue
    i=t['text_index'];item=next(x for x in stream['items'] if x['text_index']==i);a=asr['parallel-prospective'][i] if name=='builtin' else asr['parallel-melo-context'][i+(48 if name=='melo-BR' else 0)];assert a['index']==item['source_index'] and a['wav']==item['wav'] and a['request']['text']==item['request']['text'];mask=(clock[4:]>t['start'])&(clock[4:]<=t['end']);indices=np.flatnonzero(mask);assert len(indices)
    model={}
    for n in NAMES:
     peak=indices[np.argmax(sums[n][indices])];crossings=(np.flatnonzero(mask&(sums[n]>=1175))+5).tolist();events=[e for e in row['events'][n] if t['start']<e['relative_sample']<=t['end']];model[n]=dict(max_five_code_sum=int(sums[n][peak]),max_sum_native_step=int(peak+5),max_sum_clip_relative_sample=int(clock[peak+4]-t['start']),threshold_crossing_steps=crossings,own_events=events,isolated_original_events=item['isolated_events'].get(n) if n!='history_expert' else None)
    emitted=[e for e in row['emitted'] if t['start']<e['relative_sample']<=t['end']];suppressed=[e for e in row['suppressed'] if t['start']<e['relative_sample']<=t['end']];words=[w for seg in a['segments'] for w in seg.get('words',[])];text=''.join(seg['text'] for seg in a['segments']);record=dict(condition=name,text_index=i,source_index=item['source_index'],intended_text=item['request']['text'],pcm_sha256=item['pcm_sha256'],container=t,asr=dict(text=text,words=words,reference_only_not_phonetic_truth=True),models=model,emitted=emitted,suppressed=suppressed,hit=bool(emitted),all_models_without_crossing=all(not x['threshold_crossing_steps'] for x in model.values()),emitted_clip_relative_samples=[e['relative_sample']-t['start'] for e in emitted],isolated_dual_hit=any(e['relative_sample']>0 for e in item['isolated_emitted']));out.append(record);condition.append(record)
  assert len(condition)==12;misses=[x for x in condition if not x['hit']];summary.append(dict(condition=name,hits=sum(x['hit'] for x in condition),miss_text_indices=[x['text_index'] for x in misses],misses_without_any_model_crossing=sum(x['all_models_without_crossing'] for x in misses),misses_with_own_events=sum(any(v['own_events'] for v in x['models'].values()) for x in misses),isolated_dual_hits_lost=[x['text_index'] for x in misses if x['isolated_dual_hit']],isolated_dual_misses_recovered=[x['text_index'] for x in condition if x['hit'] and not x['isolated_dual_hit']]))
 assert len(out)==36
 result=dict(schema='parallel-history-context-gaps-original-v1',plan_sha256=sha(PLAN),all_passed=True,summary=summary,records=out,optimizer_updates=0,inference_calls=0,limits=p['scope'])
 with (R/'4090-parallel-history-context-gaps-original-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(summary),flush=True)
 for x in out:
  if not x['hit']:print(json.dumps(dict(condition=x['condition'],text_index=x['text_index'],text=x['intended_text'],asr=x['asr']['text'],max_sums={n:v['max_five_code_sum'] for n,v in x['models'].items()},isolated_dual_hit=x['isolated_dual_hit'])),flush=True)
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
