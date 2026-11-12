"""Audit pinned v44 diagnostic metadata, without model or array replay.

Run at the repository root after fetching the GPU research branch; JSON goes
to stdout. No audio/device operation or unused-development/final access.
"""
from pathlib import Path
import collections,hashlib,json,math,subprocess
ref='9f166711b4d0845040a490da5b22f5d6b61ad7b4'
inputs={}
def report(name):
 raw=subprocess.check_output(['git','show',ref+':research/wakeword_p4/results/'+name]);inputs[name]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
def require(value,message):
 if not value:raise ValueError(message)
dev=report('4090-v44-old-development-float-probe-v1.json');corr=report('4090-v44-development-training-correspondence-v1.json');old=report('4090-v44-training-support-v1.json');us=report('4090-v44-us-training-support-v1.json');center=report('4090-v44-us-center-fit-v1.json');pack=report('4090-us-training-features-v1.json');export=report('4090-v44-export-audit-v1.json')
require(corr['inputs']['development_report_sha256']==inputs['4090-v44-old-development-float-probe-v1.json'] and corr['inputs']['training_report_sha256']==inputs['4090-v44-training-support-v1.json'],'correspondence input links')
for key,name in [('dev','v44_old_development_float_probe.py'),('corr','v44_development_training_correspondence.py'),('old','v44_training_support_probe.py'),('us','v44_us_training_support_probe.py'),('center','v44_us_center_fit_probe.py')]:
 raw=subprocess.check_output(['git','show',ref+':research/wakeword_p4/experiments/'+name]);require(locals()[key]['script_sha256']==hashlib.sha256(raw).hexdigest(),'script hash '+name)
require(all(r['model_sha256']==export['model_sha256'] for r in [dev,corr,old,us,center]),'model identities')
require(center['weights_sha256']==export['weights_sha256'],'center weights link')
require(len(dev['details'])==380 and len(old['details'])==6597,'complete cohort counts')
require(us['pack_report_sha256']==inputs['4090-us-training-features-v1.json'],'US original pack report link')
require(len(us['details'])==len(center['details'])==len(pack['records'])==324,'complete US count')
decisions=collections.Counter();witnesses={};protocol_summary={}
for tag,name in [('phone','4090-v44-phone-development-v1.json'),('vctk','4090-v44-vctk-development-v1.json')]:
 original=report(name);details=[d for d in dev['details'] if d['protocol']==tag];require(len(original['records'])==len(details),'protocol count')
 for index,(d,saved) in enumerate(zip(details,original['records'],strict=True)):
  require(d['index']==index and d['record']==saved['record'],'original identity')
  times=[(3*(i+4)+5)*.01 for i in d['int8_events']];require(len(times)==len(saved['nyx']['event_seconds']) and all(math.isclose(a,b,rel_tol=0,abs_tol=1e-12) for a,b in zip(times,saved['nyx']['event_seconds'],strict=True)),'INT8 saved event accounting')
  if d['record']['target']!='hey_nyx':
   f,q=bool(d['float_events']),bool(d['int8_events']);decision='both' if f and q else 'float_only' if f else 'int8_only' if q else 'neither';decisions[decision]+=1
   if f or q:witnesses[(tag,index)]=(d,decision)
require(dict(decisions)==corr['negative_decisions'] and len(witnesses)==len(corr['negative_witnesses'])==13,'failure intersection')
for witness in corr['negative_witnesses']:
 d,decision=witnesses[(witness['protocol'],witness['index'])]
 require(witness['decision']==decision and all(witness[k]==d[k] for k in ['float_max_five','int8_max_five','float_events','int8_events']) and witness['source_sha256']==d['record']['source_sha256'] and witness['path']==d['record']['path'],'witness identity/value')
threshold=.99609375
us_groups={};center_groups={};failures=[]
for index,(d,c,r) in enumerate(zip(us['details'],center['details'],pack['records'],strict=True)):
 require(d['index']==c['index']==r['pack_index']==index and all(d[k]==c[k]==r[k] for k in ['voice','target','source','variant']),'US index/source identity')
 require(d['label']==int(r['target']=='hey_nyx'),'US label')
 for group in ['all/'+str(d['label']),d['voice']+'/'+d['target']]:
  g=us_groups.setdefault(group,dict(variants=0,zero_support=0,modes={}));g['variants']+=1;g['zero_support']+=int(d['valid_support_positions']==0)
  cg=center_groups.setdefault(group,dict(variants=0,high={t:0 for t in ['0.5','0.992','0.99609375']}));cg['variants']+=1
  for t in cg['high']:cg['high'][t]+=int(c['score']>=float(t))
  for mode,v in d['modes'].items():
   for t,h in v['thresholds'].items():
    require(h['full_high']==(v['full_max_five']>=float(t)),'US score threshold')
    s=g['modes'].setdefault(mode+'/'+t,dict(full_high=0,support_hit_variants=0,support_high_positions=0,event_clips=0));s['full_high']+=int(h['full_high']);s['support_hit_variants']+=int(h['support_high_positions']>0);s['support_high_positions']+=h['support_high_positions'];s['event_clips']+=int(bool(h['event_indices']))
 if not d['label'] and d['modes']['int8']['thresholds'][str(threshold)]['full_high']:
  failures.append({k:d[k] for k in ['index','voice','source','target','variant']})
require(us_groups==us['groups'] and center_groups==center['groups'],'US all group accounting')
negative=[d for d in old['details'] if d['label']==0]
require(len(negative)==1995 and not any(d['window_float']>=threshold or d['full_int8_max_five']>=threshold for d in negative),'old training negative support')
out=dict(scope='Independent diagnostic report/hash/identity/count verification only; no array or model replay',gpu_commit=ref,input_sha256=inputs,old_development_records=380,negative_decisions=dict(decisions),witnesses_checked=13,old_training_variants=6597,old_training_negative_variants=1995,old_training_negative_high=0,new_US_variants=324,new_US_center_groups={k:center_groups[k] for k in ['all/0','all/1']},new_US_stream_groups={k:us_groups[k] for k in ['all/0','all/1']},new_US_int8_high_negative_identities=failures,limits='Saved metadata and event/hash links only. No Mac array/weight/saved-model loading or independent replay of GPU model/trace bytes; no new causal/phonetic/generalization conclusion, unused development/final data or device/audio operation.')
witness=report('4090-v44-us-negative-window-witness-v1.json')
require(witness['weights_sha256']==export['weights_sha256'] and witness['prior_report_sha256']==inputs['4090-v44-us-training-support-v1.json'] and witness['center_report_sha256']==inputs['4090-v44-us-center-fit-v1.json'] and witness['pack_report_sha256']==inputs['4090-us-training-features-v1.json'],'window witness input identities')
require([d['index'] for d in witness['details']]==[d['index'] for d in failures],'window witness complete selected identities')
window_summary=[]
for d in witness['details']:
 r=pack['records'][d['index']];c=center['details'][d['index']]
 require(all(d[k]==r[k] for k in ['source','voice','target','variant','raw_sha256']),'window source identity')
 ends=list(range(150,r['raw_shape'][0]+1));complete=[end for end in ends if (end-150)*160<=r['lead_samples'] and (end+1)*160>=r['lead_samples']+r['signal_samples']]
 require(d['legal_windows']==len(ends) and d['complete_ends']==[complete[0],complete[-1]] and d['complete_count']==len(complete) and d['trained_end']==r['feature_crop_end']==complete[0],'window endpoint ledger')
 require(d['prior_trained_window_float']==c['score'] and d['trained_window_float']-d['prior_trained_window_float']==d['trained_score_batch_difference'],'trained window batch difference accounting')
 for t,s in d['thresholds'].items():
  high=s['high_ends'];full=s['complete_high_ends'];require(len(set(high))==len(high)==s['high_count'] and high==sorted(high) and set(high)<=set(ends) and full==[end for end in high if end in complete] and len(full)==s['complete_high_count'] and len(high)-len(full)==s['noncomplete_high_count'],'window high endpoint accounting')
 window_summary.append({k:d[k] for k in ['index','legal_windows','complete_count','trained_window_float','trained_score_batch_difference','window_float_max']})
out['window_witness_ledger_checked']=window_summary
out['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
print(json.dumps(out,indent=2))
