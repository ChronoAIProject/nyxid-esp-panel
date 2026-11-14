"""All fixed-grid old regression points, with complete saved event vectors; never selects."""
import json,hashlib
from pathlib import Path
import numpy as np
from old_regression_events_v1 import events
from hey_development_metrics_v1 import GRID,far_upper
R=Path('research/wakeword_p4/results');P=R/'4090-old-fixed-grid-regression-plan-v2.json';OUT=Path('.local/wakeword/old-fixed-grid-regression-metrics-v2')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def offsets_stats(values):
 return dict(count=len(values),minimum=float(min(values)),median=float(np.median(values)),p95=float(np.quantile(values,.95)),maximum=float(max(values))) if values else dict(count=0,minimum=None,median=None,p95=None,maximum=None)
def main():
 lock=json.loads(P.read_text());collection=R/'4090-old-fixed-grid-regression-collection-v2.json';result=json.loads(collection.read_text());assert result['plan_sha256']==sha(P)
 for p,h in lock['files'].items():assert sha(p)==h,p
 materials={phase:json.loads(Path(lock[phase+'_manifest']).read_text())['records'] for phase in ['synthetic','background']};rows=result['records'];assert len(rows)==1860;seen=set();OUT.mkdir(exist_ok=False);reports=[]
 for model in lock['models']:
  loaded={};folder=OUT/model['id'];folder.mkdir()
  for phase in ['synthetic','background']:
   loaded[phase]=[]
   for index,source in enumerate(materials[phase]):
    c,=[r for r in rows if (r['model'],r['phase'],r['index'])==(model['id'],phase,index)];key=(model['id'],phase,index);assert key not in seen;seen.add(key);assert c['samples']==source['samples'] and c['source_id']==source['source_id'] and c['repeat_exact'];assert sha(c['arrays_path'])==c['arrays_sha256']
    with np.load(c['arrays_path'],allow_pickle=False) as a:
     assert a['uint8'].dtype==np.uint8 and a['floating'].dtype==np.float32 and np.array_equal(a['uint8'],a['uint8_repeat']) and np.array_equal(a['floating'],a['float_repeat']) and np.isfinite(a['floating']).all()
     assert np.array_equal(a['sample_end'],np.minimum((3*np.arange(1,len(a['uint8'])+1,dtype=np.int64)+2)*160,source['samples']))
     loaded[phase].append((source,a['uint8'].copy(),a['floating'].copy()))
  for kind in ['uint8']+(['float'] if model['weights'] is not None else []):
   for order,threshold in enumerate(GRID):
    vectors={};positive=[];negative={'common':dict(clips=0,activated_clips=0,events=0),'cross_target':dict(clips=0,activated_clips=0,events=0)};early=0;latency=[];rir_latency=[]
    for phase in ['synthetic','background']:
     flat=[];offsets=[0];background_events=0
     for source,integer,floating in loaded[phase]:
      raw=integer if kind=='uint8' else floating;steps=events(raw,threshold,kind);flat.extend(steps);offsets.append(len(flat));times=[min((3*s+2)*160,source['samples']) for s in steps]
      if phase=='background':background_events+=len(steps);continue
      if source['target_id']==model['target']:
       complete=[t for t in times if t>=source['generated_prefix_end_sample']];rir=[t for t in times if t>=source['full_rir_prefix_end_sample']];positive.append((source,bool(complete),bool(rir)));early+=sum(t<source['lead_samples'] for t in times)
       if complete:latency.append((complete[0]-source['generated_prefix_end_sample'])/16)
       if rir:rir_latency.append((rir[0]-source['full_rir_prefix_end_sample'])/16)
      else:
       other='hey_nyx' if model['target']=='hey_willow' else 'hey_willow';category='cross_target' if source['target_id']==other else 'common';negative[category]['clips']+=1;negative[category]['activated_clips']+=int(bool(steps));negative[category]['events']+=len(steps)
     vectors[phase+'_offsets']=np.asarray(offsets,np.int64);vectors[phase+'_steps']=np.asarray(flat,np.int64)
    assert len(positive)==36 and negative['common']['clips']==288 and negative['cross_target']['clips']==36
    rate=lambda group:sum(int(r[1]) for r in group)/len(group)
    conditions={str(v):rate([r for r in positive if r[0]['variant']==v]) for v in range(3)};hours=sum(s['samples'] for s in materials['background'])/16000/3600
    path=folder/f'{kind}-{order:03}-events.npz';np.savez_compressed(path,**vectors)
    reports.append(dict(model=model['id'],target=model['target'],kind=kind,threshold=threshold,synthetic=dict(positive_clips=36,complete_phrase_activated=sum(p[1] for p in positive),macro=float(np.mean(list(conditions.values()))),conditions=conditions,standalone=rate([r for r in positive if not r[0]['carrier']]),command=rate([r for r in positive if r[0]['carrier']]),full_rir_prefix_recall=sum(p[2] for p in positive)/36,pre_source_positive_events=early,negatives=negative,first_complete_hit_generated_reference_ms=offsets_stats(latency),first_rir_complete_hit_reference_ms=offsets_stats(rir_latency)),background=dict(hours=hours,events=background_events,far_per_hour=background_events/hours,upper95_per_hour=far_upper(background_events,hours)),events_path=str(path),events_sha256=sha(path),frozen_v30_reference_point=bool(model['id']=='v30' and threshold==.91796875)))
   print(model['id'],kind,'all138points',flush=True)
 assert len(seen)==1860 and len(reports)==9*138
 out=R/'4090-old-fixed-grid-regression-metrics-v2.json'
 with out.open('x') as f:json.dump(dict(plan_sha256=sha(P),collection_sha256=sha(collection),points=reports,selection_authorized=False,limits='Old consumed regression only. Every fixed138 point retained with complete event offsets/steps for all360synthetic+12background. No eligible/choose/selected fields or promotion. Generated-reference latency only, no human phonetic truth or fresh independent evidence.'),f,indent=2);f.write('\n')
if __name__=='__main__':main()
