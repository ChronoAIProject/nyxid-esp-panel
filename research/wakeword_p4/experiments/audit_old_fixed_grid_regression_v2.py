"""Independent full-event-ledger proofs and metric reduction, no model inference."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from hey_development_metrics_v1 import GRID
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prove_events(mean,steps,threshold):
    steps=np.asarray(steps);assert steps.dtype==np.int64 and steps.ndim==1
    high=np.flatnonzero(mean>=threshold)+5;low=np.flatnonzero(mean<threshold/2)+5
    if not len(high):assert not len(steps);return
    assert len(steps) and int(steps[0])==int(high[0])
    # Prove each reported next event is the first crossing after legal rearm;
    # prove the tail cannot contain a further legal event. No producer helper.
    for j,previous in enumerate(steps):
        rearm=low[low>=int(previous)+67]
        if not len(rearm):assert j==len(steps)-1;return
        following=high[high>=rearm[0]]
        if not len(following):assert j==len(steps)-1;return
        assert j+1<len(steps) and int(steps[j+1])==int(following[0])
    raise AssertionError('Unreported next event')
def main():
 plan=R/'4090-old-fixed-grid-regression-plan-v2.json';collection=R/'4090-old-fixed-grid-regression-collection-v2.json';metrics=R/'4090-old-fixed-grid-regression-metrics-v2.json';lock=json.loads(plan.read_text());rawresult=json.loads(collection.read_text());result=json.loads(metrics.read_text());assert result['plan_sha256']==rawresult['plan_sha256']==sha(plan) and result['collection_sha256']==sha(collection) and result['selection_authorized'] is False
 for p,h in lock['files'].items():assert sha(p)==h,p
 materials={phase:json.loads(Path(lock[phase+'_manifest']).read_text())['records'] for phase in ['synthetic','background']};rawrows={(r['model'],r['phase'],r['index']):r for r in rawresult['records']};assert len(rawrows)==len(rawresult['records'])==1860;assert len(result['points'])==9*138
 summaries=[];points_seen=set()
 for model in lock['models']:
  for kind in ['uint8']+(['float'] if model['weights'] is not None else []):
   sources={};means={}
   for phase,rows in materials.items():
    sources[phase]=[];means[phase]=[]
    for index,row in enumerate(rows):
     receipt=rawrows[model['id'],phase,index];assert receipt['source_id']==row['source_id'] and receipt['samples']==row['samples'] and receipt['raw_frames']==max(0,(row['samples']+159)//160-2) and receipt['final_feature_padding']==(-receipt['raw_frames'])%3 and receipt['steps']==(receipt['raw_frames']+2)//3;assert sha(receipt['arrays_path'])==receipt['arrays_sha256']
     with np.load(receipt['arrays_path'],allow_pickle=False) as a:
      assert np.array_equal(a['uint8'],a['uint8_repeat']) and np.array_equal(a['floating'],a['float_repeat']);assert a['uint8'].dtype==np.uint8 and a['floating'].dtype==np.float32
      assert a['uint8'].shape==(receipt['steps'],) and a['floating'].shape==((receipt['steps'],) if model['weights'] is not None else (0,))
      assert np.array_equal(a['sample_end'],np.minimum((3*np.arange(1,receipt['steps']+1,dtype=np.int64)+2)*160,row['samples']))
      x=a['uint8'].astype(np.int64) if kind=='uint8' else a['floating'].astype(np.float64);assert np.isfinite(x).all();windows=np.lib.stride_tricks.sliding_window_view(x,5);mean=windows.sum(axis=1)/(1280 if kind=='uint8' else 5)
     sources[phase].append(row);means[phase].append(mean)
   selected=[p for p in result['points'] if p['model']==model['id'] and p['kind']==kind];assert len(selected)==138;assert {p['threshold'] for p in selected}==set(GRID);assert all(p['target']==model['target'] for p in selected)
   total_events=0
   for point in selected:
    key=(model['id'],kind,point['threshold']);assert key not in points_seen;points_seen.add(key);assert sha(point['events_path'])==point['events_sha256'];positive=[];negative={'common':dict(clips=0,activated_clips=0,events=0),'cross_target':dict(clips=0,activated_clips=0,events=0)};early=0;latency=[];rirlatency=[];background_events=0
    with np.load(point['events_path'],allow_pickle=False) as ledger:
     for phase in ['synthetic','background']:
      offsets=ledger[phase+'_offsets'];steps=ledger[phase+'_steps'];assert offsets.dtype==steps.dtype==np.int64 and offsets.shape==(len(sources[phase])+1,) and offsets[0]==0 and offsets[-1]==len(steps) and np.all(np.diff(offsets)>=0)
      for j,row in enumerate(sources[phase]):
       segment=steps[offsets[j]:offsets[j+1]];prove_events(means[phase][j],segment,point['threshold']);total_events+=len(segment)
       if phase=='background':background_events+=len(segment);continue
       times=np.minimum((3*segment+2)*160,row['samples'])
       if row['target_id']==model['target']:
        hit=times[times>=row['generated_prefix_end_sample']];rir=times[times>=row['full_rir_prefix_end_sample']];positive.append((row,bool(len(hit)),bool(len(rir))));early+=int(np.sum(times<row['lead_samples']))
        if len(hit):latency.append(float((hit[0]-row['generated_prefix_end_sample'])/16))
        if len(rir):rirlatency.append(float((rir[0]-row['full_rir_prefix_end_sample'])/16))
       else:
        other='hey_nyx' if model['target']=='hey_willow' else 'hey_willow';category='cross_target' if row['target_id']==other else 'common';negative[category]['clips']+=1;negative[category]['activated_clips']+=int(bool(len(segment)));negative[category]['events']+=len(segment)
    reported=point['synthetic'];assert len(positive)==36 and reported['positive_clips']==36 and reported['complete_phrase_activated']==sum(p[1] for p in positive) and reported['negatives']==negative and reported['pre_source_positive_events']==early
    rate=lambda group:sum(x[1] for x in group)/len(group)
    conditions={str(v):rate([r for r in positive if r[0]['variant']==v]) for v in range(3)};assert reported['conditions']==conditions and reported['macro']==float(np.mean(list(conditions.values()))) and reported['standalone']==rate([r for r in positive if not r[0]['carrier']]) and reported['command']==rate([r for r in positive if r[0]['carrier']]) and reported['full_rir_prefix_recall']==sum(r[2] for r in positive)/36
    for values,label in [(latency,'first_complete_hit_generated_reference_ms'),(rirlatency,'first_rir_complete_hit_reference_ms')]:
     observed=reported[label];assert observed['count']==len(values)
     if values:assert observed['minimum']==min(values) and observed['maximum']==max(values) and observed['median']==float(np.median(values)) and observed['p95']==float(np.quantile(values,.95))
    hours=sum(r['samples'] for r in sources['background'])/16000/3600;assert point['background']['hours']==hours and point['background']['events']==background_events and point['background']['far_per_hour']==background_events/hours and point['background']['upper95_per_hour']==float(chi2.ppf(.95,2*(background_events+1))/(2*hours))
   summaries.append(dict(model=model['id'],kind=kind,points=138,complete_event_ledgers_proved=True,all_metric_reductions_exact=True,total_events_across_fixed_points=total_events))
 assert len(points_seen)==1242 and sum(bool(p['frozen_v30_reference_point']) for p in result['points'])==1
 out=R/'4090-old-fixed-grid-regression-independent-audit-v2.json'
 with out.open('x') as f:json.dump(dict(metrics_sha256=sha(metrics),collection_sha256=sha(collection),summary=summaries,limits='Independent saved score/hash/repeat/shape/tail/clock, event completeness via first crossing and legal rearm proofs, all138 metrics/denominators/signedgenerated latency/FAR bounds. No independent detector/frontend inference, candidate admission or threshold selection; reuseddevelopment only.'),f,indent=2);f.write('\n')
 print('All1242 event ledgers and metrics independently audited')
if __name__=='__main__':main()
