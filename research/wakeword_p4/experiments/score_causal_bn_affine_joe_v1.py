"""Joe behavior with whole-generated-utterance timing only, fixed parameters."""
import json,sys
from pathlib import Path
import numpy as np
from collect_causal_bn_affine_joe_v1 import R,LOCK,OUT
from collect_multispeaker_priming_v1 import sha,PRIME
from uint8_event_policy_v1 import event_steps
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from evaluate import smooth,event_indices
def main():
    result=json.loads((OUT/'result.json').read_text());assert result['all_passed'] and result['execution_sha256']==sha(LOCK)
    dest=OUT/'diagnostic';dest.mkdir(exist_ok=False);points=[]
    for m in result['models']:
        report=json.loads(Path(m['path']).read_text());assert sha(m['path'])==m['sha256'] and len(report['records'])==360
        name=m['model'];details=[];groups={};negative={};pre=0
        for rec in report['records']:
            assert sha(rec['trace_path'])==rec['trace_sha256'];row=rec['record']
            with np.load(rec['trace_path'],allow_pickle=False) as a:raw=a['uint8'].copy()
            steps=([int(i)+5 for i in event_indices(smooth(raw.astype(np.float32)/256),.91796875)] if name.endswith('-v30') else event_steps(raw,.91796875))
            times=[min(((3*s+2)*160+479)//480*480,row['samples']+PRIME)-PRIME for s in steps];real=[t for t in times if t>0]
            item=dict(index=rec['index'],source_id=row['source_id'],text=row['text'],variant=row['variant'],draw=row['draw'],priming_event_samples=[t for t in times if t<=0],event_samples=real)
            if row['target_id']=='hey_nyx':
                after=[t for t in real if t>=row['lead_samples']];carrier='command' if ', ' in row['text'] else 'bare';pre+=sum(t<row['lead_samples'] for t in real)
                item.update(hit=bool(after),whole_generated_complete=any(t>=row['generated_output_end_sample'] for t in after),first_signed_container_ms=(after[0]-row['generated_output_end_sample'])/16 if after else None,carrier=carrier);groups.setdefault(carrier,[]).append(item)
            else:
                kind='cross_target' if row['target_id']=='hi_esp' else 'common';n=negative.setdefault(kind,dict(clips=0,activated_clips=0,events=0));n['clips']+=1;n['activated_clips']+=bool(real);n['events']+=len(real)
            details.append(item)
        positives=sum(groups.values(),[])
        def stats(a):
            lat=[x['first_signed_container_ms'] for x in a if x['hit']]
            return dict(clips=len(a),hits=sum(x['hit'] for x in a),whole_generated_complete=sum(x['whole_generated_complete'] for x in a),signed_container_median_ms=float(np.median(lat)) if lat else None)
        assert len(positives)==72 and sum(n['clips'] for n in negative.values())==288
        point=dict(model=name,threshold=.91796875,smoothing_steps=5,positive=dict(all=stats(positives),**{k:stats(v) for k,v in groups.items()}),negative=negative,pre_source_events=pre,priming_events_ignored_by_app=sum(len(x['priming_event_samples']) for x in details),event_policy='original-float64-v30' if name.endswith('-v30') else 'uint8-sum-binary64-v1')
        (dest/f'{name}.jsonl').write_text(json.dumps(dict(summary=point,records=details))+'\n');points.append(point);print(json.dumps(point))
    out=dict(schema='causal-bn-affine-joe-behavior-v1',execution_sha256=sha(LOCK),points=points,threshold_selected=False,acceptance_accessed=False,candidate_promoted=False,limits='Existing consumed-source Joe development only; shared Mike parent is not independent acceptance. Whole utterance end is generated container, not wake-prefix or human phonetic boundary; negative signed command times do not prove early wakephoneme detection. Continuous81600prime/native480clock, samefixedprofile all3models, frozenv30 originalevent arithmetic. No newTTS/train/thresholdsearch/backgroundFAR evidence.')
    (dest/'summary.json').write_text(json.dumps(out,indent=2)+'\n');(R/'4090-causal-bn-affine-joe-behavior-v1.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
