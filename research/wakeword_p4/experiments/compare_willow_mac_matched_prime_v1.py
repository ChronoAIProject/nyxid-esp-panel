"""Compare saved own-target events on the identical12 consumed source WAVs."""
import hashlib,json
from pathlib import Path
import numpy as np
from uint8_event_policy_v1 import event_steps
R=Path('research/wakeword_p4/results');W=Path('.local/wakeword/willow-mac-matched-prime-v1');C=Path('.local/wakeword/c161-mac12-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    wp=R/'4090-willow-mac-matched-prime-plan-v1.json';cp=R/'4090-c161-mac12-plan-v1.json'
    plan=json.loads(wp.read_text());c=json.loads((C/'result.json').read_text());assert c['plan_sha256']==sha(cp)
    assert c['model_sha256']=='c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81' and c['threshold']==.91796875
    assert len(plan['records'])==len(c['records'])==12
    rows=[];groups={}
    for i,(source,old) in enumerate(zip(plan['records'],c['records'],strict=True)):
        report=W/f'{i:03d}/report.json';w=json.loads(report.read_text())
        assert w['execution_sha256']==sha(wp) and w['record']==source and old['index']==i
        assert w['source_pcm_sha256']==source['pcm_sha256']==old['source']['pcm_sha256']
        assert w['total_prime_samples']==81600 and not w['transition_reset'] and w['actual_float32_bits']=='3f24dd2f'
        trace=C/f'{i:03d}.npz';assert sha(trace)==old['trace_sha256']
        with np.load(trace) as z:
            events=[dict(step=s,relative_sample=int(z['relative_block_return'][s-1]),finish=bool(z['support'][s-1]>source['samples']+81600)) for s in event_steps(z['uint8'],.91796875)]
        assert events==old['events'] and [e for e in events if e['relative_sample']>0]==old['real_events']
        row=dict(index=i,voice=source['voice'],target=source['target'],form=source['form'],source_pcm_sha256=source['pcm_sha256'],willow_events=w['real_events'],c161_events=old['real_events'],willow_report_sha256=sha(report),c161_trace_sha256=sha(trace));rows.append(row)
        for detector,target,activated in [('official-willow','hey_willow',bool(w['real_events'])),('c161','hey_nyx',bool(old['real_events']))]:
            key=detector+'/'+source['voice']+'/'+('own-target' if source['target']==target else 'other-target')+'/'+source['form']
            g=groups.setdefault(key,dict(clips=0,hits=0));g['clips']+=1;g['hits']+=activated
    result=dict(actual_official_replay_exit_code=0,official_exec_session=97113,willow_plan_sha256=sha(wp),c161_result_sha256=sha(C/'result.json'),groups=groups,records=rows,limits='Same consumed Mac12 source bytes and total81600continuouszero prime, distinct target phrases and native event functions/512vs480 clocks/default.644vsC161.91796875+5. Small positivecontrol, no matched score operatingpoint, general superiority, FAR exposure, human or P4 claim. Original reports unchanged.')
    with (R/'4090-willow-mac-matched-prime-c161-comparison-v1.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(groups,indent=2))
if __name__=='__main__':main()
