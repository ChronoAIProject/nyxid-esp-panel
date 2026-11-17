"""All23 saved fit traces, exact scores and original end-of-wave support."""
import hashlib,json
from pathlib import Path
import numpy as np
import soundfile as sf
from uint8_event_policy_v1 import event_steps
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    source=Path('.local/wakeword/bare-support-training-bare23-replay-v1/result.json');admission=R/'4090-chatterbox-bare-training-admission-v1.json'
    rows=json.loads(source.read_text())['records'];ad={r['index']:r for r in json.loads(admission.read_text())['admitted']};assert len(rows)==len(ad)==23
    records=[];files={str(p):sha(p) for p in [Path(__file__),source,admission]}
    for r in rows:
        a=ad[r['index']];trace=Path(r['trace']['path']);assert sha(trace)==r['trace']['sha256'];files[str(trace)]=sha(trace)
        wav=Path(a['wav']['path']);assert sha(wav)==a['wav']['sha256'];samples=sf.info(wav).frames;files[str(wav)]=sha(wav)
        with np.load(trace) as z:
            codes=z['uint8'];assert codes.dtype==np.uint8
            sums=np.convolve(codes.astype(np.int64),np.ones(5,dtype=np.int64),'valid');clock=z['relative_block_return'];eligible=np.flatnonzero(clock[4:]>0);j=int(eligible[np.argmax(sums[eligible])]);events=event_steps(codes,.91796875)
            assert [k for k in events if clock[k-1]>0]==[e['step'] for e in r['real_events']]
            endpoint=round(a['automatic_prefix_end_seconds']*16000)
            records.append(dict(index=r['index'],voice=r['request']['voice'],request=r['request'],original_samples=samples,endpoint_sample=endpoint,original_tail_after_endpoint_samples=samples-endpoint,supervision_window_end_sample=endpoint+4320,window_end_past_original_samples=max(0,endpoint+4320-samples),hit=bool(r['real_events']),max_raw=int(codes[clock>0].max()),max_real_sum5=int(sums[j]),fixed_required_sum5=1175,peak_relative_sample=int(clock[j+4]),peak_at_last_return=bool(j+4==len(clock)-1),final5_sum=int(codes[-5:].astype(int).sum()),trace_sha256=sha(trace)))
    report=dict(schema='bare-support-training-fit-score-support-v1',files=files,records=records,policy='All23consumed provisionaltrainingpositives, no new inference or threshold search. Original no-tail-extension isolated replay retained; max score wholepostprime, no phoneme attribution. Trainingcache retains8000sample tail unlike this isolated witness. Missing future model outputs cannot be inferred from final score. Near-boundary/end-of-file is diagnostic, not a retrospective rescued hit, model qualification or evidence that appended silence will fix it.')
    with (R/'4090-bare-support-training-fit-score-support-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    for r in records:
        if not r['hit']:print(json.dumps(r))
if __name__=='__main__':main()
