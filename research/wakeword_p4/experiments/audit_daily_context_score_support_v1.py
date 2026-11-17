"""Describe saved British bare score support, never search operating thresholds."""
import json
from pathlib import Path
import numpy as np
from score_context_daily_context_coverage_v1 import rows,sha,R
from uint8_event_policy_v1 import event_steps

def main():
    ledgers=rows('melo');records=[];files={str(Path(__file__)):sha(__file__)}
    for i,base in enumerate(ledgers['c161']):
        s=base['record']
        if base['phase']=='background' or not s['voice_id'].endswith('EN-BR') or s.get('phrase_id')!='nyx' or s.get('carrier')!='bare':continue
        item=dict(index=base['index'],source=s,models={})
        for name,ledger in ledgers.items():
            row=ledger[i];assert row['record']==s
            p=Path(row['trace_path']);assert sha(p)==row['trace_sha256'];files[str(p)]=sha(p)
            with np.load(p) as z:
                codes=z['uint8'];assert codes.dtype==np.uint8
                clock=z['block_return_sample']-81600
                sums=np.convolve(codes.astype(np.int64),np.ones(5,dtype=np.int64),'valid')
                real=clock[4:]>0;eligible=np.flatnonzero(real);j=int(eligible[np.argmax(sums[eligible])]);events=event_steps(codes,.91796875)
                crossed=[dict(step=int(k+5),relative_sample=int(clock[k+4])) for k in eligible if sums[k]>=1175]
                item['models'][name]=dict(max_real_raw=int(codes[clock>0].max()),max_real_sum5=int(sums[j]),fixed_required_sum5=1175,peak_relative_sample=int(clock[j+4]),real_events=[dict(step=k,relative_sample=int(clock[k-1])) for k in events if clock[k-1]>0],crossing_windows=len(crossed),crossing_but_no_real_event=bool(crossed) and not any(clock[k-1]>0 for k in events))
        records.append(item)
    assert len(records)==9
    report=dict(schema='daily-context-british-bare-score-support-v1',files=files,records=records,policy='All9 consumed British bare targets, saved raw bytes only. Fixed threshold .91796875/5/67, sum1175/1280. Postprime includes original leading/tail silence; peak is descriptive, not phoneme attribution. No inference, threshold search/rescue, training admission or acceptance. Historical source role strings retained verbatim; these inputs are now consumed.')
    with (R/'4090-daily-context-british-bare-score-support-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    for r in records:print(r['index'],r['source']['speed'],r['source']['variant'],{n:(m['max_real_sum5'],len(m['real_events']),m['crossing_but_no_real_event']) for n,m in r['models'].items()})
if __name__=='__main__':main()
