"""Independently audit saved training diagnostic raw archive and integer peaks."""
import json,hashlib
from pathlib import Path
import numpy as np
from diagnose_hey_training_stream_support_v1 import OUT,PLAN,sha

def main():
    plan=json.loads(PLAN.read_text());report=json.loads((OUT/'report.json').read_text());assert report['plan_sha256']==sha(PLAN)
    arms={}
    for target in ('hey_willow','hey_nyx'):
        root=OUT/target;s=report['summaries'][target]
        assert sha(root/'raw-uint8.bin')==s['raw_archive_sha256'] and sha(root/'records.jsonl')==s['records_sha256']
        rows=[json.loads(line) for line in (root/'records.jsonl').read_text().splitlines()]
        expected=json.loads((Path(plan['arms'][target]['features'])/'features.json').read_text())
        assert len(rows)==len(expected)==7200 and [r['record'] for r in rows]==expected
        offset=0
        with (root/'raw-uint8.bin').open('rb') as f:
            for i,r in enumerate(rows):
                assert r['index']==i and r['trace_offset']==offset and r['trace_bytes']==r['record']['frames']//3
                data=f.read(r['trace_bytes']);assert hashlib.sha256(data).hexdigest()==r['trace_sha256']
                values=np.frombuffer(data,np.uint8);prefix=np.cumsum(np.r_[np.int64(0),values.astype(np.int64)])
                sums=prefix[5:]-prefix[:-5]
                assert int(values.max())==r['raw_peak'] and int(np.argmax(values))+1==r['raw_peak_first_step']
                assert int(sums.max())==r['five_sum_peak'] and int(np.argmax(sums))+5==r['five_peak_first_step']
                assert float(sums.max()/1280)==r['five_mean_peak']
                for name,step in [('raw',r['raw_peak_first_step']),('five',r['five_peak_first_step'])]:
                    assert r[name+'_peak_first_support_sample']==min((3*step+2)*160,r['record']['samples'])
                offset+=len(data)
            assert f.read(1)==b''
        assert sum(r['record']['label']==1 for r in rows)==720
        arms[target]=dict(rows=7200,positive_rows=720,negative_rows=6480,raw_bytes=offset,all_peak_and_trace_records_verified=True)
    result=dict(schema='hey-training-stream-diagnostic-audit-v1',auditor_sha256=sha(__file__),report_sha256=sha(OUT/'report.json'),arms=arms,inference_run=False,limits='Independent saved raw-byte ledger and cumsum integer peak check only. Not independent recognition/FAR or isolated cause.')
    dest=Path('research/wakeword_p4/results/4090-hey-training-stream-diagnostic-audit-v1.json')
    with dest.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
