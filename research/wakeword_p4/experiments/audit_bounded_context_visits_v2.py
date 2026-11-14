"""Independent saved-ledger/metadata audit; no sampler, features or models."""
import collections
import hashlib
import json
from pathlib import Path

R=Path('research/wakeword_p4/results')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def run():
    resultp=R/'4090-bounded-context-visits-result-v2.json'
    result=json.loads(resultp.read_text());planp=R/'4090-bounded-context-visits-plan-v2.json'
    assert sha(planp)==result['plan_sha256'];plan=json.loads(planp.read_text())
    for name,h in plan['files'].items():assert sha(Path(name))==h
    summaries=[]
    for report in result['targets']:
        target=report['target'];lp=Path(report['ledger_path']);assert sha(lp)==report['ledger_sha256']
        rows=json.loads((Path('.local/wakeword/hey-broad-training-features-v1')/target/'features.json').read_text())
        byoffset={r['offset']:r for r in rows};assert len(byoffset)==len(rows)
        traces=[]
        for condition in ('raw_logit_control','bounded_logit_consistency'):
            path=Path('.local/wakeword/hey-broad-bounded-consistency-formal-v1')/f'{target}-{condition}-0'/'actual-batch-schedule.jsonl'
            traces.append([json.loads(s) for s in path.read_text().splitlines()])
        counts=collections.Counter();roles=collections.Counter();seen=collections.defaultdict(set)
        with lp.open() as f:
            for step,line in enumerate(f,1):
                batch=json.loads(line);assert batch['step']==step
                assert all(t[step-1]['batch_sha256']==batch['batch_sha256'] for t in traces)
                assert len(batch['draws'])==96
                for index,d in enumerate(batch['draws']):
                    role='positive' if index<32 else 'discarded_negative' if index<48 else 'ordinary_negative' if index<80 else 'prefix_negative'
                    assert d['role']==role
                    r=byoffset[d['offset']];assert r['split']=='training'
                    for key in ('source_id','variant','engine','label','keyword_frame'):assert d[key]==r[key]
                    assert d['end']-d['start']==225 and 0<=d['start']<d['end']<=r['frames']
                    k=r['keyword_frame']
                    if role=='positive':assert 3<=d['end']-k<=27 and d['start']<=r['lead_frame']
                    elif role=='prefix_negative':assert r['engine'] in ('piper','kokoro') and 15<=k-d['end']<=35
                    else:assert d['end']>=max(225,r['lead_frame']-25)
                    coordinate='end_minus_keyword' if k is not None else 'keyword_unavailable'
                    counts[(role,r['engine'],coordinate,d['end']-k if k is not None else 0)]+=1
                    roles[role]+=1;seen[role].add(r['offset'])
        assert step==10000
        expected=collections.Counter({(d['role'],d['engine'],d['coordinate'],d['value']):d['count'] for d in report['counts']})
        assert counts==expected
        assert dict(roles)==dict(positive=320000,discarded_negative=160000,ordinary_negative=320000,prefix_negative=160000)
        summaries.append(dict(target=target,batches=step,draws_by_role=dict(roles),unique_rendered_rows_by_role={k:len(v) for k,v in seen.items()},saved_metadata_and_counts_exact=True))
    out=dict(result_sha256=sha(resultp),auditor_sha256=sha(Path(__file__)),targets=summaries,
             limits='Independent saved-ledger metadata, roles, legal ends, batch digest links and aggregate counts only. Actual tensor recomputation is producer evidence; no independent sampler replay or proof of historical source visits. No model or heldout inference, selection or promotion.')
    with (R/'4090-bounded-context-visits-independent-audit-v2.json').open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))
if __name__=='__main__':run()
