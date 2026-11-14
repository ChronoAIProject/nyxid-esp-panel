"""Describe saved branch opposition and one-way local gradient algebra."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
    ap=R/'4090-bounded-support-gradient-independent-audit-v1.json';rp=R/'4090-bounded-support-gradient-result-v1.json'
    audit=json.loads(ap.read_text());assert audit['result_sha256']==sha(rp);original=json.loads(rp.read_text());cases=[]
    for c in original['cases']:
        rows=[]
        for record in c['records']:
            if record['repeat']!=0:continue
            assert sha(record['path'])==record['sha256']
            with np.load(record['path']) as z:
                g={n:np.concatenate([z[f'{n}_{i:02}'].astype(float).ravel() for i in range(15)]) for n in ('S','C','F','Q')}
            norm=lambda a:float(np.linalg.norm(a))
            cosine=lambda a,b:float(np.dot(a,b)/(norm(a)*norm(b))) if norm(a)*norm(b) else None
            den=norm(g['F'])+norm(g['Q'])
            rows.append(dict(mode=record['mode'],step=record['step'],FQ_cosine=cosine(g['F'],g['Q']),C_norm=norm(g['C']),Q_norm=norm(g['Q']),C_over_sum_branch_norms=norm(g['C'])/den if den else None,SQ_cosine=cosine(g['S'],g['Q']),original_SC_norm=norm(g['S']+g['C']),oneway_SQ_norm=norm(g['S']+g['Q'])))
        summaries=[]
        for mode in ('nominal','phase'):
            subset=[r for r in rows if r['mode']==mode];valid=[r for r in subset if r['FQ_cosine'] is not None]
            summaries.append(dict(mode=mode,batches=len(subset),both_branch_nonzero=len(valid),FQ_negative_cosine=sum(r['FQ_cosine']<0 for r in valid),SQ_negative_cosine=sum(r['SQ_cosine'] is not None and r['SQ_cosine']<0 for r in subset)))
        cases.append(dict(target=c['target'],condition=c['condition'],rows=rows,summary=summaries))
    output=dict(audit_sha256=sha(ap),result_sha256=sha(rp),reducer_sha256=sha(__file__),cases=cases,
                limits='Algebra of saved final-snapshot gradients, not executed stop-gradient graphs, optimizer updates or historical dynamics. Opposing branch directions are partly expected for a difference loss and do not prove harmful cancellation. Larger gradient norm is not evidence of better training. No example/threshold/lambda selection or promotion; motivates a prospectively controlled one-way intervention only.')
    with (R/'4090-bounded-directional-gradient-reduction-v1.json').open('x') as f:f.write(json.dumps(output,indent=2)+'\n')
    for c in cases:print(c['target'],c['condition'],c['summary'])
if __name__=='__main__':run()
