"""Independent no-TF saved gradient reductions; no autodiff correctness claim."""
import hashlib
import json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def norm(v):return float(np.sqrt(np.dot(v,v)))
def cosine(a,b):
    den=norm(a)*norm(b)
    return float(np.dot(a,b)/den) if den else None
def run():
    rp=R/'4090-bounded-support-gradient-result-v1.json';result=json.loads(rp.read_text());pp=R/'4090-bounded-support-gradient-plan-v1.json'
    assert sha(pp)==result['plan_sha256'];plan=json.loads(pp.read_text())
    for p,h in plan['files'].items():assert sha(p)==h
    assert len(result['cases'])==4;reports=[]
    for c in result['cases']:
        pc,=[v for v in plan['cases'] if (v['target'],v['condition'])==(c['target'],c['condition'])]
        assert len(c['records'])==40 and len(c['trainable_inventory'])==15
        with np.load(pc['inputs']['path']) as z:labels=z['y'];weights=z['weights']
        rows=[]
        with np.load(pc['prior_arrays']) as prior:
            for mode in ('nominal','phase'):
                for batch,step in enumerate(plan['batch_steps']):
                    records=[v for v in c['records'] if (v['mode'],v['step'])==(mode,step)]
                    assert len(records)==2 and {v['repeat'] for v in records}=={0,1}
                    saved=[]
                    for record in sorted(records,key=lambda v:v['repeat']):
                        assert sha(record['path'])==record['sha256']
                        with np.load(record['path'],allow_pickle=False) as z:a={k:z[k] for k in z.files}
                        assert len(a)==75 and all(np.isfinite(v).all() for v in a.values());saved.append(a)
                    a,b=saved;assert a.keys()==b.keys() and all(np.array_equal(a[k],b[k]) for k in a)
                    sl=slice(batch*128,(batch+1)*128)
                    for name in ('float_logit','raw_qat_logit','qat_probability','exact_fc'):assert np.array_equal(a[name],prior[f'{mode}_r0_{name}'][sl])
                    vectors={}
                    for name in ('S','C','F','Q'):
                        assert a[name+'_none'].shape==(15,) and a[name+'_none'].dtype==bool
                        for i,inv in enumerate(c['trainable_inventory']):
                            v=a[f'{name}_{i:02}'];assert list(v.shape)==inv['shape'] and str(v.dtype)==inv['dtype']
                            if a[name+'_none'][i]:assert not np.any(v)
                        vectors[name]=np.concatenate([a[f'{name}_{i:02}'].astype(float).ravel() for i in range(15)])
                    f=a['float_logit'];q=a['raw_qat_logit'];fc=a['exact_fc'];bf=np.clip(f,-8,8);preq=(fc.astype(np.float32)-3)*np.float32(8/3);bq=np.clip(preq,-8,8)
                    gap=(bf.astype(float)-bq.astype(float))/8;h=np.clip(gap,-1,1)/8/128
                    sf=h*((f>=-8)&(f<=8));sq=-h*np.float32(8/3)*((preq>=-8)&(preq<=8))
                    cfg=c['configuration']['operators'][-1];scale=cfg['output']['scales'][0];zero=cfg['output']['zeros'][0]
                    multiplier=cfg['input']['scales'][0]*cfg['weight']['scales'][0]/scale
                    acc=a['accumulator'];assert np.array_equal(acc,np.rint(acc))
                    real=acc*multiplier+zero;low=max(-128,zero);high=min(127,int(np.rint(6/scale))+zero);active=(real>=low)&(real<=high)
                    sa=a['seed_fc']*multiplier*scale*active
                    y=labels[sl];w=weights[sl]
                    bce=lambda z:np.maximum(z,0)-z*y+np.log1p(np.exp(-np.abs(z)))
                    supervised=np.mean(.5*(bce(f.astype(float))+bce(q.astype(float)))*w[:,None])
                    consistency=np.mean(np.where(np.abs(gap)<=1,.5*gap*gap,np.abs(gap)-.5))
                    residual=vectors['C']-vectors['F']-vectors['Q']
                    rows.append(dict(mode=mode,step=step,supervised=float(a['supervised']),consistency=float(a['consistency']),norms={n:norm(v) for n,v in vectors.items()},supervised_consistency_cosine=cosine(vectors['S'],vectors['C']),float_quant_consistency_cosine=cosine(vectors['F'],vectors['Q']),merged_SC_norm=norm(vectors['S']+vectors['C']),C_minus_FQ_maxabs=float(np.max(np.abs(residual))),none_indices={n:np.flatnonzero(a[n+'_none']).tolist() for n in vectors},seed_formula_maxabs=dict(float=float(np.max(np.abs(sf-a['seed_float']))),fc=float(np.max(np.abs(sq-a['seed_fc']))),accumulator=float(np.max(np.abs(sa-a['seed_accumulator'])))),loss_formula_abs=dict(supervised=abs(supervised-float(a['supervised'])),consistency=abs(consistency-float(a['consistency']))),nonzero_fc_seed=int(np.count_nonzero(a['seed_fc'])),nonzero_accumulator_seed=int(np.count_nonzero(a['seed_accumulator'])),nonzero_fc_seed_blocked_at_scalar_fc=int(np.sum((a['seed_fc']!=0)&~active))))
        reports.append(dict(target=c['target'],condition=c['condition'],rows=rows,all_saved_repeats_and_prior_forwards_exact=True))
    output=dict(result_sha256=sha(rp),auditor_sha256=sha(__file__),cases=reports,
                limits='All saved file SHA/repeats/prior forward/gradient shapes and None arrays independently checked. Float64 seed/loss formula differences and branch-sum residuals descriptive, not new tolerance gates. Saved gradient norms/cosines and original scalarFC activity computed without TF. Autodiff correctness, encoder masks and model-unchanged remain producer evidence. No inference/updates/selection or promotion.')
    with (R/'4090-bounded-support-gradient-independent-audit-v1.json').open('x') as f:f.write(json.dumps(output,indent=2)+'\n')
    for c in reports:
        for mode in ('nominal','phase'):
            rows=[v for v in c['rows'] if v['mode']==mode]
            print(c['target'],c['condition'],mode,'Czero',sum(v['norms']['C']==0 for v in rows),'SCconflict',sum(v['supervised_consistency_cosine'] is not None and v['supervised_consistency_cosine']<0 for v in rows),'FCblocked',sum(v['nonzero_fc_seed_blocked_at_scalar_fc'] for v in rows),'branchsummax',max(v['C_minus_FQ_maxabs'] for v in rows))
if __name__=='__main__':run()
