"""No-TF saved-array audit and label-stratified descriptive reductions."""
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
def sigmoid(z):
    z=np.asarray(z,dtype=float);a=np.exp(-np.abs(z))
    return np.where(z>=0,1/(1+a),a/(1+a))
def describe(v):
    return dict(count=int(v.size),mean=float(v.mean()),p99=float(np.percentile(v,99)),maximum=float(v.max()))
def run():
    rp=R/'4090-bounded-training-support-forward-result-v1.json';result=json.loads(rp.read_text())
    pp=R/'4090-bounded-training-support-forward-plan-v1.json';assert sha(pp)==result['plan_sha256']
    plan=json.loads(pp.read_text())
    for p,h in plan['files'].items():assert sha(p)==h
    assert len(result['cases'])==4
    records=[]
    for c in result['cases']:
        pc,=[v for v in plan['cases'] if (v['target'],v['condition'])==(c['target'],c['condition'])]
        ip=c['inputs'];assert sha(ip['path'])==ip['sha256'];assert ip['batch_steps']==plan['batch_steps']
        with np.load(ip['path'],allow_pickle=False) as z:x=z['x'];y=z['y'];w=z['weights']
        assert x.shape==(1280,225,40) and x.dtype==np.float32
        assert y.shape==(1280,1) and w.shape==(1280,)
        assert np.isfinite(x).all() and 0<=x.min()<=x.max()<=26
        trace=[json.loads(s) for s in Path(pc['trace']).read_text().splitlines()]
        for i,step in enumerate(plan['batch_steps']):
            sl=slice(128*i,128*(i+1));h=hashlib.sha256()
            for a in (x[sl],y[sl],w[sl]):h.update(a.tobytes())
            assert h.hexdigest()==ip['batch_digests'][i]==trace[step-1]['batch_sha256']
            assert np.count_nonzero(y[sl]==1)==32 and np.count_nonzero(y[sl]==0)==96
        assert np.array_equal(w,np.where(y[:,0]==1,1.,20.))
        assert sha(c['arrays_path'])==c['arrays_sha256']
        with np.load(c['arrays_path'],allow_pickle=False) as z:
            assert len(z.files)==16
            modes={}
            for mode in ('nominal','phase'):
                arrays=[]
                for name in ('float_logit','raw_qat_logit','qat_probability','exact_fc'):
                    a=z[f'{mode}_r0_{name}'];b=z[f'{mode}_r1_{name}']
                    assert a.shape==(1280,1) and np.isfinite(a).all() and np.array_equal(a,b)
                    arrays.append(a)
                zf,zq,prob,fc=arrays;assert 0<=prob.min()<=prob.max()<=1 and 0<=fc.min()<=fc.max()<=6
                bf=np.clip(zf,-8,8);bq=np.clip((fc.astype(np.float32)-3)*np.float32(8/3),-8,8)
                gap=np.abs(bf.astype(float)-bq.astype(float))
                expected=dict(examples=1280,bounded_logit_abs_mean=float(gap.mean()),bounded_logit_abs_max=float(gap.max()),same_side_saturated=int(np.sum(((zf>=8)&(zq>=8))|((zf<=-8)&(zq<=-8)))),saved_repeats_exact=True)
                assert expected==c['metrics'][mode]
                groups={}
                for label in (0,1):
                    mask=y[:,0]==label
                    groups[str(label)]=dict(examples=int(mask.sum()),bounded_logit_abs=describe(gap[mask]),nominal_sigmoid_abs=describe(np.abs(sigmoid(bf[mask])-sigmoid(bq[mask]))),float_probability_mean=float(sigmoid(zf[mask]).mean()),exact_fc_nominal_probability_mean=float(sigmoid(bq[mask]).mean()),qat_surrogate_probability_mean=float(prob[mask].mean()),bounded_consistency_zero=int(np.sum(gap[mask]==0)))
                modes[mode]=groups
        records.append(dict(target=c['target'],condition=c['condition'],modes=modes,input_batches_authenticated=True,saved_repeats_exact=True,producer_metrics_exact=True))
    output=dict(result_sha256=sha(rp),auditor_sha256=sha(__file__),cases=records,
                limits='Independent saved inputs/full selected batch SHA, labels/weights, repeats/finite/ranges and all producer reductions. Intermediate skipped replay batches, live model unchanged and inference provenance remain producer evidence. exactFC nominal sigmoid is NOT deployed post-head UINT8; QAT probability is surrogate. Static1280examples per target, not wholewave/five-step/recognition/FAR or independent acceptance. No threshold selection, model invoke, gradient or promotion.')
    with (R/'4090-bounded-training-support-forward-independent-audit-v1.json').open('x') as f:f.write(json.dumps(output,indent=2)+'\n')
    for c in records:
        print(c['target'],c['condition'],json.dumps({m:{l:v['nominal_sigmoid_abs'] for l,v in groups.items()} for m,groups in c['modes'].items()}))
if __name__=='__main__':run()
