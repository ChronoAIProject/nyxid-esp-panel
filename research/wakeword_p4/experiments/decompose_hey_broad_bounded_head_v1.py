"""Saved-witness boundary decomposition; read constants only, never invoke/export."""
import argparse
import json
from pathlib import Path
import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
from run_hey_broad_trained_quantization_v1 import ROOT,published

PLAN=ROOT/'results/4090-hey-broad-bounded-head-decomposition-plan-v1.json'
RESULT=ROOT/'results/4090-hey-broad-bounded-head-decomposition-result-v1.json'
OUT=Path('.local/wakeword/hey-broad-bounded-head-decomposition-v1')

def plan():
    source=ROOT/'results/4090-hey-broad-bounded-tensors-result-v1.json'
    parent=ROOT/'results/4090-hey-broad-bounded-tensors-plan-v1.json'
    traces=ROOT/'results/4090-hey-broad-logit-sentinel-result-v1.json'
    screenpath=ROOT/'results/4090-hey-broad-bounded-screen-result-v1.json'
    execution=ROOT/'results/4090-hey-broad-bounded-screen-execution-v1.json'
    export=ROOT/'results/4090-hey-broad-bounded-export-result-v1.json'
    locked=json.loads(parent.read_text());files=dict(locked['files']);cases=[]
    for p in [source,parent,traces,screenpath,execution,export,Path(__file__)]:files[str(p)]=sha(p)
    logs=json.loads(traces.read_text())['cases'];screen=json.loads(screenpath.read_text())['cases']
    for row in json.loads(source.read_text())['cases']:
        target,order=row['target'],row['request_order']
        files[row['arrays_path']]=row['arrays_sha256']
        log=next(r['arms']['natural'] for r in logs if r['target']==target and r['request_order']==order)
        files[log['arrays_path']]=log['arrays_sha256']
        scored=next(r for r in screen if r['target']==target and r['request_order']==order)
        files[scored['arrays_path']]=scored['arrays_sha256']
        cases.append(dict(tensor_trace=row,original_logit_trace=log,scored=scored))
    result=dict(schema='hey-broad-bounded-head-decomposition-plan-v1',files=files,cases=cases,versions=locked['versions'],policy='Same two frozen worst-error training witnesses. Read original H5 Dense W,b and actual bounded TFLite quantized head constants, no invoke. A=bounded original float probability; B=sigmoid clip(original float W,b applied to dequantized actual INT8 encoder embedding); D=quantized transformed W,b dequantized and applied to same embedding, clip RELU6 then inverse affine; E=actual saved FC output dequantized then inverse affine/sigmoid; F=actual final UINT8. Float64 mathematical references, not independent integer kernel oracle. Save per-step A/B/D/E/F and original/embedding logits. Signed F-A telescopes (B-A)+(D-B)+(E-D)+(F-E) at1e-12. Reconstruct original clipped-logit A within1e-4. Report wholeinput five-step mean differences for each boundary, without gate changes.',limits='Two selected training witnesses only. Hybrid boundary decomposition does not attribute individual encoder layers or claim statistical causal share, complete270 accuracy, new candidate admission, recognition/FAR or freshdev/acceptance. No optimizer/PTQ/newexport.')
    with PLAN.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')

def sigmoid(z):return np.exp(-np.logaddexp(0,-np.asarray(z,np.float64)))
def stats(x):
    five=np.asarray([np.mean(x[j:j+5]) for j in range(len(x)-4)])
    return dict(raw_max_abs=float(np.max(np.abs(x))),five_max_abs=float(np.max(np.abs(five))),five_mean_abs=float(np.mean(np.abs(five))),five_p99_abs=float(np.percentile(np.abs(five),99)))
def dequant(value,detail):
    p=detail['quantization_parameters'];scales=p['scales'].astype(np.float64);zeros=p['zero_points'].astype(np.float64)
    assert len(scales)>0
    if len(scales)==1:return (value.astype(np.float64)-zeros[0])*scales[0]
    shape=[1]*value.ndim;shape[p['quantized_dimension']]=len(scales)
    return (value.astype(np.float64)-zeros.reshape(shape))*scales.reshape(shape)

def run():
    published(PLAN);locked=json.loads(PLAN.read_text())
    for p,h in locked['files'].items():assert sha(p)==h,p
    import importlib.metadata
    for n,v in locked['versions'].items():assert importlib.metadata.version(n)==v
    import h5py
    from ai_edge_litert.interpreter import Interpreter,OpResolverType
    execution=json.loads((ROOT/'results/4090-hey-broad-bounded-screen-execution-v1.json').read_text())
    export=json.loads((ROOT/'results/4090-hey-broad-bounded-export-result-v1.json').read_text())
    OUT.mkdir(exist_ok=False);cases=[]
    for row in locked['cases']:
        trace=row['tensor_trace'];target=trace['target']
        arm=next(r for r in export['arms'] if r['target']==target)
        fc,=[o for o in arm['ops'] if o['op']=='FULLY_CONNECTED'];assert fc['fused_activation']==3
        embedding_index,weight_index,bias_index=fc['inputs'];output_index,=fc['outputs']
        tensor={t['index']:t for t in trace['tensors']}
        with np.load(trace['arrays_path'],allow_pickle=False) as saved:
            quantized=saved[f'tensor_{embedding_index}'];actual_fc=saved[f'tensor_{output_index}'].reshape(-1)
        meta=tensor[embedding_index];embedding=(quantized.astype(np.float64)-meta['zero'])*meta['scale'];embedding=embedding.reshape(len(embedding),-1)
        folder=Path(execution['models'][target]);weights=folder/'best.weights.h5'
        assert sha(weights)==execution['files'][str(weights)]
        with h5py.File(weights) as h:w=np.array(h['layers/dense/vars/0']);b=np.array(h['layers/dense/vars/1'])
        assert w.shape==(embedding.shape[1],1) and b.shape==(1,)
        logit=(embedding@w.astype(np.float64)+b).reshape(-1);B=sigmoid(np.clip(logit,-8,8))
        interpreter=Interpreter(model_path=arm['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);interpreter.allocate_tensors()
        details={d['index']:d for d in interpreter.get_tensor_details()}
        qw=dequant(interpreter.get_tensor(weight_index),details[weight_index]);qb=dequant(interpreter.get_tensor(bias_index),details[bias_index])
        dense=embedding@qw.T+qb;D=sigmoid(np.clip(dense.reshape(-1),0,6)*(8/3)-8)
        fm=tensor[output_index];E=sigmoid(((actual_fc.astype(np.float64)-fm['zero'])*fm['scale'])*(8/3)-8)
        with np.load(row['scored']['arrays_path'],allow_pickle=False) as saved:
            A=saved['float'];oscale,ozero=row['scored']['output_quantization'];F=(saved['uint8'].astype(np.float64)-ozero)*oscale
        with np.load(row['original_logit_trace']['arrays_path'],allow_pickle=False) as saved:original_logit=saved['logit']
        assert np.max(np.abs(A-sigmoid(np.clip(original_logit,-8,8))))<=1e-4
        boundaries=dict(encoder_embedding=B-A,head_weight_bias=D-B,integer_fc=E-D,post_fc_probability=F-E)
        assert np.max(np.abs(sum(boundaries.values())-(F-A)))<=1e-12
        path=OUT/f'{target}.npz';np.savez(path,A=A,B=B,D=D,E=E,F=F,embedding_logit=logit,original_logit=original_logit)
        report=dict(target=target,request_order=trace['request_order'],boundaries={n:stats(v) for n,v in boundaries.items()},total=stats(F-A),arrays_path=str(path),arrays_sha256=sha(path),embedding_shape=list(embedding.shape),model_invocations=0)
        cases.append(report);print(json.dumps(report),flush=True)
    with RESULT.open('x') as f:f.write(json.dumps(dict(schema='hey-broad-bounded-head-decomposition-result-v1',plan_sha256=sha(PLAN),cases=cases,limits=locked['limits']),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
