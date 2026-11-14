"""Registered two-witness integer internal trace; no conversion or optimization."""
import argparse
import json
from pathlib import Path
import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
from run_hey_broad_trained_quantization_v1 import ROOT,published

PLAN=ROOT/'results/4090-hey-broad-bounded-tensors-plan-v1.json'
RESULT=ROOT/'results/4090-hey-broad-bounded-tensors-result-v1.json'
OUT=Path('.local/wakeword/hey-broad-bounded-tensors-v1')

def plan():
    screen=ROOT/'results/4090-hey-broad-bounded-screen-result-v1.json'
    execution=ROOT/'results/4090-hey-broad-bounded-screen-execution-v1.json'
    original=json.loads(screen.read_text());lock=json.loads(execution.read_text())
    selected=[];files=dict(lock['files'])
    for target in ['hey_willow','hey_nyx']:
        # Maximum saved five-error, tie by ascending original request order.
        row=min((r for r in original['cases'] if r['target']==target),key=lambda r:(-r['smoothed_abs_error']['maximum'],r['request_order']))
        selected.append(row);files[row['arrays_path']]=row['arrays_sha256']
    for p in [screen,execution,Path(__file__)]:files[str(p)]=sha(p)
    result=dict(schema='hey-broad-bounded-tensors-plan-v1',files=files,versions=lock['versions'],exports=lock['exports'],cases=selected,policy='Exactly maximum saved five-error witness/arm, ties original request order. Original saved exact quantized input, builtin reference1thread, experimental_preserve_all_tensors true. Twice fresh interpreter each wholeinput; record all quantized operation-output int8/uint8 tensors every step. Must exactly reproduce original full saved UINT8 outputs and all tensor repeats. Save original tensor identities/producer ops/quantization, per-step extrema and raw endpoint occupancy. Endpoint occupancy is not proof of clipping or quantization error. No float intermediate matching or attribution claimed.',limits='Training-only mechanism observation on two deterministically selected existing failure witnesses. Not coverage-wide saturation or recognition/FAR evidence. No new model/export/PTQ/training/newdev/acceptance/tolerance change.')
    with PLAN.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')

def run():
    published(PLAN);locked=json.loads(PLAN.read_text())
    for p,h in locked['files'].items():assert sha(p)==h,p
    import importlib.metadata
    for n,v in locked['versions'].items():assert importlib.metadata.version(n)==v
    from ai_edge_litert.interpreter import Interpreter,OpResolverType
    OUT.mkdir(exist_ok=False);cases=[]
    for row in locked['cases']:
        with np.load(row['arrays_path'],allow_pickle=False) as source:q=source['quantized_input'];baseline=source['uint8']
        repeats=[];metadata=None
        for _ in range(2):
            interpreter=Interpreter(model_path=locked['exports'][row['target']],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1,experimental_preserve_all_tensors=True)
            interpreter.allocate_tensors();details={d['index']:d for d in interpreter.get_tensor_details()}
            producers={int(index):o['op_name'] for o in interpreter._get_ops_details() for index in o['outputs'] if index>=0}
            ids=sorted(i for i in producers if details[i]['dtype'] in (np.int8,np.uint8) and details[i]['quantization'][0]>0)
            inp=interpreter.get_input_details()[0];output=interpreter.get_output_details()[0]
            traces={i:[] for i in ids};scores=[]
            for at in range(0,len(q),3):
                interpreter.set_tensor(inp['index'],q[None,at:at+3]);interpreter.invoke()
                scores.append(int(interpreter.get_tensor(output['index'])[0,0]))
                for i in ids:traces[i].append(interpreter.get_tensor(i).copy())
            assert np.array_equal(np.asarray(scores,np.uint8),baseline),'Preservation changed full output'
            traces={i:np.stack(v) for i,v in traces.items()};repeats.append(traces)
            metadata=[dict(index=i,name=details[i]['name'],producer=producers[i],shape=details[i]['shape'].tolist(),dtype=str(details[i]['dtype']),scale=details[i]['quantization'][0],zero=details[i]['quantization'][1]) for i in ids]
        assert repeats[0].keys()==repeats[1].keys()
        assert all(np.array_equal(repeats[0][i],repeats[1][i]) for i in repeats[0])
        path=OUT/f'{row["target"]}.npz';np.savez_compressed(path,**{f'tensor_{i}':v for i,v in repeats[0].items()})
        for m in metadata:
            value=repeats[0][m['index']];flat=value.reshape(len(value),-1);info=np.iinfo(value.dtype)
            m.update(minimum=int(value.min()),maximum=int(value.max()),lower_endpoint_fraction=float(np.mean(value==info.min)),upper_endpoint_fraction=float(np.mean(value==info.max)),per_step_min=flat.min(axis=1).tolist(),per_step_max=flat.max(axis=1).tolist())
        case=dict(target=row['target'],request_order=row['request_order'],original_uint8_exact=True,all_tensors_twice_reset_exact=True,arrays_path=str(path),arrays_sha256=sha(path),tensors=metadata)
        cases.append(case);print(json.dumps(dict(target=row['target'],tensors=len(metadata),steps=len(baseline))),flush=True)
    with RESULT.open('x') as f:f.write(json.dumps(dict(schema='hey-broad-bounded-tensors-result-v1',plan_sha256=sha(PLAN),cases=cases,limits=locked['limits']),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
