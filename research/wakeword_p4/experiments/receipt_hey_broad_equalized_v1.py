"""Immutable equalized export provenance plus independent saved-array audit, no invoke."""
import json
from pathlib import Path
import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
from run_hey_broad_trained_quantization_v1 import ROOT,published

RESULTS=ROOT/'results'
OUT=RESULTS/'4090-hey-broad-equalized-export-receipt-v1.json'

def states(path):
    from ai_edge_litert.interpreter import Interpreter,OpResolverType
    i=Interpreter(model_path=str(path),experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1)
    i.allocate_tensors() # Metadata only, never invoke.
    tensors={t['index']:t for t in i.get_tensor_details()};ops=i._get_ops_details()
    reads={int(o['inputs'][0]):tensors[int(o['outputs'][0])] for o in ops if o['op_name']=='READ_VARIABLE'}
    writes={int(o['inputs'][0]):tensors[int(o['inputs'][1])] for o in ops if o['op_name']=='ASSIGN_VARIABLE'}
    assert reads.keys()==writes.keys() and len(reads)==6
    rows=[]
    for handle in sorted(reads):
        a,b=reads[handle],writes[handle]
        matched=bool(np.array_equal(a['shape'],b['shape']) and a['dtype']==b['dtype'] and a['quantization']==b['quantization'])
        assert matched
        rows.append(dict(handle=handle,read_name=a['name'],assigned_name=b['name'],shape=a['shape'].tolist(),dtype=str(a['dtype']),quantization=list(a['quantization']),read_assign_metadata_match=matched))
    return rows

def artifact(path):
    p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))

def main():
    assert not OUT.exists()
    paths={name:RESULTS/f'4090-hey-broad-{name}-v1.json' for name in ['equalized-export-execution','equalized-export-result','equalized-screen-execution','equalized-screen-result','equalized-float-plan','equalized-float-result','ptq-material-plan','ptq-material-result','trained-quantization-plan']}
    for name in ['equalized-weights-receipt','equalized-calibration-result','channel-ranges-result']:
        paths[name]=RESULTS/f'4090-hey-broad-{name}-v1.json'
    records={}
    for name,p in paths.items():published(p);records[name]=json.loads(p.read_text())
    execution=records['equalized-export-execution'];screen=records['equalized-screen-result'];export=records['equalized-export-result'];control=records['equalized-float-result']
    assert export['execution_sha256']==sha(paths['equalized-export-execution'])
    assert screen['execution_sha256']==sha(paths['equalized-screen-execution'])
    assert len(screen['cases'])==270 and len(control['cases'])==270
    gates=records['trained-quantization-plan']['numerical_gates'];audited=[]
    for row in screen['cases']:
        assert sha(row['arrays_path'])==row['arrays_sha256']
        with np.load(row['arrays_path'],allow_pickle=False) as a:
            exact=np.array_equal(a['float'],a['float_repeat']) and np.array_equal(a['uint8'],a['uint8_repeat'])
            scale,zero=row['output_quantization'];signed=(a['uint8'].astype(np.float64)-zero)*scale-a['float']
            five=np.asarray([np.mean(signed[j:j+5]) for j in range(len(signed)-4)])
            absolute=np.abs(five)
            statistics=dict(mean=float(np.mean(absolute)),p99=float(np.percentile(absolute,99,method='linear')),maximum=float(np.max(absolute)))
            for key,value in statistics.items():assert abs(value-row['smoothed_abs_error'][key])<=1e-12
            phase=float(np.max(np.abs(a['offline_phase']-a['float'][74:])))
            assert phase==row['phase_max_error']
            clocks=(3*np.arange(1,len(a['float'])+1)+2)*160
            assert np.array_equal(clocks,a['support_unclamped'])
            assert len(a['raw'])//3+(len(a['raw'])%3!=0)==len(a['float'])
            passed=bool(exact and np.isfinite(a['float']).all() and phase<=gates['float_offline_stream_max_abs'] and statistics['mean']<=gates['per_input_five_step_mean_abs_error_max'] and statistics['p99']<=gates['per_input_five_step_p99_abs_error_max'] and statistics['maximum']<=gates['per_input_five_step_max_abs_error_max'])
            assert passed==row['passed']
        audited.append(dict(target=row['target'],request_order=row['request_order'],passed=passed,arrays_sha256=row['arrays_sha256']))
    arms=[]
    for row in export['arms']:
        target=row['target'];folder=Path(execution['models'][target]);original=folder/'hey_nyx.tflite'
        for p in [folder/'best.weights.h5',folder/'training.json',original]:assert sha(p)==execution['files'][str(p)]
        assert sha(row['path'])==row['sha256']
        material=next(r for r in records['ptq-material-result']['arms'] if r['target']==target)
        assert sha(material['path'])==material['sha256']
        subset=[r for r in audited if r['target']==target];assert len(subset)==135
        arms.append(dict(target=target,original_final_h5=artifact(folder/'best.weights.h5'),original_training=artifact(folder/'training.json'),original_tflite=artifact(original),equalized_tflite=artifact(row['path']),representative_material=material,original_states=states(original),equalized_states=states(row['path']),numerical_passed=sum(r['passed'] for r in subset),numerical_total=135))
    receipt=dict(schema='hey-broad-equalized-export-receipt-v1',script_sha256=sha(__file__),artifacts={name:artifact(p) for name,p in paths.items()},arms=arms,channel_reparameterization=records['equalized-weights-receipt'],exact_head_transform=dict(kernel='W * float32(3/8)',bias='b * float32(3/8) + float32(3)',activation='RELU6 then Rescaling(8/3,offset=-8) then sigmoid'),state_evidence='Immutable original and compensated weight identities; full270 equalized float probabilities exact versus shared bounded reference and full7502/15site/sixactualstate scaling exact, per bound reports. Static six resource READ/ASSIGN pairing audited for each original and new INT8. This does not independently prove dynamic integer-state accuracy.',representative_order='zero, full26, original100 authenticated x[:1] arrays each75 consecutive slices, no source reset',independent_saved_array_audit=audited,all_numerical_passed=all(r['passed'] for r in audited),development_authorized=False,p4_candidate_ready=False,limits='Immutable provenance and numerical evidence, original training report never rewritten. Even numerical success requires separately frozen qualified fresh development before P4 candidate profile. No model invoke/export or fresh data access here.')
    with OUT.open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
