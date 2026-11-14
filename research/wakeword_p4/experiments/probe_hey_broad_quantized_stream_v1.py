"""Full saved-waveform surrogate state/phase and actual-INT8 comparisons."""
import argparse,json
from pathlib import Path
import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
from run_hey_broad_trained_quantization_v1 import ROOT,published,errors
PLAN=ROOT/'results/4090-hey-broad-quantized-stream-plan-v1.json'
RESULT=ROOT/'results/4090-hey-broad-quantized-stream-result-v1.json'
OUT=Path('.local/wakeword/hey-broad-quantized-stream-v1')
def plan():
    parent=ROOT/'results/4090-hey-broad-quantized-operator-plan-v2.json'
    old=json.loads(parent.read_text());files=dict(old['files'])
    actual=ROOT/'results/4090-hey-broad-equalized-screen-result-v1.json'
    primitive=ROOT/'results/4090-hey-broad-quantized-operator-result-v2.json'
    for p in [parent,actual,primitive,Path(__file__),Path(__file__).with_name('hey_broad_quantized_stream_v1.py')]:files[str(p)]=sha(p)
    for row in json.loads(actual.read_text())['cases']:files[row['arrays_path']]=row['arrays_sha256']
    r=dict(schema='hey-broad-quantized-stream-plan-v1',files=files,versions=old['versions'],arms=old['arms'],export=old['export'],actual=str(actual),policy='Same270wholetrainingfeature inputs and actualsaved equalizedINT8 traces, no modelconversion/newdata. Surrogate explicit sixfloatstate tensors on fixedquantizationgrids, firstinputhistory2frames, fourdepthwisehistories4/8/12/20, final29activations. Zeroreset twicecompleteinput, fullstate/outputhash exactrepeat. Original225offlinegraph phase224+unusedzero mustmatchstreamprob<=1e-4; rawlogit discrepancy separatelyretained. Use originalfive MAE.02/p99.1/max.25 thresholds to assess surrogate fidelity to actualINT8, never candidateadmission. Save alltraces/phase/clocks, allcases evenfailures. CPU2threads, nooptimizer/parameterschanged.',limits='Nearest-even float-arithmetic quantization surrogate, not bitexact integerkernel. Phase or runtime-fidelity failure retained; doesnot authorize formaltraining/dev/acceptance/export.')
    with PLAN.open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
def run():
    published(PLAN);lock=json.loads(PLAN.read_text())
    for p,h in lock['files'].items():assert sha(p)==h,p
    import importlib.metadata,hashlib
    for n,v in lock['versions'].items():assert importlib.metadata.version(n)==v
    import probe_hey_context_transfer_shapes_v1 as helper
    tf=helper.tf
    from hey_broad_quantized_stream_v1 import configuration,operators
    exports=json.loads(Path(lock['export']).read_text())['arms'];actual=json.loads(Path(lock['actual']).read_text())['cases']
    OUT.mkdir(exist_ok=False);cases=[]
    shapes=[[1,2,1,40],[1,4,1,30],[1,8,1,60],[1,12,1,60],[1,20,1,60],[1,29,1,60]]
    specs=[tf.TensorSpec(s,tf.float32) for s in shapes]
    for arm in lock['arms']:
        target=arm['target'];model=helper.create(225);model.load_weights(arm['path'])
        for l in model.layers:
            c=getattr(l,'cell',l)
            if isinstance(c,tf.keras.layers.BatchNormalization):l.trainable=c.trainable=False
        before=[v.numpy().copy() for v in model.variables]
        config=configuration(next(r for r in exports if r['target']==target))
        stream=tf.function(operators(model,config,tf,True),input_signature=[tf.TensorSpec([1,3,40],tf.float32),specs])
        offline=tf.function(operators(model,config,tf),input_signature=[tf.TensorSpec([None,225,40],tf.float32)])
        for order,row in enumerate(r for r in actual if r['target']==target):
            with np.load(row['arrays_path'],allow_pickle=False) as saved:
                raw=saved['raw'];codes=saved['uint8'];clocks=saved['sample_end'];supports=saved['support_unclamped']
            values=np.clip(np.pad(raw,((0,(-len(raw))%3),(0,0))).astype(np.float32)/25.6,0,26)
            outputs=[];statehash=[]
            for repeat in range(2):
                states=[tf.zeros(s,tf.float32) for s in shapes];traces=[];digest=hashlib.sha256()
                for at in range(0,len(values),3):
                    logit,probability,states=stream(values[None,at:at+3],states)
                    traces.append([float(logit.numpy()[0,0]),float(probability.numpy()[0,0])])
                    for state in states:digest.update(state.numpy().tobytes())
                outputs.append(np.asarray(traces));statehash.append(digest.hexdigest())
            exact=np.array_equal(*outputs) and statehash[0]==statehash[1]
            windows=np.stack([np.concatenate([values[end-224:end],np.zeros((1,40),np.float32)]) for end in range(225,len(values)+1,3)])
            wl,wp=offline(windows);phase=wp.numpy().reshape(-1);phase_logits=wl.numpy().reshape(-1)
            probability=outputs[0][:,1];logits=outputs[0][:,0]
            pe=float(np.max(np.abs(phase-probability[74:])));le=float(np.max(np.abs(phase_logits-logits[74:])))
            scale,zero=row['output_quantization'];dequant=(codes.astype(np.float64)-zero)*scale
            signed=probability-dequant;smooth=np.convolve(signed,np.ones(5)/5,mode='valid');e=errors(np.abs(smooth))
            fidelity=bool(e['mean']<=.02 and e['p99']<=.1 and e['maximum']<=.25)
            path=OUT/f'{target}-{order:03}.npz'
            np.savez(path,probability=probability,logit=logits,repeat=outputs[1],actual_int8=dequant,offline_phase=phase,offline_logit_phase=phase_logits,sample_end=clocks,support_unclamped=supports)
            case=dict(target=target,request_order=row['request_order'],twice_output_state_exact=exact,phase_probability_max_error=pe,phase_logit_max_error=le,phase_passed=pe<=1e-4,actual_int8_smoothed_abs_error=e,runtime_fidelity_passed=fidelity,arrays_path=str(path),arrays_sha256=sha(path),state_trace_sha256=statehash[0]);cases.append(case)
            with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(case)+'\n')
            print(json.dumps(dict(target=target,completed=order+1,phase=case['phase_passed'],runtime_fidelity=fidelity)),flush=True)
        assert all(np.array_equal(a,v.numpy()) for a,v in zip(before,model.variables,strict=True))
    assert len(cases)==270
    with RESULT.open('x') as f:f.write(json.dumps(dict(schema='hey-broad-quantized-stream-result-v1',plan_sha256=sha(PLAN),cases=cases,all_phase_passed=all(r['phase_passed'] for r in cases),all_runtime_fidelity_passed=all(r['runtime_fidelity_passed'] for r in cases),limits=lock['limits']),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
