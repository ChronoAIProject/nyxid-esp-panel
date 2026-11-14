"""Lock and verify all270 whole-wave float controls before equalized export."""
import argparse
import json
from pathlib import Path
import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
from run_hey_broad_trained_quantization_v1 import ROOT,published
PLAN=ROOT/'results/4090-hey-broad-equalized-float-plan-v1.json'
RECEIPT=ROOT/'results/4090-hey-broad-equalized-weights-receipt-v1.json'
RESULT=ROOT/'results/4090-hey-broad-equalized-float-result-v1.json'
OUT=Path('.local/wakeword/hey-broad-equalized-v1')

def plan():
    parent=ROOT/'results/4090-hey-broad-bounded-float-plan-v1.json'
    baseline=ROOT/'results/4090-hey-broad-bounded-float-result-v1.json'
    channels=ROOT/'results/4090-hey-broad-channel-ranges-result-v1.json'
    old=json.loads(parent.read_text());files=dict(old['files'])
    for p in [parent,baseline,channels,Path(__file__),Path(__file__).with_name('equalize_hey_broad_channels_v1.py'),Path(__file__).with_name('probe_hey_broad_bounded_float_v1.py'),Path(__file__).with_name('hey_broad_exact_channel_equalization_v1.md')]:files[str(p)]=sha(p)
    for row in json.loads(baseline.read_text())['cases']:files[row['arrays_path']]=row['arrays_sha256']
    record=dict(schema='hey-broad-equalized-float-plan-v1',files=files,versions=old['versions'],models=old['models'],original_result_path=old['original_result_path'],baseline=str(baseline),channels=str(channels),policy='Prepare deterministic power2 compensated original-weight copies; publish immutable factor/weight receipt before inference. Original H5 unchanged. Run all270original wholeinput rawfeatures twicefreshsix-state CPU2threads; compare sharedboundedfloat savedprobability<=1e-4 and actualequalized offlinephase224+zero versusstream<=1e-4. Savealltraces/phase/repeats; no droppedfailure. Separate fullrepresentative15site scale relation control remains required before export; pin atol1e-4/rtol1e-5 for its scaledactivations before that execution. No export here.',limits='Float reparameterization control only, no INT8 admission, training, newdev/acceptance or threshold change.')
    with PLAN.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
def checked():
    published(PLAN);record=json.loads(PLAN.read_text())
    for p,h in record['files'].items():assert sha(p)==h,p
    import importlib.metadata
    for n,v in record['versions'].items():assert importlib.metadata.version(n)==v
    return record
def prepare():
    lock=checked();OUT.mkdir(exist_ok=False)
    import probe_hey_context_transfer_shapes_v1 as helper
    from equalize_hey_broad_channels_v1 import transform
    channels=json.loads(Path(lock['channels']).read_text());arms=[]
    for target,folder in lock['models'].items():
        m=helper.create(225);source=Path(folder)/'best.weights.h5';original_sha=sha(source);m.load_weights(source)
        original=[v.numpy().copy() for v in m.variables]
        record=transform(m,next(r for r in channels['cases'] if r['target']==target),helper.tf)
        dest=OUT/target;dest.mkdir();weights=dest/'equalized.weights.h5';m.save_weights(weights)
        assert sha(source)==original_sha
        inventory=[dict(shape=list(v.shape),before_sha256=__import__('hashlib').sha256(a.tobytes()).hexdigest(),after_sha256=__import__('hashlib').sha256(v.numpy().tobytes()).hexdigest()) for a,v in zip(original,m.variables,strict=True)]
        arms.append(dict(target=target,original_h5_path=str(source),original_h5_sha256=original_sha,path=str(weights),sha256=sha(weights),bytes=weights.stat().st_size,inventory=inventory,**record))
    with RECEIPT.open('x') as f:f.write(json.dumps(dict(schema='hey-broad-equalized-weights-receipt-v1',plan_sha256=sha(PLAN),arms=arms,limits=lock['limits']),indent=2)+'\n')
def run():
    lock=checked();published(RECEIPT);receipt=json.loads(RECEIPT.read_text());assert receipt['plan_sha256']==sha(PLAN)
    import probe_hey_context_transfer_shapes_v1 as helper
    from probe_hey_broad_bounded_float_v1 import bounded
    tf=helper.tf
    from microwakeword import utils
    from microwakeword.layers import modes
    original=json.loads(Path(lock['original_result_path']).read_text())
    baseline={(r['target'],r['request_order']):r for r in json.loads(Path(lock['baseline']).read_text())['cases']}
    arrays=OUT/'float-arrays';arrays.mkdir(exist_ok=False);cases=[]
    for arm in receipt['arms']:
        target=arm['target'];assert sha(arm['path'])==arm['sha256'] and sha(arm['original_h5_path'])==arm['original_h5_sha256']
        instances=[]
        for _ in range(2):
            m=helper.create(225);m.load_weights(arm['path'])
            for l in m.layers:
                c=getattr(l,'cell',l)
                if isinstance(c,tf.keras.layers.BatchNormalization):l.trainable=c.trainable=False
            instances.append(bounded(m,tf))
        stream=utils.to_streaming_inference(instances[0],{'stride':3,'spectrogram_length':225},modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
        stream(np.zeros((1,3,40),np.float32),training=False)
        states=[v for v in stream.variables if 'states' in v.name]
        assert sorted(list(v.shape) for v in states)==sorted(arm['state_shapes'])
        infer=tf.function(lambda x:stream(x,training=False),input_signature=[tf.TensorSpec([1,3,40],tf.float32)])
        window=tf.function(lambda x:instances[1](x,training=False),input_signature=[tf.TensorSpec([1,225,40],tf.float32)])
        for row in (r for r in original['cases'] if r['target']==target):
            with np.load(row['arrays_path'],allow_pickle=False) as saved:raw=saved['raw']
            values=np.clip(np.pad(raw,((0,(-len(raw))%3),(0,0))).astype(np.float32)/25.6,0,26)
            repeats=[]
            for _ in range(2):
                for state in states:state.assign(tf.zeros_like(state))
                repeats.append(np.asarray([float(infer(values[None,j:j+3]).numpy()[0,0]) for j in range(0,len(values),3)]))
            reference=baseline[target,row['request_order']]
            with np.load(reference['arrays_path'],allow_pickle=False) as saved:expected=saved['probability']
            error=float(np.max(np.abs(repeats[0]-expected)))
            phase=np.asarray([float(window(np.concatenate((values[end-224:end],np.zeros((1,40),np.float32)))[None]).numpy()[0,0]) for end in range(225,len(values)+1,3)])
            pe=float(np.max(np.abs(phase-repeats[0][74:])))
            exact=np.array_equal(*repeats);passed=bool(exact and np.isfinite(repeats[0]).all() and error<=1e-4 and pe<=1e-4)
            path=arrays/f'{target}-{row["request_order"]:06}.npz';np.savez(path,probability=repeats[0],repeat=repeats[1],baseline=expected,offline_phase=phase)
            case=dict(target=target,request_order=row['request_order'],baseline_max_abs_error=error,phase_max_abs_error=pe,twice_reset_exact=exact,passed=passed,arrays_path=str(path),arrays_sha256=sha(path));cases.append(case)
            with (OUT/'float-cases.jsonl').open('a') as f:f.write(json.dumps(case)+'\n')
            print(json.dumps(dict(target=target,completed=len(cases),passed=passed)),flush=True)
    assert len(cases)==270
    with RESULT.open('x') as f:f.write(json.dumps(dict(schema='hey-broad-equalized-float-result-v1',plan_sha256=sha(PLAN),weights_receipt_sha256=sha(RECEIPT),cases=cases,all_passed=all(r['passed'] for r in cases),limits=lock['limits']),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','prepare','run']);a=p.parse_args();{'plan':plan,'prepare':prepare,'run':run}[a.action]()
