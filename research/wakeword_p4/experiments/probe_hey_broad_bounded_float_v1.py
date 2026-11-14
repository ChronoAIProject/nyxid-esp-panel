"""Register/run exact bounded final-head float streaming control, no PTQ."""
import argparse
import json
from pathlib import Path
import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
from run_hey_broad_trained_quantization_v1 import ROOT, published

PLAN = ROOT/'results/4090-hey-broad-bounded-float-plan-v1.json'
RESULT = ROOT/'results/4090-hey-broad-bounded-float-result-v1.json'
OUT = Path('.local/wakeword/hey-broad-bounded-float-v1')

def plan():
    parent = ROOT/'results/4090-hey-broad-logit-sentinel-plan-v1.json'
    prior = json.loads(parent.read_text())
    traces = ROOT/'results/4090-hey-broad-logit-sentinel-result-v1.json'
    files = dict(prior['files'])
    files.update({str(p):sha(p) for p in [parent,traces,Path(__file__),ROOT/'results/4090-bounded-head-converter-probe-v1.json']})
    for row in json.loads(traces.read_text())['cases']:
        arm = row['arms']['natural']; files[arm['arrays_path']] = arm['arrays_sha256']
    record = dict(schema='hey-broad-bounded-float-plan-v1', files=files, versions=prior['versions'], models=prior['models'], original_result_path=prior['original_result_path'], traces_path=str(traces),
        policy='All original270 whole training feature witnesses, CPU2threads. Original final head W,b becomes RELU6 Dense W*3/8,b*3/8+3 then Rescaling8/3,-8 and sigmoid. Encoder weights/BN exact, six fresh-zero stream states, two exact reset repeats per case. Compare every output to sigmoid(clip(original saved natural logit,-8,8)) at absolute1e-4 (same registered float streaming phase tolerance). Independently retain original probability difference and theoretical clipping bound sigmoid(-8)+1e-4. No optimization or PTQ. Preserve original saved arrays and source files.',
        limits='Float structural control only, not INT8 admission or candidate recognition. No new development/acceptance/threshold choice. Existing270 INT8 gates unchanged; a separately pinned representative replay is required before real export.')
    with PLAN.open('x') as f: f.write(json.dumps(record,indent=2)+'\n')

def bounded(model, tf):
    heads = [l for l in model.layers if isinstance(l,tf.keras.layers.Dense)]
    assert len(heads)==1
    head = heads[0]
    w,b = head.get_weights()
    layer = tf.keras.layers.Dense(1,activation=tf.nn.relu6,name='bounded_head_relu6')
    output = layer(head.input)
    output = tf.keras.layers.Rescaling(8/3,offset=-8,name='bounded_head_logit')(output)
    output = tf.keras.layers.Activation('sigmoid',name='bounded_head_probability')(output)
    result = tf.keras.Model(model.inputs,output)
    layer.set_weights([w*np.float32(3/8),b*np.float32(3/8)+np.float32(3)])
    return result

def run():
    published(PLAN); locked=json.loads(PLAN.read_text())
    for p,h in locked['files'].items(): assert sha(p)==h,p
    import importlib.metadata
    for n,v in locked['versions'].items(): assert importlib.metadata.version(n)==v
    OUT.mkdir(exist_ok=False)
    import probe_hey_context_transfer_shapes_v1 as helper
    tf=helper.tf
    from microwakeword import utils
    from microwakeword.layers import modes
    original=json.loads(Path(locked['original_result_path']).read_text())
    traces={(r['target'],r['request_order']):r for r in json.loads(Path(locked['traces_path']).read_text())['cases']}
    cases=[]
    for target,folder in locked['models'].items():
        model=helper.create(225);model.load_weights(Path(folder)/'best.weights.h5')
        for l in model.layers:
            cell=getattr(l,'cell',l)
            if isinstance(cell,tf.keras.layers.BatchNormalization): l.trainable=cell.trainable=False
        initial=[v.numpy().copy() for v in model.variables]
        controlled=bounded(model,tf)
        assert all(np.array_equal(a,v.numpy()) for a,v in zip(initial,model.variables,strict=True))
        # Every encoder layer is the same original object; only final head is replaced.
        assert all(controlled.get_layer(l.name) is l for l in model.layers if not isinstance(l,tf.keras.layers.Dense))
        stream=utils.to_streaming_inference(controlled,{'stride':3,'spectrogram_length':225},modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
        stream(np.zeros((1,3,40),np.float32),training=False)
        states=[v for v in stream.variables if 'states' in v.name];assert len(states)==6
        infer=tf.function(lambda x:stream(x,training=False),input_signature=[tf.TensorSpec([1,3,40],tf.float32)])
        for row in (r for r in original['cases'] if r['target']==target):
            with np.load(row['arrays_path'],allow_pickle=False) as f: raw=f['raw']; baseline=f['float']
            arm=traces[target,row['request_order']]['arms']['natural']
            with np.load(arm['arrays_path'],allow_pickle=False) as f: logits=f['logit']
            values=np.clip(np.pad(raw,((0,(-len(raw))%3),(0,0))).astype(np.float32)/25.6,0,26)
            repeats=[]
            for _ in range(2):
                for state in states: state.assign(tf.zeros_like(state))
                repeats.append(np.asarray([float(infer(values[None,j:j+3]).numpy()[0,0]) for j in range(0,len(values),3)]))
            exact=np.array_equal(*repeats)
            expected=np.exp(-np.logaddexp(0,-np.clip(logits,-8,8)))
            error=float(np.max(np.abs(repeats[0]-expected)))
            perturbation=float(np.max(np.abs(repeats[0]-baseline)))
            passed=bool(exact and np.isfinite(repeats[0]).all() and error<=1e-4 and perturbation<=1/(1+np.exp(8))+1e-4)
            path=OUT/f'{target}-{row["request_order"]:06}.npz'
            np.savez(path,probability=repeats[0],repeat=repeats[1],expected=expected)
            case=dict(target=target,request_order=row['request_order'],twice_reset_exact=exact,clipped_logit_probability_max_error=error,original_probability_max_difference=perturbation,passed=passed,arrays_path=str(path),arrays_sha256=sha(path))
            cases.append(case)
            with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(case)+'\n')
            print(json.dumps(case),flush=True)
    result=dict(schema='hey-broad-bounded-float-result-v1',plan_sha256=sha(PLAN),cases=cases,all_passed=all(r['passed'] for r in cases),limits=locked['limits'])
    with RESULT.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['plan','run']);args=parser.parse_args()
    plan() if args.action=='plan' else run()
