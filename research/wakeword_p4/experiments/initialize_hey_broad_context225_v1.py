"""Save one exact paired context225 initialization; CPU, no model inference/update."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

RESULTS=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword')
PLAN=RESULTS/'4090-hey-broad-context225-initialization-plan-v1.json'
OUT=BASE/'hey-broad-context225-initialization-v1'
WEIGHTS=BASE/'mww-pretrained-reconstruction-v1/reconstructed.weights.h5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def plan():
    import tensorflow as tf
    helper=Path(__file__).with_name('probe_hey_context_transfer_shapes_v1.py')
    decision=RESULTS/'4090-hey-broad-context-decision-v1.json'
    assert json.loads(decision.read_text())['context_frames']==225
    d=dict(schema='hey-broad-context225-initialization-plan-v1',files={str(p):sha(p) for p in [Path(__file__),helper,decision,WEIGHTS]},
        tensorflow=tf.__version__,seed=726,context_frames=225,
        policy='Create original context150 and load original authenticated reconstructed weights; snapshot ordered layer arrays. Clear Keras session, seed726, create context225. Copy all non-Dense weights exactly, freeze BN; leave new classifier at its actual seeded initializer. Save same-instance H5 plus every variable value-byte inventory before any update. Both future arms must load this exact snapshot.',
        operations='CPU shape/weight assignment/save only; no data, forward inference, optimizer or export.')
    with PLAN.open('x') as f:f.write(json.dumps(d,indent=2)+'\n')

def run():
    import probe_hey_context_transfer_shapes_v1 as helper
    tf=helper.tf
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes();d=json.loads(remote)
    for p,h in d['files'].items():assert sha(Path(p))==h
    assert tf.__version__==d['tensorflow'] and not OUT.exists()
    tf.keras.utils.set_random_seed(d['seed']);original=helper.create(150);original.load_weights(WEIGHTS)
    old=[]
    for layer in original.layers:
        cell=getattr(layer,'cell',layer);old.append((type(cell).__name__,[a.copy() for a in layer.get_weights()]))
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(d['seed']);model=helper.create(225)
    copied=[];head=None
    for i,(layer,(kind,values)) in enumerate(zip(model.layers,old,strict=True)):
        cell=getattr(layer,'cell',layer);assert type(cell).__name__==kind
        if isinstance(cell,tf.keras.layers.Dense):
            head=layer;continue
        if values:
            layer.set_weights(values)
            assert all(np.array_equal(a,b) for a,b in zip(values,layer.get_weights(),strict=True))
            copied.append(dict(layer_index=i,kind=kind,sha256=hashlib.sha256(b''.join(a.tobytes() for a in values)).hexdigest()))
        if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=False;cell.trainable=False
    assert head is not None and model.count_params()==24301
    def inventory():
        return [dict(index=i,name=getattr(v,'path',v.name),shape=list(v.shape),dtype=str(v.dtype),
                     trainable=bool(v.trainable),sha256=hashlib.sha256(np.asarray(v.numpy()).tobytes()).hexdigest()) for i,v in enumerate(model.variables)]
    before=inventory();OUT.mkdir();snapshot=OUT/'initial.weights.h5';model.save_weights(snapshot)
    assert inventory()==before
    (OUT/'variables.json').write_text(json.dumps(before,indent=2)+'\n')
    # Reload same architecture from actual saved bytes and independently compare all values.
    for v in model.variables:v.assign(np.zeros(v.shape,dtype=v.numpy().dtype))
    model.load_weights(snapshot);assert inventory()==before
    hd=head.get_weights();result=dict(schema='hey-broad-context225-initialization-v1',plan_sha256=sha(PLAN),seed=d['seed'],
        context_frames=225,parameters=model.count_params(),snapshot=dict(path=str(snapshot),bytes=snapshot.stat().st_size,sha256=sha(snapshot)),
        variable_inventory=dict(path=str(OUT/'variables.json'),sha256=sha(OUT/'variables.json'),variables=before),
        copied_encoder_bn=copied,classifier_shapes=[list(a.shape) for a in hd],
        classifier_value_sha256=hashlib.sha256(b''.join(a.tobytes() for a in hd)).hexdigest(),
        same_instance_saved=True,reloaded_variables_exact=True,updates=0,inference_calls=0,
        limits='Exact context225 initialization only; original context150 classifier intentionally differs. No training, stream/export or acceptance evidence.')
    with (RESULTS/'4090-hey-broad-context225-initialization-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(parameters=model.count_params(),snapshot_sha256=sha(snapshot),classifier_sha256=result['classifier_value_sha256'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args()
    plan() if a.action=='plan' else run()
