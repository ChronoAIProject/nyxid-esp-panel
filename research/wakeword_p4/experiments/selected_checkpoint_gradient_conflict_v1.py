"""Selected phrase-ranking checkpoint gradient conflict on original saved training batches."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
p = argparse.ArgumentParser(); p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
p.add_argument('--checkpoint',choices=['control','treatment'],required=True); p.add_argument('--output', type=Path, required=True); a = p.parse_args(); assert not a.output.exists()
if a.device == 'cpu': tf.config.set_visible_devices([], 'GPU')
else: assert tf.config.get_visible_devices('GPU')
tf.config.experimental.enable_op_determinism()
tf.config.experimental.enable_tensor_float_32_execution(False)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.keras.utils.set_random_seed(726)
base = Path('.local/wakeword'); sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sys.path.insert(0, str((base/'microWakeWord').resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'training'))
from microwakeword import mixednet
from batchnorm_policy import apply_policy
from phrase_ranking import phrase_ranking, verified_phrase_groups
training = json.loads((base/'model-paired-control-smoke-v1/training.json').read_text())
arch = argparse.ArgumentParser(); mixednet.model_parameters(arch)
model = mixednet.model(arch.parse_args(training['architecture_args']), (150, 40), None)
checkpoint_root=base/f'model-phrase-ranking-{a.checkpoint}-formal-v1'
snapshot=checkpoint_root/'best.weights.h5'
audit=json.loads(Path(f'research/wakeword_p4/results/4090-phrase-ranking-{a.checkpoint}-formal-v1-export-audit.json').read_text())
assert sha(snapshot)==audit['weights_sha256']
model.load_weights(snapshot); apply_policy(model, 'frozen')
variables = model.trainable_variables; head = model.layers[-1]
extractor = tf.keras.Model(model.inputs, head.input)
batches = base/'paired-consistency-init-gradient-probe-v1/probe-batches.npz'
assert sha(batches) == '7d781dfc435417632aaa2c8995d6a955772b29b3ca3f88fbfd87d57108e994cb'
with np.load(batches) as z: values = [z[k][0] for k in ['main_x', 'main_y', 'main_weight', 'pair_x', 'pair_y', 'pair_weight']]
before = [v.numpy().copy() for v in model.variables]
pack=base/'factorized-training-features-v1'
assert sha(pack/'report.json')=='7b798653f73e2dcd3b01fd5b7092c947a26b9c8ec2525ff295de2e6478042e44'
metadata=json.loads((pack/'report.json').read_text())['records'];lookup={r['pack_index']:r for r in metadata}
probe=base/'paired-consistency-init-gradient-probe-v1/gradient-probe.json'
assert sha(probe)=='0cb06dcb3276125d56b0bc260ef7c6be7e57921fed03b4b799a482e3d675e42f'
traces=json.loads(probe.read_text())['pair_trace']
with np.load(batches) as z: data={k:z[k] for k in z.files}
raw_cache={};groups=[]
for i,tr in enumerate(traces):
    records=[lookup[k] for k in tr['pack_indices']];groups.append(verified_phrase_groups(records))
    end=tr['endpoint_inclusive'];assert end%3==0
    for j,r in enumerate(records):
        assert r['length_multiplier']==tr['speed'] and int(r['voice']) in tr['voices']
        assert r['feature_crop_end']<=end<=min(350,r['raw_shape'][0]-1)
        path=pack/r['raw']
        if r['pack_index'] not in raw_cache:
            assert sha(path)==r['raw_file_sha256'];raw_cache[r['pack_index']]=np.load(path,allow_pickle=False)
            assert hashlib.sha256(raw_cache[r['pack_index']].tobytes()).hexdigest()==r['raw_sha256']
            assert raw_cache[r['pack_index']].dtype==np.uint16 and list(raw_cache[r['pack_index']].shape)==r['raw_shape']
        window=raw_cache[r['pack_index']][end-149:end+1]
        assert hashlib.sha256(window.tobytes()).hexdigest()==tr['window_sha256'][j]
        np.testing.assert_array_equal(np.clip(window.astype(np.float32)/25.6,0,26),data['pair_x'][i,j])
        assert data['pair_y'][i,j,0]==float(r['target']=='hey_nyx')
@tf.function
def infer(x,y,w,sx,sy,sw,indices):
    with tf.GradientTape(persistent=True) as tape:
        e=extractor(x,training=True);se=extractor(sx,training=True)
        logits=tf.matmul(e,head.kernel)+head.bias;sl=tf.matmul(se,head.kernel)+head.bias
        main=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logits)[:,0]*w)
        pair=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=sy,logits=sl)[:,0]*sw)
        supervised=main+pair;ranking=phrase_ranking(sl,indices)
    gs=tape.gradient(supervised,variables);gr=tape.gradient(ranking,variables)
    assert all(g is not None for g in gs+gr)
    return tf.stack([main,pair,ranking]),gs,gr,sl
arrays={'metrics':[],'supervised':[],'ranking':[],'pair_logits':[]};observations=[]
for i in range(50):
    args=[data[k][i] for k in ['main_x','main_y','main_weight','pair_x','pair_y','pair_weight']]
    metrics,gs,gr,sl=infer(*args,np.array(groups[i],np.int32))
    gs=np.concatenate([g.numpy().ravel() for g in gs]);gr=np.concatenate([g.numpy().ravel() for g in gr])
    for k,v in [('metrics',metrics.numpy()),('supervised',gs),('ranking',gr),('pair_logits',sl.numpy())]:
        assert np.isfinite(v).all();arrays[k].append(v)
    ns=float(np.linalg.norm(gs.astype(np.float64)));nr=float(np.linalg.norm(gr.astype(np.float64)));assert ns>0 and nr>0
    observations.append({'batch':i,'supervised_norm':ns,'ranking_norm':nr,'cosine':float(gs.astype(np.float64)@gr.astype(np.float64)/(ns*nr))})
    if i%10==0:print(json.dumps(observations[-1]),flush=True)
for x,v in zip(before,model.variables):np.testing.assert_array_equal(x,v.numpy())
a.output.mkdir();np.savez_compressed(a.output/'arrays.npz',**{k:np.stack(v) for k,v in arrays.items()})
r={'checkpoint':a.checkpoint,'device':a.device,'batches':50,'raw_windows_authenticated':2000,'metadata_groups_authenticated':500,'optimizer_updates':0,'all_variables_unchanged':True,'all_gradients_connected_and_finite':True,'fixed_coefficient':1,'coefficient_reestimated':False,'snapshot_sha256':sha(snapshot),'batches_sha256':sha(batches),'arrays_sha256':sha(a.output/'arrays.npz'),'script_sha256':sha(Path(__file__)),'observations':observations,'limits':'Selected-checkpoint old training batches only; not held-out or streaming evidence, no optimizer or tuning. Main/paired supervised combined versus ranking gradients, not unique causal attribution. Bias ranking gradient may be zero due to common-shift invariance.'}
(a.output/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='observations'}))
