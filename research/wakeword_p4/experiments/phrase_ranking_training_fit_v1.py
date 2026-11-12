"""Bounded selected-checkpoint fit on the original fifty training batches."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import tensorflow as tf
base = Path('.local/wakeword')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert tf.config.get_visible_devices('GPU')
tf.config.experimental.enable_op_determinism()
tf.config.experimental.enable_tensor_float_32_execution(False)
tf.keras.utils.set_random_seed(726)
sys.path.insert(0, str((base/'microWakeWord').resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'training'))
from microwakeword import mixednet
from batchnorm_policy import apply_policy
from phrase_ranking import verified_phrase_groups
batchpath = base/'paired-consistency-init-gradient-probe-v1/probe-batches.npz'
assert sha(batchpath) == '7d781dfc435417632aaa2c8995d6a955772b29b3ca3f88fbfd87d57108e994cb'
probepath = base/'paired-consistency-init-gradient-probe-v1/gradient-probe.json'
assert sha(probepath) == '0cb06dcb3276125d56b0bc260ef7c6be7e57921fed03b4b799a482e3d675e42f'
traces = json.loads(probepath.read_text())['pair_trace']
packpath = base/'factorized-training-features-v1/report.json'
assert sha(packpath) == '7b798653f73e2dcd3b01fd5b7092c947a26b9c8ec2525ff295de2e6478042e44'
records = json.loads(packpath.read_text())['records']
with np.load(batchpath, allow_pickle=False) as z: x = z['pair_x'].copy()
assert x.shape == (50,40,150,40)
indices = [np.asarray(verified_phrase_groups([records[k] for k in trace['pack_indices']])) for trace in traces]
output = {'batches_sha256': sha(batchpath), 'pack_report_sha256': sha(packpath), 'arms': {},
 'limits': 'Original first50 training batches only, correlated and reused, not held-out evidence. Float windows, not streaming INT8 or complete waveform metrics. Zero optimizer updates; no coefficient/threshold selection, new dev/final access or promotion.'}
for arm in ('control','treatment'):
 root = base/f'model-phrase-ranking-{arm}-formal-v1'
 training = json.loads((root/'training.json').read_text())
 assert training['selected_step'] == 5000 and sha(root/'hey_nyx.tflite') == training['model_sha256']
 audit = json.loads(Path(f'research/wakeword_p4/results/4090-phrase-ranking-{arm}-formal-v1-export-audit.json').read_text())
 # Record actual selected H5 identity and require the original export audit to authenticate it.
 weights = root/'best.weights.h5'
 assert sha(weights) in json.dumps(audit)
 arch = argparse.ArgumentParser(); mixednet.model_parameters(arch)
 model = mixednet.model(arch.parse_args(training['architecture_args']), (150,40), None)
 model.load_weights(weights); apply_policy(model,'frozen')
 before = [v.numpy().copy() for v in model.variables]
 head = model.layers[-1]; extractor = tf.keras.Model(model.inputs,head.input)
 logits = []
 for batch in x:
  embeddings = extractor(batch,training=False)
  logits.append((tf.matmul(embeddings,head.kernel)+head.bias).numpy()[:,0])
 logits = np.stack(logits); assert np.isfinite(logits).all()
 for previous,variable in zip(before,model.variables): np.testing.assert_array_equal(previous,variable.numpy())
 grouped = np.stack([values[index] for values,index in zip(logits,indices)])
 margins = grouped[:,:,0]-np.max(grouped[:,:,1:],axis=2)
 probabilities = tf.math.sigmoid(logits).numpy()
 summary = {'weights_sha256':sha(weights), 'training_sha256':sha(root/'training.json'),
  'model_sha256':training['model_sha256'], 'groups':500, 'target_strict_top1':int(np.count_nonzero(margins>0)),
  'margin_quantiles':np.quantile(margins,[0,.1,.5,.9,1]).tolist(), 'by_rendering':{},
  'all_variables_unchanged':True}
 for rendering in ['quiet','noise25','combined','noise15','rir']:
  values = np.asarray([margins[b,g] for b in range(50) for g in range(10)
                       if records[traces[b]['pack_indices'][indices[b][g,0]]]['arm'] == rendering])
  summary['by_rendering'][rendering]={'groups':int(values.size),'target_strict_top1':int(np.count_nonzero(values>0))}
 summary['fixed_probability_diagnostics']={}
 for threshold in [.5,.992,.99609375]:
  pos=[]; neg=[]
  for values,trace in zip(probabilities,traces):
   for value,k in zip(values,trace['pack_indices']):
    (pos if records[k]['target']=='hey_nyx' else neg).append(value)
  summary['fixed_probability_diagnostics'][str(threshold)]={'positive_above':int(np.count_nonzero(np.asarray(pos)>=threshold)), 'positives':len(pos),'negative_above':int(np.count_nonzero(np.asarray(neg)>=threshold)), 'negatives':len(neg)}
 output['arms'][arm]=summary
 print(arm,json.dumps(summary),flush=True)
output['script_sha256']=sha(Path(__file__))
Path('research/wakeword_p4/results/4090-phrase-ranking-training-fit-v1.json').write_text(json.dumps(output,indent=2)+'\n')
