"""Measure easy/hard logit-gradient mass on actual saved training batches."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import tensorflow as tf
base=Path('.local/wakeword');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert tf.config.get_visible_devices('GPU');tf.config.experimental.enable_op_determinism();tf.config.experimental.enable_tensor_float_32_execution(False);tf.keras.utils.set_random_seed(726)
sys.path.insert(0,str((base/'microWakeWord').resolve()));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from microwakeword import mixednet
from batchnorm_policy import apply_policy
p=base/'paired-consistency-init-gradient-probe-v1/probe-batches.npz';assert sha(p)=='7d781dfc435417632aaa2c8995d6a955772b29b3ca3f88fbfd87d57108e994cb'
with np.load(p) as z:data={k:z[k] for k in ['main_x','main_y','main_weight','pair_x','pair_y','pair_weight']}
out={'batch_sha256':sha(p),'arms':{},'limits':'Original saved50 actual training batches only. Absolute derivative wrt each logit for weighted BCE; not parameter gradient or objective causality. Fixed p<.1 easy-negative diagnostic, not threshold selection. No optimizer/newdev/final or promotion.'}
for arm in ['control','treatment']:
 root=base/f'model-phrase-ranking-{arm}-formal-v1';training=json.loads((root/'training.json').read_text());weights=root/'best.weights.h5';audit=json.loads(Path(f'research/wakeword_p4/results/4090-phrase-ranking-{arm}-formal-v1-export-audit.json').read_text());assert sha(weights)==audit['weights_sha256']
 arch=argparse.ArgumentParser();mixednet.model_parameters(arch);model=mixednet.model(arch.parse_args(training['architecture_args']),(150,40),None);model.load_weights(weights);apply_policy(model,'frozen');before=[v.numpy().copy() for v in model.variables];result={}
 for prefix in ['main','pair']:
  probabilities=np.stack([model(x,training=False).numpy()[:,0] for x in data[prefix+'_x']]);y=data[prefix+'_y'][:,:,0];w=data[prefix+'_weight'];mass=np.abs(probabilities-y)*w/probabilities.shape[1];assert np.isfinite(mass).all()
  negative=y==0;easy=negative&(probabilities<.1);positive=y==1
  result[prefix]={'examples':int(y.size),'positive_logit_gradient_mass':float(mass[positive].sum()),'negative_logit_gradient_mass':float(mass[negative].sum()),'easy_negative_examples':int(easy.sum()),'negative_examples':int(negative.sum()),'easy_negative_mass_fraction_of_negative':float(mass[easy].sum()/mass[negative].sum()),'positive_probability_quantiles':np.quantile(probabilities[positive],[0,.1,.5,.9,1]).tolist()}
 for v,a in zip(model.variables,before):np.testing.assert_array_equal(v.numpy(),a)
 out['arms'][arm]={'weights_sha256':sha(weights),'all_variables_unchanged':True,**result}
out['script_sha256']=sha(Path(__file__));Path('research/wakeword_p4/results/4090-bce-logit-gradient-census-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
