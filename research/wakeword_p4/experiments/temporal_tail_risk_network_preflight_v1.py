"""Original fixed initialization, 50 independent authenticated temporal draws, zero updates."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import tensorflow as tf
base=Path('.local/wakeword');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tf.config.experimental.enable_op_determinism();tf.config.experimental.enable_tensor_float_32_execution(False);tf.keras.utils.set_random_seed(726)
assert tf.config.get_visible_devices('GPU')
sys.path.insert(0,str((base/'microWakeWord').resolve()))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from microwakeword import mixednet
from batchnorm_policy import apply_policy
from paired_temporal_sampler import PairedTemporalSampler
from phrase_ranking import phrase_ranking,verified_phrase_groups
from temporal_tail_risk import temporal_tail_risk
outdir=base/'temporal-tail-risk-network-preflight-v1';assert not outdir.exists()
training=json.loads((base/'model-phrase-ranking-control-formal-v1/training.json').read_text())
arch=argparse.ArgumentParser();mixednet.model_parameters(arch)
model=mixednet.model(arch.parse_args(training['architecture_args']),(150,40),None)
snapshot=base/'derived-probe-initialization-v1/derived-probe-initialization.weights.h5'
assert sha(snapshot)=='21f53feec03bdd73ae51cbc92091bf723d9de5fba07e3aa337c818ed722fe350'
model.load_weights(snapshot);apply_policy(model,'frozen')
head=model.layers[-1];extractor=tf.keras.Model(model.inputs,head.input);variables=model.trainable_variables
before=[v.numpy().copy() for v in model.variables]
sampler=PairedTemporalSampler(base/'factorized-training-features-v1')
# Independent metadata lookup and legal endpoint enumeration, not sampler helpers.
records=json.loads((sampler.root/'report.json').read_text())['records']
lookup={(int(r['voice']),r['length_multiplier'],r['target'],r['arm']):r for r in records}
rng=np.random.default_rng(237726)
@tf.function
def gradients(x,y,w,groups):
 with tf.GradientTape(persistent=True) as tape:
  embeddings=extractor(tf.reshape(x,[200,150,40]),training=True)
  logits=tf.reshape(tf.matmul(embeddings,head.kernel)+head.bias,[40,5])
  last=logits[:,-1]
  bce=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=last)*w)
  ranking=phrase_ranking(last,groups);supervised=bce+ranking
  risk=temporal_tail_risk(logits,y,w)
 gs=tape.gradient(supervised,variables);gt=tape.gradient(risk,variables)
 assert all(g is not None for g in gs+gt)
 return logits,tf.stack([bce,ranking,risk]),gs,gt
arrays={'logits':[],'metrics':[],'base_gradients':[],'temporal_gradients':[]};traces=[];observations=[]
for batch in range(50):
 x,y,w=sampler.batch();trace=sampler.last_trace
 speed=float(rng.choice([.9,1.,1.1]));voices=[int(v) for v in rng.choice(sampler.voices,size=2,replace=False)]
 rows=[lookup[v,speed,p,a] for v in voices for p in ('hey_nyx','hey_nick','hey_next','hey_mix') for a in ('quiet','noise25','combined','noise15','rir')]
 legal=[e for e in range(150,351,3) if all(e-12>=r['feature_crop_end'] and e<r['raw_shape'][0] for r in rows)]
 end=int(rng.choice(legal));assert trace['voices']==voices and trace['speed']==speed and trace['endpoint_inclusive']==end
 assert trace['pack_indices']==[r['pack_index'] for r in rows]
 for j,r in enumerate(rows):
  path=sampler.root/r['raw'];assert sha(path)==r['raw_file_sha256']
  raw=np.load(path,allow_pickle=False);assert hashlib.sha256(raw.tobytes()).hexdigest()==r['raw_sha256']
  for k,e in enumerate(range(end-12,end+1,3)):
   window=raw[e-149:e+1];assert hashlib.sha256(window.tobytes()).hexdigest()==trace['window_sha256'][j][k]
   np.testing.assert_array_equal(x[j,k],np.clip(window.astype(np.float32)/25.6,0,26))
  assert y[j]==float(r['target']=='hey_nyx') and w[j]==(1 if y[j] else 20)
 groups=np.asarray(verified_phrase_groups(rows),np.int32)
 logits,metrics,gs,gt=gradients(x,y,w,groups)
 gs=np.concatenate([g.numpy().ravel() for g in gs]);gt=np.concatenate([g.numpy().ravel() for g in gt])
 for key,value in [('logits',logits.numpy()),('metrics',metrics.numpy()),('base_gradients',gs),('temporal_gradients',gt)]:
  assert np.isfinite(value).all();arrays[key].append(value)
 head_count=sum(int(np.prod(v.shape)) for v in head.trainable_variables)
 observation={'batch':batch,'base_norm':float(np.linalg.norm(gs.astype(np.float64))),
 'temporal_encoder_norm':float(np.linalg.norm(gt[:-head_count].astype(np.float64))),
 'temporal_head_norm':float(np.linalg.norm(gt[-head_count:].astype(np.float64)))}
 assert observation['temporal_encoder_norm']>0 and observation['temporal_head_norm']>0
 observations.append(observation);traces.append(trace)
 if batch%10==0:print(json.dumps(observation),flush=True)
for value,variable in zip(before,model.variables):np.testing.assert_array_equal(value,variable.numpy())
outdir.mkdir();np.savez_compressed(outdir/'arrays.npz',**{k:np.stack(v) for k,v in arrays.items()})
report={'snapshot_sha256':sha(snapshot),'arrays_sha256':sha(outdir/'arrays.npz'),'sampler_sha256':sha(Path('research/wakeword_p4/training/paired_temporal_sampler.py')),
 'script_sha256':sha(Path(__file__)),'batches':50,'authenticated_windows':10000,'authenticated_phrase_groups':500,
 'independent_schedule_exact':True,'all_variables_unchanged':True,'optimizer_updates':0,'fixed_coefficient':1,
 'observations':observations,'traces':traces,'limits':'Paired supplementary gradients only; main128/full trainer integration, smoke, export and formal execution not yet checked. Derived initialization explicitly remains derived. No new dev/final or acceptance.'}
(outdir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
Path('research/wakeword_p4/results/4090-temporal-tail-risk-network-preflight-v1.json').write_bytes((outdir/'report.json').read_bytes())
print('completed',sha(outdir/'report.json'),flush=True)
