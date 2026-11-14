"""One independent fresh backend/precision process, original snapshot, zero updates."""
import argparse,ast,hashlib,json,os,sys,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-engine-gradient-precision-execution-v1.json';OUT=Path('.local/wakeword/engine-gradient-precision-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 mode=sys.argv[1];assert mode in ['gpu-default','cpu-default','gpu-off','cpu-off'];d=json.loads(LOCK.read_text());folder=OUT/mode;folder.mkdir(exist_ok=False)
 try:
  import tensorflow as tf
  physical=tf.config.list_physical_devices('GPU');before=tf.config.experimental.tensor_float_32_execution_enabled()
  if mode.startswith('cpu'):tf.config.set_visible_devices([],'GPU');device='/CPU:0'
  else:
   assert len(physical)==1 and '4090' in tf.config.experimental.get_device_details(physical[0]).get('device_name','');tf.config.experimental.set_memory_growth(physical[0],True);device='/GPU:0'
  if mode.endswith('off'):tf.config.experimental.enable_tensor_float_32_execution(False)
  after=tf.config.experimental.tensor_float_32_execution_enabled();assert not mode.endswith('off') or after is False
  tf.config.threading.set_intra_op_parallelism_threads(2);tf.keras.utils.set_random_seed(20261114);tf.config.experimental.enable_op_determinism()
  sys.path.insert(0,str(Path('.local/wakeword/microWakeWord').resolve()));from microwakeword import mixednet
  architecture=Path('research/wakeword_p4/experiments/probe_hey_context_transfer_shapes_v1.py');node=next(n for n in ast.parse(architecture.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ARCH' for t in n.targets));arch=ast.literal_eval(node.value);probe=json.loads(Path(d['original_probe']).read_text());aux=json.loads(Path(d['original_auxiliary']).read_text())['batches'][0];cases=[]
  for target in ['hey_nyx','hey_willow']:
   original=next(t for t in probe['targets'] if t['target']==target);first=original['cases'][0];assert sha(first['arrays_path'])==first['arrays_sha256'];assert sha(original['initialization_path'])==original['initialization_sha256'];init=json.loads(Path(original['initialization_path']).read_text());assert sha(init['initial_h5_path'])==init['initial_h5_sha256'] and sha(init['state_path'])==init['state_sha256']
   with np.load(first['arrays_path'],allow_pickle=False) as a:main_x=a['main_x'].copy();y=a['main_y'].copy();w=a['main_weights'].copy();old={k:a[k].copy() for k in a.files if k.startswith(('S_','D_')) or k in ['main_logits','domain_logits']};old['losses']=a['losses'][:2].copy()
   with np.load(aux['arrays_path'],allow_pickle=False) as a:ax=a['x'].copy();ey=a['engine'].copy()
   with np.load(init['state_path'],allow_pickle=False) as a:saved_state=[a[f'v{i}'].copy() for i in range(len(a.files))]
   with tf.device(device):
    parser=argparse.ArgumentParser();mixednet.model_parameters(parser);m=mixednet.model(parser.parse_args(arch),(225,40),None);m.load_weights(init['initial_h5_path']);head=m.layers[-1]
    for layer in m.layers:
     cell=getattr(layer,'cell',layer)
     if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
    assert len(m.variables)==31 and all(np.array_equal(v.numpy(),a) for v,a in zip(m.variables,saved_state[:31],strict=True));kernel=tf.Variable(saved_state[31],name='engine_kernel');bias=tf.Variable(saved_state[32],name='engine_bias');vs=list(m.trainable_variables)+[kernel,bias];state=list(m.variables)+[kernel,bias];assert len(vs)==17;assert all(np.array_equal(v.numpy(),a) for v,a in zip(state,saved_state,strict=True));emb=tf.keras.Model(m.inputs,head.input);wake=[i for i,v in enumerate(vs) if any(v is h for h in head.trainable_variables)];payload=None
    for repeat in range(2):
     with tf.GradientTape(persistent=True) as tape:
      main_features=emb(tf.constant(main_x),training=False);z=tf.linalg.matmul(main_features,head.kernel)+head.bias;S=tf.reduce_mean(tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant(y),logits=z),[-1])*tf.constant(w));domain_features=emb(tf.constant(ax),training=False);dz=tf.linalg.matmul(domain_features,kernel)+bias;D=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ey,logits=dz))
     gs=tape.gradient(S,vs);gd=tape.gradient(D,vs);del tape
     none=[[i for i,g in enumerate(gs) if g is None],[i for i,g in enumerate(gd) if g is None]];assert none==[[15,16],wake]
     assert '/device:'+device[1:] in z.device and '/device:'+device[1:] in dz.device
     current=dict(main_features=main_features.numpy(),domain_features=domain_features.numpy(),main_logits=z.numpy(),domain_logits=dz.numpy(),losses=np.array([S,D],np.float32))
     for tag,gradients in [('S',gs),('D',gd)]:
      for i,g in enumerate(gradients):
       if g is not None:current[f'{tag}_{i}']=g.numpy()
     assert all(np.isfinite(a).all() for a in current.values()) and all(np.array_equal(v.numpy(),a) for v,a in zip(state,saved_state,strict=True))
     if payload is None:payload=current
     else:assert all(np.array_equal(payload[k],current[k]) for k in payload)
   p=folder/(target+'.npz');np.savez_compressed(p,**payload);baseline_exact=all(np.array_equal(payload[k],v) for k,v in old.items());cases.append(dict(target=target,arrays_path=str(p),arrays_sha256=sha(p),repeat_exact=True,state_unchanged=True,snapshot_all33_arrays_exact=True,none_indices=none,original_gpu_common_arrays_exact=baseline_exact,original_arrays_path=first['arrays_path'],original_arrays_sha256=first['arrays_sha256'],input_main_sha256=first['main_tensor_sha256'],input_aux_sha256=aux['arrays_sha256'],actual_logits_device=z.device));print(json.dumps(dict(mode=mode,target=target,original_gpu_common_arrays_exact=baseline_exact)),flush=True)
  settings=dict(mode=mode,requested_device=device,tf32_before=before,tf32_after=after,tf_enable_onednn_opts_env=os.environ.get('TF_ENABLE_ONEDNN_OPTS'),intraop=tf.config.threading.get_intra_op_parallelism_threads(),interop=tf.config.threading.get_inter_op_parallelism_threads(),physical_gpu_names=[x.name for x in physical],visible_gpu_names=[x.name for x in tf.config.get_visible_devices('GPU')],build_info=tf.sysconfig.get_build_info(),eager=tf.executing_eagerly(),optimizer_created=False)
  (folder/'report.json').write_text(json.dumps(dict(settings=settings,cases=cases,all_mechanics_passed=True,limits='Only fixed original batch1/two words; no updates or original precision changes. Cross-mode numeric status reduced independently.'),indent=2)+'\n')
 except Exception:
  (folder/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
