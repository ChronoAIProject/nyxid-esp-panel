"""Shared exact update mechanics, explicit S/D backwards and encoder-only reversal."""
import hashlib
import numpy as np
import hey_broad_gpu_training_factory_v1 as factory
tf=factory.tf
def sig(vs):return hashlib.sha256(b''.join(v.numpy().tobytes() for v in vs)).hexdigest()
def make(initial,seed):
 m=factory.create(225);m.load_weights(initial);head=m.layers[-1]
 for layer in m.layers:
  cell=getattr(layer,'cell',layer)
  if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
 vs=list(m.trainable_variables);assert len(vs)==15;wake=[i for i,v in enumerate(vs) if any(v is w for w in head.trainable_variables)];assert len(wake)==2
 with tf.device('/GPU:0'):
  kernel=tf.Variable(tf.keras.initializers.GlorotUniform(seed=seed)((int(head.kernel.shape[0]),3),dtype=tf.float32),name='engine_kernel');bias=tf.Variable(tf.zeros([3],tf.float32),name='engine_bias')
 return dict(model=m,head=head,embedding=tf.keras.Model(m.inputs,head.input),variables=vs+[kernel,bias],wake=wake,encoder=[i for i in range(15) if i not in wake],domain=[15,16],state=list(m.variables)+[kernel,bias],bn=m.non_trainable_variables)
def components(b,x,y,w,ax,ey):
 with tf.GradientTape(persistent=True) as tape:
  z=tf.linalg.matmul(b['embedding'](tf.constant(x),training=False),b['head'].kernel)+b['head'].bias
  S=tf.reduce_mean(tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant(y),logits=z),[-1])*tf.constant(w))
  dz=tf.linalg.matmul(b['embedding'](tf.constant(ax),training=False),b['variables'][15])+b['variables'][16]
  D=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ey,logits=dz))
 gs=tape.gradient(S,b['variables']);gd=tape.gradient(D,b['variables']);del tape
 assert [i for i,g in enumerate(gs) if g is None]==[15,16] and [i for i,g in enumerate(gd) if g is None]==b['wake']
 for g in gs+gd:
  if g is not None:tf.debugging.assert_all_finite(g,'component gradient')
 tf.debugging.assert_all_finite(S,'S');tf.debugging.assert_all_finite(D,'D')
 return S,D,z,dz,gs,gd
def merge(b,gs,gd,coefficient):
 assert coefficient in (0.,1.)
 merged=[]
 for i in range(17):
  g=gd[i] if i in b['domain'] else gs[i] if i in b['wake'] or coefficient==0. else gs[i]-gd[i]
  tf.debugging.assert_all_finite(g,'merged gradient');merged.append(g)
 return merged
