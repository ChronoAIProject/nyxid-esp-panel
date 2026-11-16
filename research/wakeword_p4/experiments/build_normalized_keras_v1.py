"""Flat normalized cache graph with explicit trained-weight layout."""
import tensorflow as tf
def build_normalized(state,classes=34):
 L=tf.keras.layers;chunk=L.Input(batch_shape=(1,3,1,64),name='chunk');x=L.Lambda(lambda z:(z-tf.reduce_mean(z,axis=-1,keepdims=True))*tf.math.rsqrt(tf.reduce_mean(tf.square(z-tf.reduce_mean(z,axis=-1,keepdims=True)),axis=-1,keepdims=True)+1e-5))(chunk);inputs=[chunk];outputs={}
 for i,h in enumerate([4,8,16,32,64,128]):
  cache=L.Input(batch_shape=(1,h,1,64),name=f'cache{i}');inputs.append(cache);joined=L.Concatenate(axis=1)([cache,x]);outputs[f'cache{i}_out']=L.Lambda(lambda z,h=h:z[:,-h:,:,:])(joined)
  y=L.DepthwiseConv2D((5,1),dilation_rate=(h//4,1),name=f'dw{i}')(joined);y=L.ReLU()(y);y=L.Conv2D(64,1,name=f'pw{i}')(y);x=L.ReLU()(L.Add()([y,x]))
 outputs['logits']=L.Conv2D(classes,1,name='readout')(x);model=tf.keras.Model(inputs,outputs)
 def arr(k):return state[k].detach().cpu().numpy()
 for i in range(6):
  prefix=f'encoder.blocks.{i}.'
  model.get_layer(f'dw{i}').set_weights([arr(prefix+'dw.weight').transpose(2,0,1)[:,None,:,:],arr(prefix+'dw.bias')])
  model.get_layer(f'pw{i}').set_weights([arr(prefix+'pw.weight').transpose(2,1,0)[None,:,:,:],arr(prefix+'pw.bias')])
 model.get_layer('readout').set_weights([arr('readout.weight').transpose(2,1,0)[None,:,:,:],arr('readout.bias')]);return model

