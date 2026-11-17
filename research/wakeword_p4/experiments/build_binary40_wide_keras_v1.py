"""Explicit128channel Keras graph; original export graph remains immutable."""
def build(state):
    import tensorflow as tf
    L=tf.keras.layers
    chunk=L.Input(batch_shape=(1,3,1,40),name='chunk')
    histories=[2,4,8,16,32,40,1];channels=[40,128,128,128,128,128,128]
    caches=[L.Input(batch_shape=(1,h,1,c),name=f'cache{i}') for i,(h,c) in enumerate(zip(histories,channels))]
    outputs={}
    def join(i,x):
        y=L.Concatenate(axis=1)([caches[i],x]);h=histories[i]
        outputs[f'cache{i}_out']=L.Lambda(lambda a,h=h:a[:,-h:,:,:])(y)
        return y
    x=L.ReLU()(L.Conv2D(128,(3,1),strides=(3,1),name='stem')(join(0,chunk)))
    for i,d in enumerate([1,2,4,8,10]):
        y=L.ReLU()(L.DepthwiseConv2D((5,1),dilation_rate=(d,1),name=f'dw{i}')(join(i+1,x)))
        y=L.Conv2D(128,1,name=f'pw{i}')(y);x=L.ReLU()(L.Add()([x,y]))
    outputs['probability']=L.Activation('sigmoid')(L.Conv2D(1,(2,1),name='head')(join(6,x)))
    m=tf.keras.Model([chunk,*caches],outputs)
    def arr(k):return state[k].detach().cpu().numpy()
    for layer,key in [('stem','stem'),('head','head'),*[(f'pw{i}',f'pw.{i}') for i in range(5)]]:
        m.get_layer(layer).set_weights([arr(key+'.weight').transpose(2,1,0)[:,None,:,:],arr(key+'.bias')])
    for i in range(5):m.get_layer(f'dw{i}').set_weights([arr(f'dw.{i}.weight').transpose(2,0,1)[:,None,:,:],arr(f'dw.{i}.bias')])
    return m,histories,channels

