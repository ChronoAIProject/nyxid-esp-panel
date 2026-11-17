"""Explicit-cache Keras transfer for the fixed matched dilation architectures."""
def build(state,dilations):
    import tensorflow as tf
    L=tf.keras.layers
    chunk=L.Input(batch_shape=(1,3,1,40),name='chunk')
    histories=[2,*[4*d for d in dilations],1];channels=[40,64,64,64,64,64,384]
    caches=[L.Input(batch_shape=(1,h,1,c),name=f'cache{i}') for i,(h,c) in enumerate(zip(histories,channels))]
    outputs={}
    def join(i,x):
        y=L.Concatenate(axis=1)([caches[i],x]);h=histories[i]
        outputs[f'cache{i}_out']=L.Lambda(lambda a,h=h:a[:,-h:,:,:])(y)
        return y
    x=L.ReLU()(L.Conv2D(64,(3,1),strides=(3,1),name='stem')(join(0,chunk)))
    scales=[x]
    for i,d in enumerate(dilations):
        y=L.ReLU()(L.DepthwiseConv2D((5,1),dilation_rate=(d,1),name=f'dw{i}')(join(i+1,x)))
        y=L.Conv2D(64,1,name=f'pw{i}')(y);x=L.ReLU()(L.Add()([x,y]));scales.append(x)
    outputs['probability']=L.Activation('sigmoid')(L.Conv2D(1,(2,1),name='head')(join(6,L.Concatenate(axis=3)(scales))))
    m=tf.keras.Model([chunk,*caches],outputs)
    def arr(k):return state[k].detach().cpu().numpy()
    for layer,key in [('stem','stem'),('head','head'),*[(f'pw{i}',f'pw.{i}') for i in range(5)]]:
        m.get_layer(layer).set_weights([arr(key+'.weight').transpose(2,1,0)[:,None,:,:],arr(key+'.bias')])
    for i in range(5):m.get_layer(f'dw{i}').set_weights([arr(f'dw.{i}.weight').transpose(2,0,1)[:,None,:,:],arr(f'dw.{i}.bias')])
    return m,histories,channels
