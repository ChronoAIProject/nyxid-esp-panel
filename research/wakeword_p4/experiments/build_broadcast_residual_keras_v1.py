"""Explicit seven-cache transfer of the fixed causal broadcast residual prototype."""
def build(state, dilations):
    import tensorflow as tf
    assert tuple(dilations)==(1,2,4,8,10)
    L=tf.keras.layers
    chunk=L.Input(batch_shape=(1,3,1,40),name='chunk')
    histories=[2,*[4*d for d in dilations],1]
    channels=[40,*([32]*6)]
    caches=[L.Input(batch_shape=(1,h,1,c),name=f'cache{i}')
            for i,(h,c) in enumerate(zip(histories,channels))]
    outputs={}
    def join(i,x):
        y=L.Concatenate(axis=1)([caches[i],x]);h=histories[i]
        outputs[f'cache{i}_out']=L.Lambda(lambda a,h=h:a[:,-h:,:,:])(y)
        return y
    def frequency_mean(x):
        return L.Lambda(lambda a:tf.reduce_mean(a,axis=2,keepdims=True))(x)
    x=L.Permute((1,3,2))(join(0,chunk))
    x=L.ZeroPadding2D(((0,0),(2,2)))(x)
    x=L.ReLU()(L.Conv2D(32,(3,5),strides=(3,2),name='stem')(x))
    for i,d in enumerate(dilations):
        spectral=L.ReLU()(L.DepthwiseConv2D((1,3),padding='same',name=f'spectral{i}')(x))
        pooled=frequency_mean(spectral)
        temporal=L.ReLU()(L.DepthwiseConv2D((5,1),dilation_rate=(d,1),name=f'temporal{i}')(join(i+1,pooled)))
        broadcast=L.Conv2D(32,1,name=f'pointwise{i}')(temporal)
        x=L.ReLU()(L.Add()([L.Add()([x,spectral]),broadcast]))
    outputs['probability']=L.Activation('sigmoid')(L.Conv2D(1,(2,1),name='head')(join(6,frequency_mean(x))))
    model=tf.keras.Model([chunk,*caches],outputs)
    def arr(k): return state[k].detach().cpu().numpy()
    model.get_layer('stem').set_weights([arr('stem.weight').transpose(2,3,1,0),arr('stem.bias')])
    for layer,key in [('head','head'),*[(f'pointwise{i}',f'pointwise.{i}') for i in range(5)]]:
        model.get_layer(layer).set_weights([arr(key+'.weight').transpose(2,1,0)[:,None,:,:],arr(key+'.bias')])
    for i in range(5):
        model.get_layer(f'spectral{i}').set_weights([arr(f'spectral.{i}.weight').transpose(2,3,0,1),arr(f'spectral.{i}.bias')])
        model.get_layer(f'temporal{i}').set_weights([arr(f'temporal.{i}.weight').transpose(2,0,1)[:,None,:,:],arr(f'temporal.{i}.bias')])
    return model,histories,channels
