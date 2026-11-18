"""Explicit-cache local frequency stem transfer with exact Torch flatten ordering."""
def build(state,dilations):
    import tensorflow as tf
    L=tf.keras.layers
    chunk=L.Input(batch_shape=(1,3,1,40),name='chunk')
    histories=[2,*[4*d for d in dilations],1];channels=[40,64,64,64,64,64,64]
    caches=[L.Input(batch_shape=(1,h,1,c),name=f'cache{i}') for i,(h,c) in enumerate(zip(histories,channels))]
    outputs={}
    def join(i,x):
        y=L.Concatenate(axis=1)([caches[i],x]);h=histories[i]
        outputs[f'cache{i}_out']=L.Lambda(lambda a,h=h:a[:,-h:,:,:])(y)
        return y
    x=L.Permute((1,3,2))(join(0,chunk))
    x=L.ZeroPadding2D(((0,0),(2,2)))(x)
    x=L.ReLU()(L.Conv2D(8,(3,5),strides=(3,2),name='stem_local')(x))
    x=L.Permute((1,3,2))(x)
    x=L.Reshape((1,1,160))(x)
    x=L.ReLU()(L.Conv2D(64,1,name='stem_project')(x))
    for i,d in enumerate(dilations):
        y=L.ReLU()(L.DepthwiseConv2D((5,1),dilation_rate=(d,1),name=f'dw{i}')(join(i+1,x)))
        y=L.Conv2D(64,1,name=f'pw{i}')(y);x=L.ReLU()(L.Add()([x,y]))
    outputs['probability']=L.Activation('sigmoid')(L.Conv2D(1,(2,1),name='head')(join(6,x)))
    m=tf.keras.Model([chunk,*caches],outputs)
    def arr(k):return state[k].detach().cpu().numpy()
    for layer,key in [('stem_project','stem.project'),('head','head'),*[(f'pw{i}',f'pw.{i}') for i in range(5)]]:
        m.get_layer(layer).set_weights([arr(key+'.weight').transpose(2,1,0)[:,None,:,:],arr(key+'.bias')])
    m.get_layer('stem_local').set_weights([arr('stem.local.weight').transpose(2,3,1,0),arr('stem.local.bias')])
    for i in range(5):m.get_layer(f'dw{i}').set_weights([arr(f'dw.{i}.weight').transpose(2,0,1)[:,None,:,:],arr(f'dw.{i}.bias')])
    return m,histories,channels
