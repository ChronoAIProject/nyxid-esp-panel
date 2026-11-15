"""Same causal encoder/context; one fixed 16-unit nonlinear binary readout."""
HIDDEN=16

def create(base,tf):
    """Base must already hold the preserved original initializer, never a final fit."""
    assert tuple(base.variables[-2].shape)==(3480,1)
    flat=base.layers[-1].input
    assert int(flat.shape[-1])==3480
    hidden=tf.keras.layers.Dense(HIDDEN,activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1726),bias_initializer='zeros',name='binary_hidden16')(flat)
    output=tf.keras.layers.Dense(1,activation='sigmoid',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1727),bias_initializer='zeros',name='nonlinear_binary_output')(hidden)
    model=tf.keras.Model(base.inputs,output,name='causal_nonlinear16_binary')
    assert len(model.variables)==33
    assert tuple(model.variables[29].shape)==(3480,HIDDEN)
    assert tuple(model.variables[31].shape)==(HIDDEN,1)
    return model

def logits(model,features,tf):
    vs=model.variables
    assert len(vs)==33 and len(model.trainable_variables)==17
    assert tuple(vs[29].shape)==(3480,HIDDEN) and tuple(vs[30].shape)==(HIDDEN,)
    bn=[layer for layer in model.layers if isinstance(layer,tf.keras.layers.BatchNormalization)]
    assert len(bn)==4 and all(not layer.trainable for layer in bn)
    x=tf.expand_dims(features,2)
    x=tf.nn.relu(tf.nn.conv2d(tf.pad(x,[[0,0],[2,0],[0,0],[0,0]]),vs[0],strides=[1,3,1,1],padding='VALID'))
    for offset,kernel,normalization in zip([1,8,15,22],[5,9,13,21],bn,strict=True):
        assert tuple(vs[offset].shape)==(kernel,1,int(x.shape[-1]),1)
        x=tf.nn.depthwise_conv2d(tf.pad(x,[[0,0],[kernel-1,0],[0,0],[0,0]]),vs[offset],strides=[1,1,1,1],padding='VALID')+vs[offset+1]
        x=tf.nn.conv2d(x,vs[offset+2],strides=[1,1,1,1],padding='VALID')
        gamma,beta,mean,variance=vs[offset+3:offset+7]
        x=tf.nn.relu(tf.nn.batch_normalization(x,mean,variance,beta,gamma,normalization.epsilon))
    x=tf.squeeze(x,axis=2)
    encoded=x
    vs=model.variables
    history=tf.pad(encoded,[[0,0],[57,0],[0,0]])
    hidden=tf.nn.relu(tf.nn.conv1d(history,tf.reshape(vs[29],[58,60,HIDDEN]),stride=1,padding='VALID')+vs[30])
    return tf.nn.conv1d(hidden,tf.reshape(vs[31],[1,HIDDEN,1]),stride=1,padding='VALID')+vs[32]

def loss(model,x,pw,nw,tf):
    from causal_fullsequence_graph_v1 import presentation_loss
    return presentation_loss(logits(model,x,tf),pw,nw,tf)
