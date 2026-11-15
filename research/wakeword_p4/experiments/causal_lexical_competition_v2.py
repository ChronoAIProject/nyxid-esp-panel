"""Explicit six-class causal graph; source identity remains valid under tf.function."""
from causal_lexical_competition_v1 import create,CLASSES

def logits(model,features,tf):
    vs=model.variables
    assert len(vs)==31 and len(model.trainable_variables)==15
    assert tuple(vs[29].shape)==(3480,6) and tuple(vs[30].shape)==(6,)
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
    x=tf.pad(x,[[0,0],[57,0],[0,0]])
    # Dense flatten ordering is time then channel; share the original 3480x1 kernel.
    return tf.nn.conv1d(x,tf.reshape(vs[29],[58,60,6]),stride=1,padding='VALID')+vs[30]

def presentation_loss(z,positive_weights,negative_weights,labels,tf):
    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=z)
    return tf.reduce_mean(tf.reduce_sum(loss*(positive_weights+20*negative_weights),axis=1))
