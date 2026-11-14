"""Float functional graph sharing admitted teacher60 variables, no fake quant yet."""
def operators(model, tf):
    from microwakeword.layers.stream import Stream
    streams=[l for l in model.layers if isinstance(l,Stream)]
    dw=[l for l in model.layers if isinstance(l,tf.keras.layers.DepthwiseConv2D)]
    pw=[l for l in model.layers if isinstance(l,tf.keras.layers.Conv2D)]
    bn=[l for l in model.layers if isinstance(l,tf.keras.layers.BatchNormalization)]
    dense=[l for l in model.layers if isinstance(l,tf.keras.layers.Dense)]
    assert len(streams)==6 and len(dw)==len(pw)==len(bn)==4 and len(dense)==1
    first=streams[0].cell;head=dense[0]
    assert tuple(first.kernel.shape)==(5,1,40,30) and first.bias is None
    assert tuple(head.kernel.shape)==(1800,1)
    for i in range(4):
        assert tuple(dw[i].kernel.shape)==([5,9,13,21][i],1,30 if i==0 else 60,1)
        assert pw[i].bias is None and not bn[i].trainable
    def infer(value):
        x=tf.expand_dims(value,2)
        x=tf.nn.relu(tf.nn.conv2d(x,first.kernel,strides=[1,3,1,1],padding='VALID'))
        for depth,point,norm in zip(dw,pw,bn,strict=True):
            x=tf.nn.depthwise_conv2d(x,depth.kernel,strides=[1,1,1,1],padding='VALID')
            x=tf.nn.bias_add(x,depth.bias)
            scale=norm.gamma*tf.math.rsqrt(norm.moving_variance+norm.epsilon)
            kernel=point.kernel*scale[None,None,None,:]
            bias=norm.beta-norm.moving_mean*scale
            x=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,kernel,strides=[1,1,1,1],padding='VALID'),bias))
        tf.debugging.assert_equal(tf.shape(x)[1:],[30,1,60])
        embedding=tf.reshape(x,[tf.shape(x)[0],1800])
        return tf.linalg.matmul(embedding,head.kernel)+head.bias
    return infer
