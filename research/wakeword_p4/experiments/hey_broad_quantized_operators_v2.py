"""Fixed actual-export quantization grids on shared offline training variables."""
import numpy as np

def configuration(arm):
    from ai_edge_litert.interpreter import Interpreter,OpResolverType
    i=Interpreter(model_path=arm['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1)
    i.allocate_tensors() # metadata only
    details={d['index']:d for d in i.get_tensor_details()}
    def grid(index):
        d=details[index];p=d['quantization_parameters']
        assert np.all(p['scales']>0)
        return dict(index=index,scales=p['scales'].tolist(),zeros=p['zero_points'].tolist(),axis=int(p['quantized_dimension']),shape=d['shape'].tolist(),dtype=str(d['dtype']))
    operations=[o for o in arm['ops'] if o['op'] in ['CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED']]
    assert [o['op'] for o in operations]==['CONV_2D']+['DEPTHWISE_CONV_2D','CONV_2D']*4+['FULLY_CONNECTED']
    ops=[dict(kind=o['op'],input=grid(o['inputs'][0]),weight=grid(o['inputs'][1]),bias=grid(o['inputs'][2]),output=grid(o['outputs'][0])) for o in operations]
    # Quantized histories must share the consuming concatenation grid.
    read_outputs={int(o['outputs'][0]) for o in arm['ops'] if o['op']=='READ_VARIABLE'}
    concats={int(o['outputs'][0]):o for o in arm['ops'] if o['op']=='CONCATENATION'}
    pairs=[]
    for producer,concat in [(47,48),(50,51),(53,54),(56,57),(59,60),(62,63)]:
        op=concats[concat]
        reads=[int(n) for n in op['inputs'] if int(n) in read_outputs]
        assert len(reads)==1 and producer in op['inputs']
        read=reads[0];p,r,c=grid(producer),grid(read),grid(concat)
        assert p['scales']==r['scales']==c['scales'] and p['zeros']==r['zeros']==c['zeros']
        pairs.append([producer,read,concat])
    flatten,=[o for o in arm['ops'] if o['op']=='RESHAPE']
    p,c=grid(int(flatten['inputs'][0])),grid(int(flatten['outputs'][0]))
    assert p['scales']==c['scales'] and p['zeros']==c['zeros']

    return dict(input=grid(0),operators=ops,post_head=[grid(n) for n in [65,66,67,74]],state_grid_pairs=pairs)

def operators(model,config,tf):
    from microwakeword.layers.stream import Stream
    from hey_broad_fake_quant_primitives_v1 import affine_ste
    streams=[l for l in model.layers if isinstance(l,Stream)]
    dw=[l for l in model.layers if isinstance(l,tf.keras.layers.DepthwiseConv2D)]
    pw=[l for l in model.layers if isinstance(l,tf.keras.layers.Conv2D)]
    bn=[l for l in model.layers if isinstance(l,tf.keras.layers.BatchNormalization)]
    head,=[l for l in model.layers if isinstance(l,tf.keras.layers.Dense)]
    assert len(streams)==6 and len(dw)==len(pw)==len(bn)==4
    assert all(not l.trainable for l in bn)
    def q(value,grid,weight=False,depth=False,bias=False):
        scales=np.asarray(grid['scales'],np.float32);zeros=np.asarray(grid['zeros'],np.float32)
        if len(scales)>1 and not bias:
            shape=[1]*len(value.shape);shape[-2 if depth else -1]=len(scales)
            scales=scales.reshape(shape);zeros=zeros.reshape(shape)
        if not weight and not bias:assert len(grid['scales'])==1
        low,high=(-2147483648,2147483647) if bias else (-127,127) if weight else (0,255) if grid['dtype']=="<class 'numpy.uint8'>" else (-128,127)
        return affine_ste(value,scales,zeros,low,high,tf,wide=bias)
    first=streams[0].cell
    def infer(value):
        cfg=config['operators'];x=tf.expand_dims(q(value,config['input']),2)
        kernel=q(first.kernel,cfg[0]['weight'],weight=True)
        bias=q(tf.zeros([30],tf.float32),cfg[0]['bias'],bias=True)
        x=q(tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,kernel,[1,3,1,1],'VALID'),bias)),cfg[0]['output'])
        for j,(depth,point,norm) in enumerate(zip(dw,pw,bn,strict=True)):
            a,b=cfg[1+2*j:3+2*j]
            kernel=q(depth.kernel,a['weight'],weight=True,depth=True);bias=q(depth.bias,a['bias'],bias=True)
            x=q(tf.nn.bias_add(tf.nn.depthwise_conv2d(x,kernel,[1,1,1,1],'VALID'),bias),a['output'])
            scale=norm.gamma*tf.math.rsqrt(norm.moving_variance+norm.epsilon)
            kernel=q(point.kernel*scale[None,None,None,:],b['weight'],weight=True)
            bias=q(norm.beta-norm.moving_mean*scale,b['bias'],bias=True)
            x=q(tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,kernel,[1,1,1,1],'VALID'),bias)),b['output'])
        tf.debugging.assert_equal(tf.shape(x)[1:],[30,1,60])
        embedding=tf.reshape(x,[tf.shape(x)[0],1800])
        cfghead=cfg[-1]
        # Map exactly to deployed bounded FC weight/bias coordinates.
        kernel=q(head.kernel*tf.constant(3/8,tf.float32),cfghead['weight'],weight=True)
        bias=q(head.bias*tf.constant(3/8,tf.float32)+3,cfghead['bias'],bias=True)
        u=tf.linalg.matmul(embedding,kernel)+bias
        # BCE surrogate retains gradients for confident-wrong examples.
        rawlogit=(u-3)*tf.constant(8/3,tf.float32)
        h=config['post_head'];v=q(tf.clip_by_value(u,0,6),h[0]);v=q(v*tf.constant(8/3,tf.float32),h[1]);v=q(v-8,h[2])
        probability=q(tf.math.sigmoid(v),h[3])
        return rawlogit,probability
    return infer
