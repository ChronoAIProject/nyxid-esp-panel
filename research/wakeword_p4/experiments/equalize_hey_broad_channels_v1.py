"""Deterministic positive power-two exact channel reparameterization, no optimizer."""
import numpy as np

def transform(model, channel_record, tf):
    from microwakeword.layers.stream import Stream
    streams=[l for l in model.layers if isinstance(l,Stream)]
    dw=[l for l in model.layers if isinstance(l,tf.keras.layers.DepthwiseConv2D)]
    pw=[l for l in model.layers if isinstance(l,tf.keras.layers.Conv2D)]
    bn=[l for l in model.layers if isinstance(l,tf.keras.layers.BatchNormalization)]
    dense=[l for l in model.layers if isinstance(l,tf.keras.layers.Dense)]
    assert len(streams)==6 and len(dw)==len(pw)==len(bn)==4 and len(dense)==1
    first=streams[0].cell
    assert isinstance(first,tf.keras.layers.Conv2D)
    records={(r['kind'],r['ordinal']):r for r in channel_record['sites']}
    def factor(kind,index):
        r=records[kind,index];peak=np.asarray(r['channel_peak'],np.float64)
        expected=np.ones_like(peak);nz=peak>0;expected[nz]=np.exp2(-np.ceil(np.log2(peak[nz])))
        assert np.array_equal(expected,np.asarray(r['power_of_two_scale']))
        result=expected.astype(np.float32)
        assert np.isfinite(result).all() and (result>0).all()
        return result
    a=[factor('activation',i) for i in range(5)]
    d=[factor('depthwise',i) for i in range(4)]
    assert [len(s) for s in a]==[30,60,60,60,60]
    assert [len(s) for s in d]==[30,60,60,60]
    kernel,=first.get_weights();assert kernel.shape==(5,1,40,30)
    first.set_weights([kernel*a[0][None,None,None,:]])
    for i in range(4):
        kernel,bias=dw[i].get_weights()
        assert kernel.shape==([5,9,13,21][i],1,len(d[i]),1) and bias.shape==(len(d[i]),)
        dw[i].set_weights([kernel*(d[i]/a[i])[None,None,:,None],bias*d[i]])
        kernel,=pw[i].get_weights();assert kernel.shape==(1,1,len(d[i]),60)
        pw[i].set_weights([kernel/d[i][None,None,:,None]])
        gamma,beta,mean,var=bn[i].get_weights();assert gamma.shape==beta.shape==mean.shape==var.shape==(60,)
        bn[i].set_weights([gamma*a[i+1],beta*a[i+1],mean,var]);bn[i].trainable=False
        _,_,newmean,newvar=bn[i].get_weights()
        assert np.array_equal(mean,newmean) and np.array_equal(var,newvar)
    kernel,bias=dense[0].get_weights();assert kernel.shape==(1800,1)
    dense[0].set_weights([kernel/np.tile(a[-1],30)[:,None],bias])
    assert all(np.isfinite(v.numpy()).all() for v in model.variables)
    return dict(activation_factors=[s.tolist() for s in a],depthwise_factors=[s.tolist() for s in d],
                state_shapes=[[1,2,1,40],[1,4,1,30],[1,8,1,60],[1,12,1,60],[1,20,1,60],[1,29,1,60]],
                batchnorm_moving_statistics_exact=True,optimizer_updates=0)
