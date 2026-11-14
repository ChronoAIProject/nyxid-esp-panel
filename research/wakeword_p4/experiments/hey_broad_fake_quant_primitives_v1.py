"""Explicit affine nearest-even fake quantizer with declared clipped STE."""
def affine_ste(value,scale,zero,qmin,qmax,tf,wide=False):
    """Fixed/stop-gradient scale. Closed representable interval has unit STE."""
    scale=tf.stop_gradient(tf.convert_to_tensor(scale,dtype=value.dtype))
    zero=tf.stop_gradient(tf.convert_to_tensor(zero,dtype=value.dtype))
    tf.debugging.assert_positive(scale)
    @tf.custom_gradient
    def quantize(x):
        calc=tf.float64 if wide else x.dtype
        s=tf.cast(scale,calc);z=tf.cast(zero,calc);v=tf.cast(x,calc)
        low=tf.cast(qmin,calc);high=tf.cast(qmax,calc)
        code=tf.clip_by_value(tf.math.rint(v/s)+z,low,high)
        result=tf.cast((code-z)*s,x.dtype)
        minimum=(low-z)*s;maximum=(high-z)*s
        mask=tf.cast(tf.logical_and(v>=minimum,v<=maximum),x.dtype)
        def gradient(upstream):return upstream*mask
        return result,gradient
    return quantize(value)

def weight_scale(value,tf):
    """Per-last-output-channel symmetric scale; finite zero-channel guard."""
    axes=list(range(len(value.shape)-1))
    peak=tf.reduce_max(tf.abs(value),axis=axes)
    return tf.stop_gradient(tf.maximum(peak,tf.constant(1e-8,value.dtype))/tf.constant(127,value.dtype))
