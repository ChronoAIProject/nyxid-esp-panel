"""Unchanged dual raw-BCE supervision plus nominal bounded-logit consistency."""
def components(float_raw,quant_raw,float_bounded,quant_bounded,labels,weights,coefficient,tf):
    for value in (quant_raw,float_bounded,quant_bounded,labels):
        tf.debugging.assert_equal(tf.shape(float_raw),tf.shape(value))
    floating=tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=float_raw),[-1])
    quantized=tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=quant_raw),[-1])
    supervised=tf.reduce_mean((floating+quantized)*tf.cast(.5,floating.dtype)*tf.cast(weights,floating.dtype))
    left=tf.clip_by_value(float_bounded,-8.,8.)
    right=tf.clip_by_value(quant_bounded,-8.,8.)
    gap=(left-right)/tf.cast(8,left.dtype)
    magnitude=tf.abs(gap)
    consistency=tf.reduce_mean(tf.where(magnitude<=1,tf.cast(.5,gap.dtype)*gap*gap,magnitude-tf.cast(.5,gap.dtype)))
    return supervised+tf.cast(coefficient,supervised.dtype)*consistency,supervised,consistency
