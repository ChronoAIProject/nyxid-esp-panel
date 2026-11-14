"""Shared float/quant supervision plus normalized logit Huber consistency."""
def components(float_logits,quant_logits,labels,weights,coefficient,tf):
    tf.debugging.assert_equal(tf.shape(float_logits),tf.shape(quant_logits))
    tf.debugging.assert_equal(tf.shape(float_logits),tf.shape(labels))
    floating=tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=float_logits),[-1])
    quantized=tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=quant_logits),[-1])
    supervised=tf.reduce_mean((floating+quantized)*tf.cast(.5,floating.dtype)*tf.cast(weights,floating.dtype))
    gap=(float_logits-quant_logits)/tf.cast(8,float_logits.dtype)
    magnitude=tf.abs(gap)
    consistency=tf.reduce_mean(tf.where(magnitude<=1,tf.cast(.5,gap.dtype)*gap*gap,magnitude-tf.cast(.5,gap.dtype)))
    return supervised+tf.cast(coefficient,supervised.dtype)*consistency,supervised,consistency
