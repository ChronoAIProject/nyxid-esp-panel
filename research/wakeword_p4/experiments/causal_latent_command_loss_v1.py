"""Weak full-command event support; no model/data calls at import."""
def presentation_loss(z,pw,nw,eligible,weak,tf):
    # Original strong rows retain their region-normalized BCE. Weak command
    # rows have zero positive-region weights; startup negatives remain intact.
    from causal_fullsequence_graph_v1 import presentation_loss as strong_loss
    base=strong_loss(z,pw,nw,tf)
    probabilities=tf.sigmoid(tf.squeeze(z,axis=-1))
    means=tf.nn.avg_pool1d(probabilities[...,None],ksize=5,strides=1,padding='VALID')[...,0]
    valid=tf.cast(eligible,tf.bool)
    tf.debugging.assert_equal(tf.reduce_any(valid,axis=1),tf.cast(weak,tf.bool))
    best=tf.reduce_max(tf.where(valid,means,tf.zeros_like(means)),axis=1)
    # Non-weak rows contribute exactly zero; guard log(0) outside selected bags.
    safe=tf.where(tf.cast(weak,tf.bool),best,tf.ones_like(best))
    bag=-tf.math.log(tf.maximum(safe,tf.constant(1e-7,dtype=z.dtype)))
    return base+tf.reduce_mean(bag)
