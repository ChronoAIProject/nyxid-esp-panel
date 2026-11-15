"""Absolute maximum sustained negative event risk, no inference at import."""
def presentation_loss(z,pw,nw,eligible,peak,tf):
    from causal_fullsequence_graph_v1 import presentation_loss as strong_loss
    # Replace, rather than add to, the original time-mean negative term on
    # exactly eight source-matched lexical negative presentations.
    remaining=tf.where(peak[:,None],tf.zeros_like(nw),nw)
    base=strong_loss(z,pw,remaining,tf)
    # log(1-mean(sigmoid(z))) = logmeanexp(logsigmoid(-z)),
    # evaluated stably without clipping probabilities or changing gradients.
    windows=tf.signal.frame(tf.squeeze(z,axis=-1),5,1,axis=1)
    risk=tf.math.log(tf.cast(5,z.dtype))-tf.reduce_logsumexp(-tf.nn.softplus(windows),axis=-1)
    tf.debugging.assert_equal(tf.reduce_any(eligible,axis=1),peak)
    worst=tf.reduce_max(tf.where(eligible,risk,tf.zeros_like(risk)),axis=1)
    return base+20*tf.reduce_mean(tf.where(peak,worst,tf.zeros_like(worst)))
