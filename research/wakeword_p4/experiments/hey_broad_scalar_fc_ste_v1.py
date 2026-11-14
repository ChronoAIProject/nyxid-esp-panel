"""Scalar FC reference round-away forward with explicitly declared surrogate."""
def scalar_fc_ste(accumulator,multiplier,zero,low,high,output_scale,tf):
    if not (multiplier>0 and output_scale>0 and low<=high):
        raise ValueError('Invalid scalar FC grid')
    @tf.custom_gradient
    def apply(value):
        tf.debugging.assert_type(value,tf.float64)
        tf.debugging.assert_all_finite(value,'Nonfinite accumulator')
        tf.debugging.assert_equal(value,tf.round(value))
        tf.debugging.assert_greater_equal(value,tf.constant(-(1<<31),tf.float64))
        tf.debugging.assert_less(value,tf.constant(1<<31,tf.float64))
        real=value*tf.constant(multiplier,tf.float64)
        rounded=tf.sign(real)*tf.floor(tf.abs(real)+tf.constant(.5,tf.float64))
        code=tf.clip_by_value(rounded+zero,low,high)
        result=(code-zero)*tf.constant(output_scale,tf.float64)
        def gradient(dy):
            # Continuous real-code interval, inclusive; not integer derivative.
            active=(real+zero>=low)&(real+zero<=high)
            return dy*tf.constant(multiplier*output_scale,tf.float64)*tf.cast(active,tf.float64)
        return result,gradient
    return apply(accumulator)
