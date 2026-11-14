"""Canonical explicit component-gradient assembly, both backwards always observed."""
def merge(supervised,consistency,coefficient,tf):
    assert coefficient in (0,1) and len(supervised)==len(consistency)
    result=[]
    for a,b in zip(supervised,consistency,strict=True):
        assert a is not None and b is not None
        tf.debugging.assert_equal(tf.shape(a),tf.shape(b))
        tf.debugging.assert_all_finite(a,'supervised gradient')
        tf.debugging.assert_all_finite(b,'consistency gradient')
        value=a+tf.cast(coefficient,a.dtype)*b
        tf.debugging.assert_all_finite(value,'merged gradient')
        result.append(value)
    return result
