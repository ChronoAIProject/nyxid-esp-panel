"""Stop float consistency target only; original raw dual-BCE stays connected."""
def components(float_raw,quant_raw,float_bounded,quant_bounded,labels,weights,coefficient,tf):
    from hey_broad_bounded_consistency_loss_v1 import components as original
    return original(float_raw,quant_raw,tf.stop_gradient(float_bounded),quant_bounded,
                    labels,weights,coefficient,tf)
