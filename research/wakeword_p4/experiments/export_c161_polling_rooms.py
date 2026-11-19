"""Quantize paired candidate heads against identical balanced dry and room training patches."""
import numpy as np
import torch
from ai_edge_litert.interpreter import Interpreter, OpResolverType

import train_c161_polling_rooms as fit
from export_c161_onset_evidence import transfer


def main():
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    torch.set_num_threads(1)
    cfg = fit.read(fit.FIT)
    assert fit.sha(fit.CACHE)==fit.read(fit.SOURCES)["cache_sha256"]
    rng=np.random.default_rng(79183)
    with np.load(fit.CACHE) as z:
        ids=np.concatenate([rng.choice(np.flatnonzero(z["labels"]==k),20,replace=False) for k in range(5)])
        x=np.concatenate([z[k][ids] for k in fit.MODES]).astype(np.float32)[:,:,:,None]/255
    reports = {}
    for mode in fit.MODES:
        weights = fit.weights(mode)
        assert fit.sha(weights)==cfg["models"][mode]["weights_sha256"]
        net=fit.BASE.Small(fit.read(fit.base.FIT)["alphabet"]).eval()
        with np.load(weights) as z:
            net.load_state_dict({k:torch.from_numpy(z[k]) for k in z})
        model=transfer(net)
        with torch.no_grad():
            original=net(torch.from_numpy(x).permute(0,3,1,2))[0].softmax(1).numpy()
        tf_float=np.vstack([model(v[None],training=False).numpy() for v in x])
        np.testing.assert_allclose(tf_float,original,atol=2e-5,rtol=2e-5)
        converter=tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations=[tf.lite.Optimize.DEFAULT]
        converter.representative_dataset=lambda: ([v[None]] for v in x)
        converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type=converter.inference_output_type=tf.int8
        raw=converter.convert()
        vm=Interpreter(model_content=raw,num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        vm.allocate_tensors()
        inp,=vm.get_input_details()
        out,=vm.get_output_details()
        assert inp["dtype"]==out["dtype"]==np.int8 and out["quantization"]==(1/256,-128)
        target=fit.RESULTS/(fit.STEM+"-"+mode+".tflite")
        with target.open("xb") as f:
            f.write(raw)
        scale,zero=inp["quantization"]
        codes=[]
        for v in x:
            vm.set_tensor(inp["index"],np.rint(v[None]/scale+zero).clip(-128,127).astype(np.int8))
            vm.invoke()
            codes.append(vm.get_tensor(out["index"])[0].tolist())
        reports[mode]=dict(file=target.name,sha256=fit.sha(target),bytes=len(raw),calibration_codes=codes,
            input_quantization=[scale,zero],float_transfer_max_error=float(abs(tf_float-original).max()),
            nyx_decision_flips=int(((np.array(codes)[:,0]>=0)!=(original[:,0]>=.5)).sum()))
    fit.save(fit.RESULTS/(fit.STEM+"-exports.json"),dict(models=reports,representative_indices=ids.tolist(),
        fit_sha256=fit.sha(fit.FIT),tensorflow=tf.__version__))


if __name__=="__main__":
    main()
