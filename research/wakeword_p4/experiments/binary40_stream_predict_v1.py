"""Explicit-cache Binary40 adapter for existing complete-stream replay protocols.

Caller retains original frontend, priming, finish handling and event clocks.
Fresh state only at complete-stream start; never reset at prime/real boundary.
"""
import hashlib
import numpy as np


def predict(values, model_path, interpreter):
    if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != 40 or len(values)%3:
        raise ValueError('Expected original prepared float32 [3*N,40] features')
    it = interpreter(model_path=model_path)
    it.allocate_tensors()
    runner = it.get_signature_runner()
    ins, outs = runner.get_input_details(), runner.get_output_details()
    assert set(ins)=={'chunk',*[f'cache{i}' for i in range(7)]}
    assert set(outs)=={'probability',*[f'cache{i}_out' for i in range(7)]}
    assert list(ins['chunk']['shape']) == [1,3,1,40]
    assert list(outs['probability']['shape']) == [1,1,1,1]
    for d in [*ins.values(),*outs.values()]:
        assert d['dtype']==np.int8 and d['quantization'][0]>0
    scale, zero = ins['chunk']['quantization']
    assert outs['probability']['quantization'] == (1/256,-128)
    q = np.clip(np.rint(values/scale)+zero,-128,127).astype(np.int8)
    states={f'cache{i}':np.full(ins[f'cache{i}']['shape'],ins[f'cache{i}']['quantization'][1],np.int8) for i in range(7)}
    codes=np.empty(len(values)//3,np.uint8)
    native=np.empty(len(codes),np.int8)
    for call,start in enumerate(range(0,len(values),3)):
        out=runner(chunk=q[None,start:start+3,None,:],**states)
        signed=int(out['probability'][0,0,0,0])
        native[call]=signed
        codes[call]=signed+128
        for i in range(7):
            s,z=outs[f'cache{i}_out']['quantization'];ts,tz=ins[f'cache{i}']['quantization']
            v=out[f'cache{i}_out']
            if (s,z)==(ts,tz):states[f'cache{i}']=v.copy()
            else:states[f'cache{i}']=np.clip(np.rint((v.astype(np.float64)-z)*s/ts)+tz,-128,127).astype(np.int8)
    scores=(codes.astype(np.float64)/256).astype(np.float32)
    return scores,codes,dict(input=[scale,zero],output=[1/256,0],native_output=[1/256,-128],
                            signed_to_uint8_offset=128,native_signed_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
                            quantized_input_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
                            cache_reset='input quantization zero points at complete stream start only',
                            cache_requantization_required=[ins[f'cache{i}']['quantization']!=outs[f'cache{i}_out']['quantization'] for i in range(7)])
