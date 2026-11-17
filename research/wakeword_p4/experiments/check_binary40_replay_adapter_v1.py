"""Independent replay-adapter regression against preserved analytic ABI receipt."""
import hashlib,json
from pathlib import Path
import numpy as np
from ai_edge_litert.interpreter import Interpreter,OpResolverType
from binary40_stream_predict_v1 import predict
from export_binary40_trained_v1 import sha

def main():
    original=Path('research/wakeword_p4/results/4090-binary40-feasibility-original-v1.json')
    receipt=json.loads(original.read_text());path=receipt['model']['path'];assert sha(path)==receipt['model']['sha256']
    scale,zero=receipt['inputs']['chunk']['scale'],receipt['inputs']['chunk']['zero']
    rng=np.random.default_rng(1726);quantized=[]
    for _ in range(256):
        raw=rng.uniform(0,26,(1,3,1,40))
        quantized.append(np.clip(np.rint(raw/scale)+zero,-128,127).astype(np.int8).reshape(3,40))
    q=np.concatenate(quantized)
    values=((q.astype(np.float64)-zero)*scale).astype(np.float32)
    assert np.array_equal(np.clip(np.rint(values/scale)+zero,-128,127).astype(np.int8),q)
    stream=hashlib.sha256();native=[]
    class Traced:
        def __init__(self,**kw):self.inner=Interpreter(num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,**kw)
        def allocate_tensors(self):self.inner.allocate_tensors()
        def get_signature_runner(self):
            runner=self.inner.get_signature_runner()
            def call(**kw):
                out=runner(**kw)
                for k,v in sorted(out.items()):stream.update(v.tobytes())
                native.append(int(out['probability'][0,0,0,0]))
                return out
            call.get_input_details=runner.get_input_details
            call.get_output_details=runner.get_output_details
            return call
    scores,codes,info=predict(values,path,Traced)
    assert stream.hexdigest()==receipt['feedback_sha256']
    assert np.array_equal(codes,np.array(native,dtype=np.int16)+128)
    assert np.array_equal(scores,codes.astype(np.float32)/256)
    assert info['quantized_input_sha256']==hashlib.sha256(q.tobytes()).hexdigest()
    assert info['native_signed_sha256']==hashlib.sha256(np.array(native,np.int8).tobytes()).hexdigest()
    result=dict(schema='binary40-replay-adapter-regression-v1',files={str(p):sha(p) for p in [Path(__file__),original,Path('research/wakeword_p4/experiments/binary40_stream_predict_v1.py')]},
                model_sha256=sha(path),calls=256,all_output_and_cache_digest=stream.hexdigest(),
                matches_preserved_independent_export_feedback_exact=True,signed_mapping_exact=True,input_codes_exact=True,
                policy='Analytic untrained model only. Original independently executed exporter receipt predates adapter. Reconstruct same quantized feature stream and compare all native outputs/caches at every call via digest; tests reset and feedback wiring. No training/development audio or accuracy claim. Trained export/replay remain separately required.')
    dest=Path('research/wakeword_p4/results/4090-binary40-replay-adapter-regression-v1.json')
    with dest.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
