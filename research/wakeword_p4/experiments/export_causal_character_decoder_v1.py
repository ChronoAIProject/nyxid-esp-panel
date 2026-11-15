"""Separate final-head INT8 character export; not a rescue of binary results."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from export_causal_sequence_smoke_v1 import sha, R

LOCK = R/'4090-causal-character-decoder-int8-plan-v1.json'
OUT = Path('.local/wakeword/causal-character-decoder-int8-v1')
REPORT = R/'4090-causal-character-decoder-int8-original-v1.json'

def plan():
    parent = R/'4090-causal-character-auxiliary-int8-plan-v1.json'
    p = json.loads(parent.read_text())
    training = R/'4090-causal-character-auxiliary-original-v1.json'
    r = json.loads(training.read_text())
    head = Path('.local/wakeword/causal-character-auxiliary-v1/final-character-head-variables.npz')
    assert sha(head) == r['files'][head.name] and r['steps'] == 20000
    paths = [Path(__file__), parent, training, head, Path(p['weights']),
             Path(__file__).with_name('causal_character_decoder_v1.py'),
             Path(__file__).with_name('compact_character_decoder_proposal_v1.md'),
             Path(__file__).with_name('probe_hey_context_transfer_shapes_v1.py')]
    paths += list(Path('.local/wakeword/microWakeWord/microwakeword').rglob('*.py'))
    files = dict(p['files']); files.update({str(x):sha(x) for x in paths})
    result = dict(schema='causal-character-decoder-int8-plan-v1', files=files,
                  weights=p['weights'], head=str(head), batch=p['batch'], lengths=p['lengths'],
                  versions={n:importlib.metadata.version(n) for n in ('tensorflow','keras','numpy','ai-edge-litert')},
                  expected_new_gib=.1, retries=0,
                  policy='Distinct diagnostic direct character decoder; fixed final encoder/head, five encoder states, 28 raw logits, builtin integer export, same fixed representative sequences. No retraining/calibration retry/numerical equality gate/threshold rescue/head selection for promotion. Frozen binary v30 unchanged; full continuous waveform behavior required.')
    with LOCK.open('x') as f: f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(LOCK),head_sha256=sha(head))))

def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)]) == LOCK.read_bytes()
    p = json.loads(LOCK.read_text())
    for path,h in p['files'].items(): assert sha(path)==h,path
    for name,v in p['versions'].items(): assert importlib.metadata.version(name)==v
    subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','.1'],check=True)
    OUT.mkdir(exist_ok=False)
    launch=dict(pid=os.getpid(),proc_starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(LOCK))
    (OUT/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    import probe_hey_context_transfer_shapes_v1 as helper
    from microwakeword import utils
    from microwakeword.layers import modes
    from tensorflow.lite.python import schema_py_generated as schema
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    tf=helper.tf
    original=helper.create(309); original.load_weights(p['weights'])
    for layer in original.layers:
        if isinstance(layer,tf.keras.layers.BatchNormalization): layer.trainable=False
    encoder=next(layer for layer in reversed(original.layers) if isinstance(layer,tf.keras.layers.Activation))
    assert int(encoder.output.shape[-1])==60
    head=tf.keras.layers.Dense(28,name='deployed_character_logits')
    output=head(tf.keras.ops.squeeze(encoder.output,axis=2))
    model=tf.keras.Model(original.input,output)
    with np.load(p['head']) as a: head.set_weights([a['v000'],a['v001']])
    stream=utils.to_streaming_inference(model,{'stride':3,'spectrogram_length':309},modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    stream(np.zeros((1,3,40),np.float32),training=False)
    states=[v for v in stream.variables if 'states' in v.name]; assert len(states)==5
    for state in states: state.assign(tf.zeros_like(state))
    stream.export(str(OUT/'saved_model'))
    with np.load(p['batch']) as b: batch=b['x'].copy()
    sequences=[np.pad(batch[i,:n],((0,(-n)%3),(0,0))) for i,n in enumerate(p['lengths'])]
    def representative():
        yield [np.zeros((1,3,40),np.float32)]
        yield [np.full((1,3,40),26,np.float32)]
        for values in sequences:
            for at in range(0,len(values),3): yield [values[None,at:at+3].copy()]
    converter=tf.lite.TFLiteConverter.from_saved_model(str(OUT/'saved_model'))
    converter.optimizations=[tf.lite.Optimize.DEFAULT]; converter._experimental_variable_quantization=True
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type=tf.int8; converter.inference_output_type=tf.int8
    converter.representative_dataset=representative
    path=OUT/'int8.tflite'; path.write_bytes(converter.convert())
    graph=schema.Model.GetRootAsModel(path.read_bytes(),0)
    names={v:k for k,v in vars(schema.BuiltinOperator).items() if isinstance(v,int)}
    ops=[]
    for si in range(graph.SubgraphsLength()):
        sg=graph.Subgraphs(si)
        for ti in range(sg.TensorsLength()): assert sg.Tensors(ti).Type()!=schema.TensorType.FLOAT32
        for oi in range(sg.OperatorsLength()): ops.append(names[graph.OperatorCodes(sg.Operators(oi).OpcodeIndex()).BuiltinCode()])
    assert 'CUSTOM' not in ops and 'DEQUANTIZE' not in ops
    it=Interpreter(model_path=str(path),num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF); it.allocate_tensors()
    a,=it.get_input_details(); b,=it.get_output_details()
    assert list(a['shape'])==[1,3,40] and a['dtype']==np.int8
    assert list(b['shape'])==[1,1,28] and b['dtype']==np.int8 and b['quantization'][0]>0
    scale,zero=a['quantization']; codes=[]
    q=np.clip(np.rint(sequences[0]/scale)+zero,-128,127).astype(np.int8)
    for at in range(0,len(q),3):
        it.set_tensor(a['index'],q[None,at:at+3]); it.invoke(); codes.append(it.get_tensor(b['index'])[0,0].copy())
    replay=OUT/'functional-replay.npz'; np.savez_compressed(replay,logits=np.asarray(codes,np.int8))
    result=dict(schema='causal-character-decoder-int8-original-v1',launch=launch,plan_sha256=sha(LOCK),
                model=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)),encoder_states=5,
                input_quantization=list(a['quantization']),output_quantization=list(b['quantization']),
                functional_calls=len(codes),replay_sha256=sha(replay),ops=ops,
                all_static_and_runtime_abi_passed=True,recognition_scored=False,candidate_promoted=False,policy=p['policy'])
    with REPORT.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run'])
    args=parser.parse_args();plan() if args.action=='plan' else run()
