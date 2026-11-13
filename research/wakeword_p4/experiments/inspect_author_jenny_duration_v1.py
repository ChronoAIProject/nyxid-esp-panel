"""Static duration-output feasibility for existing author Jenny; no inference."""
import hashlib
import json
from pathlib import Path
import onnx


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    identity=Path('research/wakeword_p4/results/4090-author-jenny-model-identity-v1.json')
    d=json.loads(identity.read_text());source=Path(d['files']['jenny.onnx']['local_path'])
    assert sha(source)==d['files']['jenny.onnx']['sha256']
    model=onnx.load(str(source));assert [x.name for x in model.graph.input]==['input','input_lengths','scales']
    ceil=[n for n in model.graph.node if n.op_type=='Ceil'];assert len(ceil)==1
    node=ceil[0];assert node.name=='/Ceil' and list(node.input)==['/Mul_1_output_0']
    producers={name:n for n in model.graph.node for name in n.output}
    upstream=producers[node.input[0]];assert upstream.op_type=='Mul'
    before=producers[upstream.input[0]];assert before.op_type=='Mul'
    assert producers[before.input[0]].op_type=='Exp'
    consumers=[n for n in model.graph.node if node.output[0] in n.input]
    assert any(n.op_type=='ReduceSum' for n in consumers)
    upsample=[]
    for n in model.graph.node:
        if n.op_type=='ConvTranspose':
            attrs={a.name:onnx.helper.get_attribute_value(a) for a in n.attribute}
            upsample.append(dict(name=n.name,strides=list(attrs['strides']),kernel=list(attrs['kernel_shape']),pads=list(attrs['pads'])))
    assert [r['strides'] for r in upsample]==[[8],[8],[4]]
    stochastic=[dict(name=n.name,op=n.op_type) for n in model.graph.node if 'Random' in n.op_type]
    result=dict(schema='author-jenny-static-duration-feasibility-v1',source_sha256=sha(source),
        identity_receipt_sha256=sha(identity),script_sha256=sha(Path(__file__)),onnx_version=onnx.__version__,
        inputs=[x.name for x in model.graph.input],outputs=[x.name for x in model.graph.output],
        duration_candidate=dict(name=node.output[0],op=node.op_type,consumers=[n.op_type for n in consumers]),
        decoder_upsampling=upsample,proposed_samples_per_duration_frame=256,stochastic_nodes=stochastic,
        model_modified=False,inference_executed=False,source_reserved=False,
        next='Separately freeze a duration-observer export and generation-only protocol before any synthesis; actual token-length/sum*256 waveform identity must be tested on saved outputs, not inferred as passed from graph structure.',
        limits='Static graph feasibility only. Does not prove exact waveform after additional output export, seed control, source independence, acoustic completeness or duration truth.')
    path=Path('research/wakeword_p4/results/4090-author-jenny-static-duration-feasibility-v1.json')
    with path.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(duration=node.output[0],upsample=256,stochastic_nodes=len(stochastic),inference=False)))


if __name__=='__main__':main()
