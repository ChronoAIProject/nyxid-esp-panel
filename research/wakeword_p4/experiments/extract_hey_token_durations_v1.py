"""Read seeded VITS token allocations only; never regenerate/select waveforms.

Internal token allocations are generated references, not acoustic phonetic truth.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import onnx
import onnxruntime as ort

from generate_hey_paired_material_v2 import seeded_model, versions

ROOT=Path('research/wakeword_p4/results')
PLAN=ROOT/'4090-hey-token-duration-plan-v1.json'
GENERATION=ROOT/'4090-hey-paired-generation-plan-v2.json'
NATIVE=Path('.local/wakeword/hey-paired-training-native-v2')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_plan():
    generation=json.loads(GENERATION.read_text())
    r=dict(schema='hey-token-allocation-extraction-plan-v1',generation_plan_sha256=sha(GENERATION),
           files={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('generate_hey_paired_material_v2.py')]},
           versions=versions(),requests=[r['id'] for r in generation['requests'] if r['role']=='training'],
           output_node='/Ceil_output_0',expected_decoder_strides=[8,8,4],
           rules='Only graph outputs changed to expose token allocations from explicitly seeded graph. Require finite nonnegative integer allocations, one per input token; sum*256 must equal saved native waveform length. Prefix boundary excludes comma for commands or final period for bare, includes preceding inter-token blank; resampled reference is integer ceil(native_boundary*16000/22050).',
           smoke_request='v0-t00-c0-s0-r00',
           full_extraction_requires_complete_native_quality_pass=True,
           waveform_regeneration=False,training_launched=False,
           limits='Model-internal allocation reference only; not true acoustic phone durations or parity latency certification. Any mismatch invalidates this timing admission; no token adjustments to force agreement.')
    with PLAN.open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(PLAN),waveform_generated=False)))


def check_plan():
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes()
    plan=json.loads(remote);assert sha(GENERATION)==plan['generation_plan_sha256']
    assert versions()==plan['versions']
    for p,h in plan['files'].items():assert sha(p)==h
    return plan,json.loads(GENERATION.read_text())


def extract(row,generation):
    voice=next(v for v in generation['voices'] if v['voice']==row['voice'])
    model_path=voice['files']['model']['path'];config_path=voice['files']['config']['path']
    assert sha(model_path)==voice['files']['model']['sha256'] and sha(config_path)==voice['files']['config']['sha256']
    model_bytes=seeded_model(model_path,row['seed'])
    assert hashlib.sha256(model_bytes).hexdigest()==row['derived_model_sha256']
    model=onnx.load_model_from_string(model_bytes)
    assert any('/Ceil_output_0' in n.output and n.op_type=='Ceil' for n in model.graph.node)
    strides=[list(a.ints)[0] for n in model.graph.node if n.op_type=='ConvTranspose'
             for a in n.attribute if a.name=='strides']
    assert strides==[8,8,4]
    del model.graph.output[:]
    model.graph.output.append(onnx.helper.make_tensor_value_info('/Ceil_output_0',onnx.TensorProto.FLOAT,[1,1,None]))
    metadata_model=model.SerializeToString()
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.log_severity_level=3
    session=ort.InferenceSession(metadata_model,sess_options=options,providers=['CPUExecutionProvider'])
    tokens=np.asarray([row['token_ids']],dtype=np.int64)
    feed=dict(input=tokens,input_lengths=np.asarray([tokens.shape[1]],dtype=np.int64),
              scales=np.asarray([generation['noise_scale'],row['length_scale'],generation['noise_w']],dtype=np.float32))
    if any(i.name=='sid' for i in session.get_inputs()):feed['sid']=np.asarray([row['speaker_id']],dtype=np.int64)
    duration=session.run(None,feed)[0]
    assert duration.shape==(1,1,tokens.shape[1])
    d=duration.reshape(-1)
    assert np.isfinite(d).all() and np.all(d>=0) and np.array_equal(d,np.rint(d))
    total=int(d.sum())*256
    assert total==row['native_samples'],f'Allocation/sample mismatch: {total} vs {row["native_samples"]}'
    phones=row['phonemes'];marker=',' if row['carrier'] else '.'
    cutoff=phones.index(marker)
    ids=json.loads(Path(config_path).read_text())['phoneme_id_map']
    end=len(ids['^'])+len(ids['_'])+sum(len(ids[p])+len(ids['_']) for p in phones[:cutoff])
    native_end=int(d[:end].sum())*256
    assert 0<native_end<=total
    return dict(id=row['id'],role=row['role'],voice=row['voice'],
                source_raw_sha256=row['raw_sha256'],source_waveform_sha256=row['sha256'],
                original_derived_model_sha256=row['derived_model_sha256'],
                metadata_model_sha256=hashlib.sha256(metadata_model).hexdigest(),
                token_allocations_frames=d.astype(int).tolist(),native_samples_per_frame=256,
                native_samples=total,prefix_token_count=end,prefix_marker=marker,
                generated_prefix_end_native_sample=native_end,
                generated_prefix_end_16k_sample=(native_end*16000+22049)//22050,
                allocation_matches_saved_waveform_length=True,waveform_generated=False,
                limits='Internal VITS allocation before punctuation; includes blank allocation, not measured phonetic boundary.')


def run(smoke):
    plan,generation=check_plan()
    # Ignore any incomplete last line while synthesis is still running.
    rows=[json.loads(b) for b in (NATIVE/'records.jsonl').read_bytes().splitlines(keepends=True) if b.endswith(b'\n')]
    if smoke:
        rows=[r for r in rows if r['id']==plan['smoke_request']]
        assert len(rows)==1
        destination=ROOT/'4090-hey-token-duration-smoke-v1.json'
    else:
        summary=json.loads((NATIVE/'summary.json').read_text())
        assert summary['pack_qualified'] and summary['records_sha256']==sha(NATIVE/'records.jsonl')
        assert [r['id'] for r in rows]==plan['requests']
        destination=Path('.local/wakeword/hey-paired-training-token-durations-v1.jsonl')
    with destination.open('x') as log:
        for i,row in enumerate(rows):
            assert row['role']=='training' and row['audit']['passed']
            assert sha(NATIVE/row['raw_path'])==row['raw_sha256'] and sha(NATIVE/row['path'])==row['sha256']
            result=extract(row,generation)
            log.write(json.dumps(result)+'\n');log.flush()
            if i%20==0:print(json.dumps(dict(metadata_records=i+1,waveform_generated=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('write-plan','smoke','extract-training'))
    a=p.parse_args()
    make_plan() if a.action=='write-plan' else run(a.action=='smoke')
