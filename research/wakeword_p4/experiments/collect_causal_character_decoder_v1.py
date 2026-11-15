"""One fixed greedy decoder on complete consumed waves; no boundary resets."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
from collect_multispeaker_formal_v1 import prepare
from collect_multispeaker_priming_v1 import PRIME, sha, R
from causal_character_decoder_v1 import Decoder

LOCK=R/'4090-causal-character-decoder-waveform-plan-v1.json'
ROOT=Path('.local/wakeword/causal-character-decoder-waveform-v1')

def plan():
    export=R/'4090-causal-character-decoder-int8-original-v1.json'
    model=json.loads(export.read_text());assert model['all_static_and_runtime_abi_passed']
    files={};groups={}
    for group in ('primed','jenny','joe'):
        parent=R/f'4090-causal-character-auxiliary-{group}-plan-v1.json'
        p=json.loads(parent.read_text());files.update(p['files']);files[str(parent)]=sha(parent)
        groups[group]=p['materials']
    paths=[Path(__file__),Path(__file__).with_name('causal_character_decoder_v1.py'),export,Path(model['model']['path'])]
    files.update({str(p):sha(p) for p in paths})
    out=dict(schema='causal-character-decoder-waveform-plan-v1',files=files,groups=groups,model=model['model'],
             versions={n:importlib.metadata.version(n) for n in ('numpy','soundfile','ai-edge-litert','pymicro-features')},
             prime_samples=PRIME,expected_new_gib=.1,retries=0,
             policy='Direct greedy CTC exact HEY NICKS protocol committed 6e2ac312. Complete consumed legacy372/Jenny408/Joe360, continuous81600prime. Store every argmax token and actual event, lowest-ID tie. No threshold/smoothing/binary fusion/finishflush/source reset/aliases/newaudio/modeltraining/finalaccess. Preserve base complete-input frontend finalhop/pads and native480return clock. Prime notifications ignored only by app, state retained. No promotion/head selection for qualification.')
    with LOCK.open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(LOCK),groups=list(groups))))

def run(group):
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
    p=json.loads(LOCK.read_text())
    for path,h in p['files'].items():assert sha(path)==h,path
    for name,v in p['versions'].items():assert importlib.metadata.version(name)==v
    subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','.1'],check=True)
    out=ROOT/group;out.mkdir(parents=True,exist_ok=False)
    launch=dict(pid=os.getpid(),proc_starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(LOCK))
    (out/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    try:
        import soundfile as sf
        from ai_edge_litert.interpreter import Interpreter,OpResolverType
        sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
        from features import features
        records=[]
        for phase,material in p['groups'][group].items():
            (out/phase).mkdir()
            for i,row in enumerate(json.loads(Path(material['path']).read_text())['records']):
                wav=row['path'] if phase=='synthetic' else row['wav_path'];pcm,rate=sf.read(wav,dtype='int16')
                assert rate==16000 and pcm.ndim==1 and len(pcm)==row['samples']
                assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
                combined=np.concatenate((np.zeros(PRIME,np.int16),pcm));raw=features(combined)
                values,support,clock=prepare(raw,len(combined))
                blocks=np.minimum(((support+479)//480)*480,len(combined));finish=((support+479)//480)*480>len(combined)
                it=Interpreter(model_path=p['model']['path'],num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF);it.allocate_tensors()
                a,=it.get_input_details();b,=it.get_output_details()
                assert list(a['shape'])==[1,3,40] and list(b['shape'])==[1,1,28]
                scale,zero=a['quantization'];q=np.clip(np.rint(values/scale)+zero,-128,127).astype(np.int8)
                decoder=Decoder();tokens=[];events=[]
                assert len(q)==3*len(support)
                for step,j in enumerate(range(0,len(q),3)):
                    it.set_tensor(a['index'],q[None,j:j+3]);it.invoke()
                    token=int(np.argmax(it.get_tensor(b['index'])[0,0]));tokens.append(token)
                    if decoder.step(token):
                        events.append(dict(step=step,sample=int(blocks[step])-PRIME,support_relative_sample=int(support[step])-PRIME,finish_call=bool(finish[step])))
                trace=out/phase/f'{i:06}.npz'
                np.savez_compressed(trace,tokens=np.asarray(tokens,np.uint8),support_relative_sample=support-PRIME,block_return_sample=blocks-PRIME,finish_call=finish)
                rec=dict(phase=phase,index=i,record=row,events=events,trace_path=str(trace),trace_sha256=sha(trace),steps=len(tokens),prime_calls=int(np.sum(support<=PRIME)),input_sha256=hashlib.sha256(values.tobytes()).hexdigest())
                records.append(rec)
                with (out/'records.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
                print(json.dumps(dict(group=group,phase=phase,index=i,steps=len(tokens),events=len(events))),flush=True)
        assert len(records)==dict(primed=372,jenny=408,joe=360)[group]
        result=dict(schema='causal-character-decoder-waveform-original-v1',execution_sha256=sha(LOCK),launch=launch,group=group,model=p['model'],records=records,all_passed=True,threshold_selected=False,candidate_promoted=False,policy=p['policy'])
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(completed=True,group=group,records=len(records))),flush=True)
    except Exception:
        (out/'failure.log').write_text(traceback.format_exc());raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run']);parser.add_argument('--group',choices=['primed','jenny','joe'])
    a=parser.parse_args()
    if a.action=='run' and not a.group:parser.error('--group required for run')
    plan() if a.action=='plan' else run(a.group)
