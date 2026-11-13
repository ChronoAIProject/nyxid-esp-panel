"""Two actual optimizer updates and original INT8 export on a locked new pack.

Separate diagnostic outputs, never a checkpoint for formal20000 or development.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from audit_hey_training_batch_v2 import audit
from hey_batch_schedule_v2 import BatchGuard,batch_digest
from train_hey_target_adapter_v2 import adapted_tree,TRAINER
from validate_hey_target_pack_v1 import validate
from hey_prefix_crop_v2 import CropGuard


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Observe:
    def __init__(self):self.steps=[];self.initial=None

    def initialize(self,model,root):
        self.root=root;self.initial=[v.numpy().copy() for v in model.weights]
        inventory=[]
        for variable,value in zip(model.weights,self.initial,strict=True):
            inventory.append(dict(name=variable.name,shape=list(value.shape),dtype=str(value.dtype),
                                  value_sha256=hashlib.sha256(value.tobytes()).hexdigest()))
        model.save_weights(root/'smoke-initial.weights.h5')
        (root/'smoke-initial-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')

    def update(self,model,x,y,weights):
        step=len(self.steps)+1
        assert step<=2 and int(model.optimizer.iterations)==step-1
        np.savez_compressed(self.root/f'smoke-actual-batch-{step}.npz',x=x,y=y,weights=weights)
        loss=float(model.train_on_batch(x,y,sample_weight=weights))
        assert np.isfinite(loss) and int(model.optimizer.iterations)==step
        assert all(np.isfinite(v.numpy()).all() for v in model.weights)
        changed=sum(not np.array_equal(v.numpy(),old) for v,old in zip(model.weights,self.initial,strict=True))
        assert changed>0,'No actual model update'
        self.steps.append(dict(step=step,loss=loss,optimizer_iterations=step,changed_from_initial=changed,
                               actual_batch_sha256=batch_digest(x,y,weights)))
        return loss


def smoke_tree():
    tree=adapted_tree()
    class Instrument(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value=='fixed-final-20000':return ast.copy_location(ast.Constant('integration-smoke-two-updates'),node)
            if node.value=='Fixed final20000; no development scoring or selection':
                return ast.copy_location(ast.Constant('Diagnostic two updates only; never formal candidate'),node)
            return node
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and ast.unparse(node.func)=='model.train_on_batch':
                return ast.copy_location(ast.Call(func=ast.Name(id='HEY_SMOKE_UPDATE',ctx=ast.Load()),
                    args=[ast.Name(id='model',ctx=ast.Load()),*node.args,node.keywords[0].value],keywords=[]),node)
            return node
    tree=Instrument().visit(tree)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loop=next(n for n in ast.walk(main) if isinstance(n,ast.For) and ast.unparse(n.target)=='step')
    # Keep the first two formal learning rates. The original relative schedule
    # would otherwise move both decay boundaries to step1 of a two-step run.
    for node in loop.body:
        if isinstance(node,ast.If) and ast.unparse(node.test) in ('step == int(args.steps * 0.5)','step == int(args.steps * 0.8)'):
            boundary=10000 if '0.5' in ast.unparse(node.test) else 16000
            node.test=ast.parse(f'step == {boundary}',mode='eval').body
    index=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Expr) and ast.unparse(n).startswith('model.compile('))
    main.body[index+1:index+1]=ast.parse('HEY_SMOKE_INITIAL(model,root)').body
    return ast.fix_missing_locations(tree)


def training_stream_checks(model,root,output):
    """Check full training PCM against saved features and fresh raw-byte streams."""
    import soundfile as sf
    from features import features
    from uint8_stream_detector_v1 import Uint8StreamDetector
    rows=json.loads((root/'features.json').read_text())
    stored=np.memmap(root/'features.bin',mode='r',dtype='<u2').reshape(-1,40)
    selected=[next(r for r in rows if r['label']==label) for label in (1,0)]
    traces={};cases=[]
    for i,row in enumerate(selected):
        wav=Path(row['path']);assert sha(wav)==row['sha256']
        pcm,rate=sf.read(wav,dtype='int16');assert rate==16000 and len(pcm)==row['samples']
        raw=features(pcm).astype('<u2')
        assert np.array_equal(raw,stored[row['offset']:row['offset']+row['frames']])
        detector=Uint8StreamDetector(model)
        first=detector.predict(raw);detector.reset();second=detector.predict(raw)
        assert first.dtype==np.uint8 and len(first)==len(raw)//3 and len(first)>0
        assert np.array_equal(first,second),'Fresh reset streaming bytes changed'
        traces[f'case{i}_first']=first;traces[f'case{i}_repeat']=second
        cases.append(dict(source_id=row['source_id'],variant=row['variant'],label=row['label'],
                          wav_sha256=row['sha256'],samples=len(pcm),frames=len(raw),steps=len(first),
                          raw_uint8_sha256=hashlib.sha256(first.tobytes()).hexdigest(),fresh_reset_exact=True))
    path=output/'smoke-training-waveform-streams.npz';np.savez_compressed(path,**traces)
    return dict(cases=cases,source='Complete training-only rendered PCM, no new dev/acceptance',
                saved_frontend_exact=True,all_fresh_reset_repeats_exact=True,
                raw_archive_sha256=sha(path),event_scoring=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-lock',type=Path,required=True)
    p.add_argument('--target',choices=('hey_willow','hey_nyx'),required=True)
    a=p.parse_args()
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(a.training_lock)])
    assert remote==a.training_lock.read_bytes();lock=json.loads(remote)
    assert lock['schema']=='hey-paired-training-lock-v2'
    for name,h in lock['files'].items():assert sha(name)==h,name
    key=str(Path(__file__).resolve().relative_to(Path.cwd()))
    assert lock['files'][key]==sha(__file__)
    for name,v in lock['versions'].items():assert importlib.metadata.version(name)==v
    root=Path(lock['arms'][a.target]['features'])
    validate(root,a.target,required_splits=('training',))
    output=Path('.local/wakeword')/f'hey-paired-{a.target}-integration-smoke-v2'
    assert not output.exists(),'Never reuse smoke output'
    # Precompute this diagnostic schedule separately; formal schedule unchanged.
    env=audit(root,a.target,return_sampler=True,trace_path=output.with_suffix('.expected-crops.jsonl'))
    schedule=output.with_suffix('.schedule.jsonl')
    with schedule.open('x') as f:
        for i in range(1,103):
            values=env['batch']() if i<=2 else env['batch'](mine=False,augment=False)
            f.write(json.dumps(dict(index=i,kind='train' if i<=2 else 'ptq',tensor_sha256=batch_digest(*values)))+'\n')
    expected_crop=env['crop_rule'].finish()
    assert expected_crop['positive_draws']==3264
    crop=CropGuard(output/'actual-positive-crops.jsonl',expected_crop)
    guard=BatchGuard(schedule,output,train_steps=2,ptq_batches=100);observe=Observe()
    sys.path.insert(0,str(TRAINER.parent))
    sys.argv=[str(TRAINER),'--features',str(root),'--output',str(output),'--upstream',lock['upstream'],
              '--initial-weights',lock['initial_weights'],'--architecture','teacher60','--context-frames','150',
              '--batchnorm-policy','frozen','--steps','2','--eval-interval','5000',
              '--source-exponent','0','--ambient-sampling','rows','--device','cuda']
    exec(compile(smoke_tree(),str(TRAINER),'exec'),dict(__name__='__main__',__file__=str(TRAINER),
         HEY_TARGET_ID=a.target,HEY_BATCH_GUARD=guard,HEY_PREFIX_CROP=crop,HEY_SMOKE_INITIAL=observe.initialize,HEY_SMOKE_UPDATE=observe.update))
    report=json.loads((output/'training.json').read_text())
    assert report['selected_step']==report['steps']==2 and len(observe.steps)==2
    assert report['batch_schedule']['verified_batches']==102
    import tensorflow as tf
    interpreter=tf.lite.Interpreter(model_path=str(output/'hey_nyx.tflite'));interpreter.allocate_tensors()
    inp=interpreter.get_input_details()[0];out=interpreter.get_output_details()[0]
    assert list(inp['shape'])==[1,3,40] and inp['dtype']==np.int8
    assert list(out['shape'])==[1,1] and out['dtype']==np.uint8
    assert tuple(out['quantization'])==(1/256,0)
    waveform=training_stream_checks(output/'hey_nyx.tflite',root,output)
    report.update(smoke_updates=observe.steps,training_lock_sha256=sha(a.training_lock),
                  training_waveform_streaming=waveform,
                  smoke_model_sha256=sha(output/'hey_nyx.tflite'),
                  formal_candidate=False,export_contract_passed=True,smoke_source_sha256=sha(__file__))
    (output/'training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(target=a.target,actual_optimizer_updates=2,export_contract_passed=True,formal_candidate=False)))


if __name__=='__main__':main()
