"""Exactly two GPU updates and original INT8 export per arm; no formal launch."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from train_hey_broad_adapter_v1 import TRAINER, adapted_tree, admit_args, load_initial

LOCK = Path('research/wakeword_p4/results/4090-hey-broad-smoke-lock-v1.json')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''): h.update(b)
    return h.hexdigest()


class Guard:
    def __init__(self, expected):
        self.expected=expected; self.count=0; self.h=hashlib.sha256()
    def __call__(self,x,y,w):
        assert x.shape==(128,225,40) and np.isfinite(x).all()
        assert np.count_nonzero(y)==32 and np.array_equal(w,np.where(y[:,0]==1,1.,20.))
        self.count+=1
        if self.count<=50:
            for a in (x,y,w): self.h.update(a.tobytes())
        if self.count==50: assert self.h.hexdigest()==self.expected
        return x,y,w
    def finish(self):
        assert self.count==102 and self.h.hexdigest()==self.expected
        return dict(batches=self.count,first50_cpu_tensor_match=True,first50_sha256=self.expected)


def main(target):
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])
    assert remote==LOCK.read_bytes()
    lock=json.loads(remote)
    for path,h in lock['files'].items(): assert sha(path)==h, path
    root=Path('.local/wakeword/hey-broad-training-features-v1')/target
    prov=json.loads((root/'target-provenance.json').read_text())
    assert prov['target_id']==target and prov['context_frames']==225
    for name,h in prov['files'].items(): assert sha(root/name)==h
    output=Path('.local/wakeword/hey-broad-smoke-v1')/target
    cpu=json.loads(Path('research/wakeword_p4/results/4090-hey-broad-cpu-batch-audit-v1.json').read_text())
    expected=next(r['tensor_sha256'] for r in cpu['results'] if r['target']==target)
    guard=Guard(expected)
    def admit(args):
        admit_args(args)
        assert args.steps==2 and args.eval_interval==2 and args.device=='cuda'
    def initial(model):
        h=load_initial(model)
        inventory=[dict(shape=list(v.shape),dtype=str(v.numpy().dtype),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for v in model.variables]
        (output/'loaded-initial-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
        return h
    sys.path.insert(0,str(TRAINER.parent))
    sys.argv=[str(TRAINER),'--upstream','.local/wakeword/microWakeWord','--features',str(root),
        '--output',str(output),'--steps','2','--eval-interval','2','--architecture','teacher60',
        '--context-frames','225','--batchnorm-policy','frozen','--device','cuda',
        '--prefix-negatives','16','--source-exponent','0','--ambient-sampling','stratified']
    exec(compile(adapted_tree(),str(TRAINER),'exec'),dict(__name__='__main__',__file__=str(TRAINER),
        HEY_ADMIT_ARGS=admit,HEY_LOAD_INITIAL=initial,
        HEY_INITIAL_SHA='e47f196aae631d7c8dee0564bd9476fe16c04312a04a1b3fd5687a183899f155',HEY_BATCH_GUARD=guard))
    report=json.loads((output/'training.json').read_text())
    assert report['selected_step']==2 and report['parameters']==24301
    import tensorflow as tf
    interpreter=tf.lite.Interpreter(model_path=str(output/'hey_nyx.tflite'))
    interpreter.allocate_tensors()
    inp=interpreter.get_input_details()[0];out=interpreter.get_output_details()[0]
    assert list(inp['shape'])==[1,3,40] and inp['dtype']==np.int8
    assert list(out['shape'])==[1,1] and out['dtype']==np.uint8
    result=dict(target=target,steps=2,batches=guard.finish(),model_sha256=sha(output/'hey_nyx.tflite'),
        input_shape=[1,3,40],output_shape=[1,1],scope='Two GPU updates/export only; no recognition or formal candidate.')
    with (output/'smoke-result.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('hey_willow','hey_nyx')
    main(sys.argv[1])
