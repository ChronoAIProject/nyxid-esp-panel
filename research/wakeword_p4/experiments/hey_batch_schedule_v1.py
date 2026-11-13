"""Precompute and enforce the actual base batch tensor sequence; no model.

Includes20000gradient batches and100PTQ batches. Source inventory and the exact
unmodified seeded sampler determine the sequence; this is not a separate live
row-provenance recorder, nor a gradient/export correctness certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path

from audit_hey_training_batch_v1 import audit


def batch_digest(x,y,weights):
    h=hashlib.sha256()
    for value in (x,y,weights):h.update(value.tobytes())
    return h.hexdigest()


def write_schedule(root,target,dest):
    env=audit(root,target,return_sampler=True)
    with dest.open('x') as f:
        for index in range(1,20101):
            # Default training flags and PTQ flags are explicit. Mining1 and
            # zero-width augmentation preserve the same values/RNG behavior.
            values=env['batch']() if index<=20000 else env['batch'](mine=False,augment=False)
            row=dict(index=index,kind='train' if index<=20000 else 'ptq',
                     tensor_sha256=batch_digest(*values))
            f.write(json.dumps(row,sort_keys=True)+'\n')
            if index%1000==0:print(json.dumps(dict(target=target,scheduled=index)),flush=True)
    return dict(target=target,batches=20100,schedule_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                trainer_sha256=hashlib.sha256(Path(env['sequence'].__code__.co_filename).read_bytes()).hexdigest(),
                target_provenance_sha256=hashlib.sha256((root/'target-provenance.json').read_bytes()).hexdigest(),
                schedule_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                auditor_sha256=hashlib.sha256(Path(__file__).with_name('audit_hey_training_batch_v1.py').read_bytes()).hexdigest(),
                training_launched=False)


class BatchGuard:
    def __init__(self,schedule,output,train_steps=20000,ptq_batches=100):
        self.rows=[json.loads(line) for line in schedule.read_text().splitlines()]
        self.train_steps=train_steps;self.expected=train_steps+ptq_batches
        if len(self.rows)!=self.expected:raise ValueError('Wrong batch schedule length')
        for i,row in enumerate(self.rows,1):
            if row['index']!=i or row['kind']!=('train' if i<=train_steps else 'ptq'):
                raise ValueError('Wrong batch schedule order/phase')
        self.output=output;self.count=0;self.log=None

    def __call__(self,x,y,weights):
        if self.count>=self.expected:raise ValueError('Extra unplanned batch')
        wanted=self.rows[self.count]
        digest=batch_digest(x,y,weights)
        if self.log is None:self.log=(self.output/'actual-batch-schedule.jsonl').open('x')
        row=dict(index=self.count+1,kind=wanted['kind'],tensor_sha256=digest,
                 matches_expected=digest==wanted['tensor_sha256'])
        self.log.write(json.dumps(row,sort_keys=True)+'\n');self.log.flush()
        if not row['matches_expected']:raise ValueError('Actual model-input batch differs from locked schedule')
        self.count+=1
        return x,y,weights

    def finish(self):
        if self.count!=self.expected:raise ValueError('Incomplete training/PTQ batch schedule')
        if self.log:self.log.close()
        path=self.output/'actual-batch-schedule.jsonl'
        return dict(verified_batches=self.count,train_batches=self.train_steps,
                    ptq_batches=self.count-self.train_steps,actual_trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    all_actual_batch_tensors_match_precomputed=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--target',choices=('hey_willow','hey_nyx'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();result=write_schedule(a.features,a.target,a.output)
    with a.receipt.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
