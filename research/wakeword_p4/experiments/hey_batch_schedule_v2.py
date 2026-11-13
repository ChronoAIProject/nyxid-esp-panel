"""Freeze v2 full tensors and every original/effective positive draw; no model."""
import argparse
import hashlib
import json
from pathlib import Path

from audit_hey_training_batch_v2 import audit
from hey_batch_schedule_v1 import BatchGuard,batch_digest


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_schedule(root,target,dest):
    trace=dest.with_suffix('.positive-crops.jsonl')
    env=audit(root,target,return_sampler=True,trace_path=trace)
    with dest.open('x') as f:
        for index in range(1,20101):
            values=env['batch']() if index<=20000 else env['batch'](mine=False,augment=False)
            f.write(json.dumps(dict(index=index,kind='train' if index<=20000 else 'ptq',
                                    tensor_sha256=batch_digest(*values)),sort_keys=True)+'\n')
            if index%1000==0:print(json.dumps(dict(target=target,scheduled=index)),flush=True)
    crop=env['crop_rule'].finish();assert crop['positive_draws']==643200
    return dict(target=target,batches=20100,schedule_sha256=sha(dest),sampler_policy='hey-prefix-preserving-v2',
                positive_crop=crop,crop_rule_sha256=sha(Path(__file__).with_name('hey_prefix_crop_v2.py')),
                trainer_sha256=sha(Path(env['sequence'].__code__.co_filename)),
                target_provenance_sha256=sha(root/'target-provenance.json'),
                schedule_generator_sha256=sha(__file__),auditor_sha256=sha(Path(__file__).with_name('audit_hey_training_batch_v2.py')),
                training_launched=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--target',choices=('hey_willow','hey_nyx'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();result=write_schedule(a.features,a.target,a.output)
    with a.receipt.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
