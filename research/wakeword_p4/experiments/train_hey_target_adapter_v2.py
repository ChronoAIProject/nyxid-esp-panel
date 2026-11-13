"""Separate prefix-preserving v2 fixed-final adapter; original trainer unchanged.

Launch requires pushed training/material locks. --audit-source only constructs
and checks the transformed program; it never imports TF or launches training.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from validate_hey_target_pack_v1 import validate
from hey_batch_schedule_v2 import BatchGuard
from hey_prefix_crop_v2 import apply_crop_rule, CropGuard

TRAINER = Path(__file__).resolve().parents[1] / 'training/train_stream.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_initial(model,output,expected):
    inventory=[]
    for variable in model.weights:
        value=variable.numpy()
        inventory.append(dict(name=variable.name,shape=list(value.shape),dtype=str(value.dtype),
                              value_sha256=hashlib.sha256(value.tobytes()).hexdigest()))
    if inventory!=json.loads(expected.read_text()):
        raise ValueError('Actual formal initialization differs from certified smoke values')
    model.save_weights(output/'formal-initial.weights.h5')
    (output/'formal-initial-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')


def adapted_tree():
    tree = apply_crop_rule(ast.parse(TRAINER.read_text()))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    compile_index=next(i for i,n in enumerate(main.body)
                       if isinstance(n,ast.Expr) and ast.unparse(n).startswith('model.compile('))
    main.body[compile_index+1:compile_index+1]=ast.parse('HEY_INITIAL_CAPTURE(model,root)').body
    loop = next(n for n in ast.walk(main) if isinstance(n, ast.For)
                and ast.unparse(n.target) == 'step' and ast.unparse(n.iter) == 'range(1, args.steps + 1)')
    blocks = [n for n in loop.body if isinstance(n, ast.If) and 'args.eval_interval' in ast.unparse(n.test)]
    assert len(blocks) == 1, 'Trainer evaluation structure changed'
    block = blocks[0]
    assert 'selected, rank, grid = validate()' in ast.unparse(block)
    block.body = ast.parse('''
item = {'step': step, 'loss': loss, 'checkpoint_policy': 'fixed-final-20000',
        'seconds': round(time.monotonic() - started, 1)}
history.append(item)
log.write(json.dumps(item) + '\\n')
log.flush()
print(json.dumps(item), flush=True)
if step == args.steps:
    selected_step = step
    model.save_weights(root / 'best.weights.h5')
''').body
    batch_fn = next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='batch')
    returns = [n for n in ast.walk(batch_fn) if isinstance(n,ast.Return)]
    assert len(returns)==1 and isinstance(returns[0].value,ast.Tuple)
    returns[0].value = ast.Call(func=ast.Name(id='HEY_BATCH_GUARD',ctx=ast.Load()),
                               args=returns[0].value.elts,keywords=[])
    # This check is inserted after parse_args, before any data/model operations.
    parse_index = next(i for i,n in enumerate(main.body)
                       if isinstance(n, ast.Assign) and ast.unparse(n) == 'args = parser.parse_args()')
    numerical = ast.parse('''
tf.config.experimental.enable_op_determinism()
tf.config.experimental.enable_tensor_float_32_execution(False)
''').body
    main.body[parse_index+1:parse_index+1] = numerical
    report_index = next(i for i,n in enumerate(main.body)
                        if isinstance(n,ast.Expr) and 'training.json' in ast.unparse(n)
                        and '.write_text(' in ast.unparse(n))
    main.body[report_index:report_index] = ast.parse('''
report.update(target_id=HEY_TARGET_ID, checkpoint_policy='fixed-final-20000',
              batch_schedule=HEY_BATCH_GUARD.finish(),
              positive_crop=HEY_PREFIX_CROP.finish(),sampler_policy='hey-prefix-preserving-v2',
              validation_detection='not run; development external',
              selection='Fixed final20000; no development scoring or selection',
              scope='Synthetic research candidate; independent acceptance required')
''').body
    tree = ast.fix_missing_locations(tree)
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call)
             and isinstance(n.func,ast.Name) and n.func.id == 'validate']
    assert not calls, 'Development scoring remains reachable'
    assert ast.unparse(tree).count('model.save_weights(') == 2  # inactive paired initialization + fixed final
    compile(tree, str(TRAINER), 'exec')
    return tree


def source_audit():
    tree = adapted_tree()
    return dict(schema='hey-prefix-fixed-final-adapter-source-audit-v2',
                sampler_policy='hey-prefix-preserving-v2',
                crop_rule_sha256=sha(Path(__file__).with_name('hey_prefix_crop_v2.py')),
                original_trainer_sha256=sha(TRAINER), adapter_sha256=sha(Path(__file__)),
                transformed_source_sha256=hashlib.sha256(ast.unparse(tree).encode()).hexdigest(),
                checkpoint_policy='fixed-final-20000', validation_calls=0,
                deterministic_ops=True, tf32=False, training_launched=False,
                limits='AST construction/compile only; no target pack, TensorFlow execution, gradient, export or runtime certification.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-source', type=Path)
    p.add_argument('--training-lock', type=Path)
    p.add_argument('--target', choices=('hey_willow','hey_nyx'))
    p.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    a = p.parse_args()
    if a.audit_source:
        if a.training_lock or a.target:
            p.error('Source audit cannot launch')
        result = source_audit()
        with a.audit_source.open('x') as f:
            f.write(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))
        return
    if not a.training_lock or not a.target:
        p.error('Launch needs training-lock and target')
    lock_path = a.training_lock.resolve().relative_to(Path.cwd())
    remote_bytes = subprocess.check_output(['git','show',
                        'origin/research-hye-nyx-voice-gpu:'+str(lock_path)])
    if remote_bytes != a.training_lock.read_bytes():
        raise ValueError('Training lock is not pushed with identical bytes')
    lock = json.loads(remote_bytes)
    if lock['schema'] != 'hey-paired-training-lock-v2':
        raise ValueError('Unknown training lock')
    if lock.get('stage')!='formal':
        raise ValueError('Formal training requires both actual integration smokes')
    for package,version in lock['versions'].items():
        if importlib.metadata.version(package)!=version:raise ValueError('Training dependency changed: '+package)
    # The exact lock commit must already exist on the remote research branch.
    commit = lock['material_commit']
    subprocess.run(['git','merge-base','--is-ancestor',commit,
                    'origin/research-hye-nyx-voice-gpu'],check=True)
    for name, digest in lock['files'].items():
        if sha(Path(name)) != digest:
            raise ValueError(f'Locked file changed: {name}')
    for path in (TRAINER, Path(__file__), Path(__file__).with_name('validate_hey_target_pack_v1.py'),
                 Path(__file__).with_name('hey_batch_schedule_v2.py'),
                 Path(__file__).with_name('audit_hey_training_batch_v2.py'),
                 Path(__file__).with_name('hey_prefix_crop_v2.py'),
                 Path(__file__).with_name('hey_batch_schedule_v1.py'),
                 Path(__file__).with_name('audit_hey_training_batch_v1.py')):
        if lock['files'].get(str(path.relative_to(Path.cwd()))) != sha(path):
            raise ValueError('Training code not bound by lock')
    for path in TRAINER.parent.glob('*.py'):
        if lock['files'].get(str(path.relative_to(Path.cwd()))) != sha(path):
            raise ValueError('Training helper not bound by lock')
    arm = lock['arms'][a.target]
    root = Path(arm['features'])
    mandatory = [root/n for n in ('features.bin','features.json','human.json','ambient.json','target-provenance.json')]
    mandatory += [Path(lock['initial_weights']),Path(arm['batch_schedule']),
                  Path(arm['cpu_audit']),Path(arm['schedule_receipt']),Path(arm['integration_smoke'])]
    for path in mandatory:
        key = str(path.resolve().relative_to(Path.cwd()))
        if lock['files'].get(key) != sha(path):
            raise ValueError('Input not explicitly bound by training lock: '+key)
    admission=validate(root, a.target, required_splits=('training',))
    cpu=json.loads(Path(arm['cpu_audit']).read_text())
    scheduled=json.loads(Path(arm['schedule_receipt']).read_text())
    if (cpu['target_admission']['target_id']!=a.target or not cpu['target_admission']['admitted']
            or cpu['target_admission']['sidecar_sha256']!=admission['sidecar_sha256']
            or cpu['trainer_sha256']!=sha(TRAINER) or cpu['batch_shape']!=[128,150,40]
            or not cpu['deterministic_repeat']):
        raise ValueError('Actual target CPU batch admission missing/mismatched')
    if (scheduled['target']!=a.target or scheduled['batches']!=20100
            or scheduled['schedule_sha256']!=sha(Path(arm['batch_schedule']))
            or scheduled['trainer_sha256']!=sha(TRAINER)
            or scheduled['target_provenance_sha256']!=admission['sidecar_sha256']
            or scheduled['schedule_generator_sha256']!=sha(Path(__file__).with_name('hey_batch_schedule_v2.py'))
            or scheduled['auditor_sha256']!=sha(Path(__file__).with_name('audit_hey_training_batch_v2.py'))):
        raise ValueError('Complete precomputed batch schedule admission missing/mismatched')
    if cpu.get('sampler_policy')!='hey-prefix-preserving-v2' or scheduled.get('sampler_policy')!='hey-prefix-preserving-v2':
        raise ValueError('Wrong v2 sampler identity')
    policy_sha=sha(Path(__file__).with_name('hey_prefix_crop_v2.py'))
    if cpu['crop_rule_sha256']!=policy_sha or scheduled['crop_rule_sha256']!=policy_sha:
        raise ValueError('Changed v2 crop rule')
    trace=Path(scheduled['positive_crop']['trace_path'])
    key=str(trace.resolve().relative_to(Path.cwd()))
    if lock['files'].get(key)!=sha(trace) or sha(trace)!=scheduled['positive_crop']['trace_sha256']:
        raise ValueError('Expected positive draw trace not bound')
    if scheduled['positive_crop']['positive_draws']!=643200:raise ValueError('Incomplete positive draw schedule')
    first=json.loads(Path(arm['batch_schedule']).read_text().splitlines()[0])
    if first['tensor_sha256']!=cpu['batch_sha256']:
        raise ValueError('CPU audit and schedule first batch differ')
    rows = json.loads((root/'features.json').read_text())
    other = sum((json.loads((root/n).read_text()) for n in ('human.json','ambient.json')),[])
    if any(r['split'] != 'training' for r in rows+other):
        raise ValueError('Only training rows permitted; development/acceptance stay external')
    if lock['steps'] != 20000 or lock['seed'] != 726 or lock['batch_size'] != 128:
        raise ValueError('Pilot recipe mismatch')
    sys.path.insert(0,str(TRAINER.parent))
    sys.argv = [str(TRAINER), '--features',str(root), '--output',arm['output'],
                '--upstream',lock['upstream'], '--initial-weights',lock['initial_weights'],
                '--architecture','teacher60','--context-frames','150',
                '--batchnorm-policy','frozen','--steps','20000','--eval-interval','5000',
                '--source-exponent','0','--ambient-sampling','rows','--device',a.device]
    guard=BatchGuard(Path(arm['batch_schedule']),Path(arm['output']))
    tree = adapted_tree()
    crop=CropGuard(Path(arm['output'])/'actual-positive-crops.jsonl',scheduled['positive_crop'])
    initial_hook=lambda model,output:capture_initial(model,output,Path(arm['integration_smoke']).parent/'smoke-initial-inventory.json')
    exec(compile(tree,str(TRAINER),'exec'),dict(__name__='__main__',__file__=str(TRAINER), HEY_TARGET_ID=a.target, HEY_BATCH_GUARD=guard, HEY_PREFIX_CROP=crop, HEY_INITIAL_CAPTURE=initial_hook))
    output = Path(arm['output'])
    report = json.loads((output/'training.json').read_text())
    assert report['selected_step'] == 20000
    report.update(target_id=a.target,checkpoint_policy='fixed-final-20000',
                  actual_initial_inventory_sha256=sha(output/'formal-initial-inventory.json'),
                  actual_initial_values_match_certified_smoke=True,
                  adapter_sha256=sha(Path(__file__)),training_lock_sha256=sha(a.training_lock))
    (output/'training.json').write_text(json.dumps(report,indent=2)+'\n')
    if a.target == 'hey_willow':
        (output/'hey_nyx.tflite').rename(output/'hey_willow.tflite')


if __name__ == '__main__':
    main()
