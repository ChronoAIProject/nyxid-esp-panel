"""Broad225 AST adapter construction; launch requires separate execution admission."""
import ast
import hashlib
import json
from pathlib import Path

TRAINER = Path(__file__).resolve().parents[1]/'training/train_stream.py'
PIN = '2228dff55d11111c3e68a4a27433ef8b472a91c3d185f124a9018425b5f03904'


def adapted_tree():
    assert hashlib.sha256(TRAINER.read_bytes()).hexdigest() == PIN
    tree = ast.parse(TRAINER.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    parse = next(i for i,n in enumerate(main.body) if ast.unparse(n) == 'args = parser.parse_args()')
    main.body[parse+1:parse+1] = ast.parse('''
HEY_ADMIT_ARGS(args)
tf.config.experimental.enable_op_determinism()
tf.config.experimental.enable_tensor_float_32_execution(False)
''').body
    # Original initial_weights CLI must remain None until after its150-only block.
    bn = next(i for i,n in enumerate(main.body) if ast.unparse(n) == 'bn_layers = apply_policy(model, bn_policy)')
    main.body[bn:bn] = ast.parse('''
assert args.initial_weights is None
HEY_LOAD_INITIAL(model)
head_weights = [w.copy() for w in heads[0].get_weights()]
head_sha = hashlib.sha256(b''.join(w.tobytes() for w in head_weights)).hexdigest()
initial_sha = HEY_INITIAL_SHA
''').body
    prefix = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == 'prefix_sample')
    assert ast.unparse(prefix.body[0]) == "sources = groups['training'][1]"
    prefix.body[0] = ast.parse("sources = {e: r for e, r in groups['training'][1].items() if e in ('piper', 'kokoro')}").body[0]
    loop = next(n for n in ast.walk(main) if isinstance(n,ast.For) and ast.unparse(n.target)=='step'
                and ast.unparse(n.iter)=='range(1, args.steps + 1)')
    block = next(n for n in loop.body if isinstance(n,ast.If) and 'args.eval_interval' in ast.unparse(n.test))
    assert 'selected, rank, grid = validate()' in ast.unparse(block)
    block.body = ast.parse('''
item = dict(step=step, loss=loss, checkpoint_policy='fixed-final', seconds=round(time.monotonic()-started,1))
history.append(item)
log.write(json.dumps(item)+'\\n')
log.flush()
print(json.dumps(item), flush=True)
if step == args.steps:
    selected_step = step
    model.save_weights(root/'best.weights.h5')
''').body
    batch = next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='batch')
    returns = [n for n in ast.walk(batch) if isinstance(n,ast.Return)]
    assert len(returns)==1 and isinstance(returns[0].value,ast.Tuple)
    returns[0].value = ast.Call(func=ast.Name(id='HEY_BATCH_GUARD',ctx=ast.Load()),args=returns[0].value.elts,keywords=[])
    report = next(i for i,n in enumerate(main.body) if isinstance(n,ast.Expr) and 'training.json' in ast.unparse(n) and '.write_text(' in ast.unparse(n))
    main.body[report:report] = ast.parse('''
report.update(classifier_reset=False, classifier_initialization='loaded exact context225 shared snapshot; no reset',
    checkpoint_policy='fixed-final', validation_detection='not run; external development',
    selection='Fixed final checkpoint; no development selection',
    batch_schedule=HEY_BATCH_GUARD.finish(), broad_context225=True)
''').body
    assert not [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='validate']
    return ast.fix_missing_locations(tree)


def load_initial(model):
    report = json.loads(Path('research/wakeword_p4/results/4090-hey-broad-context225-initialization-v1.json').read_text())
    snapshot = Path(report['snapshot']['path'])
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == report['snapshot']['sha256']
    model.load_weights(snapshot)
    expected = report['variable_inventory']['variables']
    assert len(model.variables) == len(expected) and model.count_params() == 24301
    for variable, entry in zip(model.variables, expected):
        value = variable.numpy()
        assert list(value.shape) == entry['shape'] and str(value.dtype) == entry['dtype']
        assert hashlib.sha256(value.tobytes()).hexdigest() == entry['sha256']
    return report['snapshot']['sha256']


def admit_args(args):
    assert args.initial_weights is None
    assert args.architecture=='teacher60' and args.context_frames==225 and args.batchnorm_policy=='frozen'
    assert args.source_exponent==0 and args.ambient_sampling=='stratified' and args.prefix_negatives==16
    assert args.negative_candidates==1 and not args.mask_prefix_loss
    assert args.time_mask_width==0 and args.frequency_mask_width==0
    # Full option-by-option formal/smoke admission is supplied by the execution lock,
    # not inferred from these core checks. This module has no training launch CLI.


if __name__ == '__main__':
    tree = adapted_tree(); compile(tree,str(TRAINER),'exec')
    result = dict(schema='hey-broad-context225-adapter-source-audit-v1',
        trainer_sha256=PIN, adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        transformed_sha256=hashlib.sha256(ast.unparse(tree).encode()).hexdigest(),
        validation_calls=0, original_trainer_unchanged=True,
        limits='AST compile only; no launch entry, initial-load hook, actual GPU, schedule or export admission yet.')
    with Path('research/wakeword_p4/results/4090-hey-broad-adapter-source-audit-v1.json').open('x') as f:
        f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
