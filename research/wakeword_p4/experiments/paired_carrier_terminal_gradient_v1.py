"""Training-only fixed terminal gradient probe; no optimizer or model selection."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

R = Path('research/wakeword_p4/results')
LOCK = R / '4090-paired-carrier-terminal-gradient-plan-v1.json'


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def plan():
    training = R / '4090-binary40-paired-carrier-training-plan-v1.json'
    files = dict(json.loads(training.read_text())['files'])
    files[str(training)] = sha(training)
    files[str(Path(__file__))] = sha(__file__)
    for arm in ['control', 'paired']:
        for name in ['final.pt', 'batches.jsonl.gz']:
            path = Path(f'.local/wakeword/binary40-paired-carrier-training-v1/{arm}/{name}')
            files[str(path)] = sha(path)
    spec = dict(files=files, arms=['control', 'paired'], sampler_seed=726,
                batches=[1, 2], optimizer_updates=0,
                groups=['unchanged120', 'replaced_original8', 'paired_positive4', 'paired_negative4'],
                policy='Both frozen final20000 models, first2 actual original/paired training batches. '
                'Authenticate sampler tensors/descriptors against original logs. Original BCE weights '
                'and 1/128 gradient denominator. Save every group gradient and score. No mining, '
                'coefficient selection, development replay, optimization or causal/generalization claim.')
    with LOCK.open('x') as f: json.dump(spec, f, indent=2); f.write('\n')


def run():
    assert subprocess.check_output(['git', 'show', 'origin/research-hye-nyx-voice-gpu:'+str(LOCK)]) == LOCK.read_bytes()
    p = json.loads(LOCK.read_text())
    for path, h in p['files'].items(): assert sha(path) == h, path
    import torch
    from causal_source_balanced_batch_v1 import Sampler as Original, digest
    from causal_paired_carrier_batch_v1 import Sampler as Paired
    from paired_carrier_batch_audit_v1 import audit
    sys.path.insert(0, str(Path('research/wakeword_p4/training').resolve()))
    from causal_binary40_span_v1 import Binary40Span, CONTROL
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
    models = {}
    for arm in p['arms']:
        cp = torch.load(f'.local/wakeword/binary40-paired-carrier-training-v1/{arm}/final.pt', map_location='cpu', weights_only=False)
        assert cp['step'] == 20000
        model = Binary40Span(CONTROL).cuda().eval(); model.load_state_dict(cp['model'], strict=True)
        models[arm] = (model, cp)
    samplers = dict(control=Original(726), paired=Paired(726))
    logs = {a: gzip.open(f'.local/wakeword/binary40-paired-carrier-training-v1/{a}/batches.jsonl.gz', 'rt') for a in p['arms']}
    arrays, rows = {}, []
    for batch in p['batches']:
        draws = {a: s.batch() for a, s in samplers.items()}
        audit(*draws['control'], *draws['paired'])
        for a, (values, descriptors) in draws.items():
            log = json.loads(next(logs[a])); assert log['step'] == batch and log['descriptors'] == descriptors
            assert [digest(v) for v in values] == [log[k] for k in ['features_sha256', 'positive_sha256', 'negative_sha256']]
        descriptors = draws['paired'][1]
        ps = [i for i, d in enumerate(descriptors) if d['kind'] == 'paired_carrier_positive']
        ns = [i for i, d in enumerate(descriptors) if d['kind'] == 'paired_carrier_nick_negative']
        assert len(ps) == len(ns) == 4
        unchanged = [i for i in range(128) if i not in ps+ns]
        for arm, (model, cp) in models.items():
            grads = {}; losses = {}
            for source, names in [('control', {'replaced_original8': ps+ns}),
                                  ('paired', {'unchanged120': unchanged, 'paired_positive4': ps, 'paired_negative4': ns})]:
                x, pw, nw = draws[source][0]
                z, _ = model(torch.from_numpy(x).permute(0, 2, 1).cuda(), model.initial_state(128, 'cuda'))
                z = z[:, 0, :]
                per_row = (torch.nn.functional.softplus(-z)*torch.from_numpy(pw).cuda()
                           +20*torch.nn.functional.softplus(z)*torch.from_numpy(nw).cuda()).sum(1)
                arrays[f'b{batch}_{arm}_{source}_probability'] = torch.sigmoid(z).detach().cpu().numpy()
                arrays[f'b{batch}_{arm}_{source}_loss'] = per_row.detach().cpu().numpy()
                for name, slots in names.items():
                    loss = per_row[slots].sum()/128
                    g = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True)
                    flat = torch.cat([v.flatten() for v in g]).detach().cpu().numpy()
                    assert np.isfinite(flat).all()
                    grads[name] = flat; losses[name] = float(loss.detach())
                    arrays[f'b{batch}_{arm}_{name}_gradient'] = flat
                del z, per_row
            names = list(grads); pairs = []
            for i, a in enumerate(names):
                for b in names[i+1:]:
                    x, y = grads[a].astype(np.float64), grads[b].astype(np.float64)
                    denom = float(np.linalg.norm(x)*np.linalg.norm(y))
                    pairs.append(dict(a=a, b=b, cosine=float(x@y/denom) if denom else None))
            assert all(torch.equal(v.cpu(), cp['model'][k]) for k, v in model.state_dict().items())
            rows.append(dict(batch=batch, arm=arm, group_losses=losses,
                             group_gradient_norms={k: float(np.linalg.norm(v.astype(np.float64))) for k,v in grads.items()}, pairs=pairs))
    for f in logs.values(): f.close()
    dest = R / '4090-paired-carrier-terminal-gradient-arrays-v1.npz'
    with dest.open('xb') as f: np.savez_compressed(f, **arrays)
    out = dict(plan_sha256=sha(LOCK), gpu=torch.cuda.get_device_name(), optimizer_updates=0,
               weights_unchanged=True, rows=rows, arrays=dict(path=str(dest), sha256=sha(dest)), policy=p['policy'])
    with (R / '4090-paired-carrier-terminal-gradient-original-v1.json').open('x') as f:
        json.dump(out, f, indent=2); f.write('\n')
    print(json.dumps(out))


if __name__ == '__main__': {'plan': plan, 'run': run}[sys.argv[1]]()
