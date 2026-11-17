"""Prelocked six-stream float diagnostic against frozen paired-carrier INT8 traces."""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path
import numpy as np

R = Path('research/wakeword_p4/results')
B = Path('.local/wakeword/paired-carrier-float-context-v1')
LOCK = R / '4090-paired-carrier-float-context-plan-v1.json'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def plan():
    files = {}
    for arm in ['control', 'paired']:
        p = R / f'4090-binary40-paired-carrier-{arm}-context-plan-v1.json'
        files.update(read(p)['files'])
        paths = [p, R / f'4090-binary40-paired-carrier-{arm}-context-original-v1.json',
                 Path(f'.local/wakeword/binary40-paired-carrier-training-v1/{arm}/final.pt')]
        paths += [Path(s['trace']['path']) for s in read(paths[1])['streams']]
        files.update({str(p): sha(p) for p in paths})
    for folder in ['experiments', 'training']:
        for p in Path(f'research/wakeword_p4/{folder}').glob('*.py'):
            files[str(p)] = sha(p)
    spec = dict(schema='paired-carrier-float-context-plan-v1', files=files,
                arms=['control', 'paired'], streams=['builtin', 'melo-US', 'melo-BR'],
                chunk_frames=3000, optimizer_updates=0,
                policy='All six existing complete physical context streams only; no isolated '
                'Joe/background replay or claims about those causes. Frozen final20000 weights, '
                'CUDA4090 float inference, continuous81600zeroPCMprime, native480clock. '
                'Compare saved INT8 codes, no new export/calibration/training/synthesis, '
                'threshold or quota selection. Consumed diagnostic, not acceptance or P4 proof.')
    with LOCK.open('x') as f:
        json.dump(spec, f, indent=2)
        f.write('\n')


def run():
    assert subprocess.check_output(['git', 'show', 'origin/research-hye-nyx-voice-gpu:' + str(LOCK)]) == LOCK.read_bytes()
    p = read(LOCK)
    for path, expected in p['files'].items():
        assert sha(path) == expected, path
    subprocess.run(['/usr/bin/python3', 'research/wakeword_p4/tools/check_disk_budget.py', '--expected-gib', '.1'], check=True)
    import torch
    sys.path.insert(0, str(Path('research/wakeword_p4/training').resolve()))
    from causal_binary40_span_v1 import Binary40Span, CONTROL
    from features import features
    from collect_multispeaker_formal_v1 import prepare
    from run_parallel_context_streams_v1 import material
    from replay_binary40_float_v2 import float_events
    from uint8_event_policy_v1 import event_steps
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
    B.mkdir(exist_ok=False)
    rows = []
    streams = read(R / '4090-parallel-context-streams-plan-v1.json')['streams']
    for arm in p['arms']:
        cp = torch.load(f'.local/wakeword/binary40-paired-carrier-training-v1/{arm}/final.pt', map_location='cpu', weights_only=False)
        assert cp['step'] == 20000
        model = Binary40Span(CONTROL).cuda().eval()
        model.load_state_dict(cp['model'], strict=True)
        assert all(torch.equal(v.cpu(), cp['model'][k]) for k, v in model.state_dict().items())
        probe = torch.from_numpy(np.random.default_rng(726).normal(size=(1, 40, 300)).astype('float32')).cuda()
        with torch.inference_mode():
            full, fs = model(probe, model.initial_state(device='cuda'))
            state = model.initial_state(device='cuda'); parts = []
            for i in range(0, 300, 3):
                y, state = model(probe[:, :, i:i+3], state); parts.append(y)
            parity = float((full - torch.cat(parts, 2)).abs().max())
            assert parity < 1e-4
            assert all(torch.allclose(a, b, atol=1e-4, rtol=1e-5) for a, b in zip(fs, state))
        saved = read(R / f'4090-binary40-paired-carrier-{arm}-context-original-v1.json')['streams']
        assert [s['name'] for s in saved] == p['streams']
        for spec, ref in zip(streams, saved, strict=True):
            assert spec['name'] == ref['name']
            pcm, timeline = material(spec, 8000)
            assert timeline == ref['timeline']
            whole = np.r_[np.zeros(81600, np.int16), pcm]
            values, support, _ = prepare(features(whole), len(whole))
            clock = np.minimum(((support + 479)//480)*480, len(whole)) - 81600
            with np.load(ref['trace']['path'], allow_pickle=False) as z:
                assert np.array_equal(clock, z['relative_block_return'])
                codes = z['uint8'].copy()
            assert len(codes) == len(values)//3
            state = model.initial_state(device='cuda'); parts = []
            with torch.inference_mode():
                for i in range(0, len(values), p['chunk_frames']):
                    y, state = model(torch.from_numpy(values[i:i+p['chunk_frames']].T.copy()[None]).cuda(), state)
                    parts.append(torch.sigmoid(y[0, 0]).cpu().numpy())
            prob = np.concatenate(parts)
            assert len(prob) == len(codes) and np.isfinite(prob).all()
            events = {k: [dict(step=int(s), relative_sample=int(clock[s-1])) for s in steps]
                      for k, steps in [('float', float_events(prob)), ('int8', event_steps(codes, 235/256))]}
            assert events['int8'] == ref['events']
            clips = []
            for clip in timeline:
                if clip['kind'] != 'clip': continue
                hits = {k: any(clip['start'] < e['relative_sample'] <= clip['end'] for e in ev) for k, ev in events.items()}
                selected = (clock > clip['start']) & (clock <= clip['end'])
                clips.append(dict(clip=clip, hits=hits, peak_float=float(prob[selected].max())))
            trace = B / f'{arm}-{ref["name"]}.npz'
            np.savez_compressed(trace, probability=prob, uint8=codes, relative_block_return=clock)
            row = dict(arm=arm, name=ref['name'], events=events, clips=clips,
                       native_parity_max_error=parity, calls=len(prob),
                       pcm_sha256=hashlib.sha256(whole.astype('<i2').tobytes()).hexdigest(),
                       trace=dict(path=str(trace), bytes=trace.stat().st_size, sha256=sha(trace)))
            rows.append(row)
            with (B / 'records.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            print(json.dumps(dict(arm=arm, stream=ref['name'], calls=len(prob))), flush=True)
    archive = R / '4090-paired-carrier-float-context-evidence-v1.tar.gz'
    with tarfile.open(archive, 'x:gz') as t:
        for path in sorted(B.iterdir()): t.add(path, arcname=path.name)
    result = dict(schema='paired-carrier-float-context-original-v1', plan_sha256=sha(LOCK),
                  rows=rows, gpu=torch.cuda.get_device_name(), optimizer_updates=0,
                  archive=dict(path=str(archive), bytes=archive.stat().st_size, sha256=sha(archive)),
                  policy=p['policy'], independent_review_pending=True)
    with (R / '4090-paired-carrier-float-context-original-v1.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')


if __name__ == '__main__':
    {'plan': plan, 'run': run}[sys.argv[1]]()
