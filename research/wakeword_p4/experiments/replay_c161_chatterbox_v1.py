"""Full 36-request diagnostic at original C161 event profile, no tuning."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from collect_multispeaker_formal_v1 import prepare, predict
from uint8_event_policy_v1 import event_steps

DATA = Path('.local/wakeword/chatterbox-development-v1')
OUT = Path('.local/wakeword/c161-chatterbox-v1')
PLAN = Path('research/wakeword_p4/results/4090-c161-chatterbox-replay-lock-v1.json')
MODEL = Path('.local/wakeword/causal-source-balanced-int8-v1/int8.tflite')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock():
    assert MODEL.stat().st_size == 60952
    assert sha(MODEL) == 'c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81'
    completed = DATA / 'completed.json'
    assert json.loads(completed.read_text())['requests'] == 36
    rows = [json.loads((DATA / f'{index:03d}.json').read_text()) for index in range(36)]
    paths = [MODEL, completed, Path(__file__),
             Path(__file__).with_name('collect_multispeaker_formal_v1.py'),
             Path(__file__).with_name('uint8_event_policy_v1.py'),
             Path('research/wakeword_p4/training/features.py')]
    for index, row in enumerate(rows):
        assert row['index'] == index
        paths.append(DATA / f'{index:03d}.json')
        for item in row['files']:
            path = Path(item['path'])
            assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
            paths.append(path)
    record = dict(files={str(path): sha(path) for path in paths}, rows=rows,
                  versions={name: importlib.metadata.version(name) for name in
                            ['numpy', 'soundfile', 'ai-edge-litert', 'pymicro-features']},
                  threshold=.91796875, smoothing=5, refractory=67, prime=81600,
                  scope='All36 full waveforms; Nyx target, Willow reciprocal diagnostic, everyday negatives. Continuous zero prime, no boundary reset, fresh model/frontend per input. No phoneme timing truth, threshold search, background duration or human generalization claim.')
    with PLAN.open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')


def run():
    assert subprocess.check_output(['git', 'show',
        'origin/research-hye-nyx-voice-gpu:' + str(PLAN)]) == PLAN.read_bytes()
    plan = json.loads(PLAN.read_text())
    for path, expected in plan['files'].items():
        assert sha(path) == expected, path
    for name, expected in plan['versions'].items():
        assert importlib.metadata.version(name) == expected
    import soundfile as sf
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    sys.path.insert(0, str(Path('research/wakeword_p4/training').resolve()))
    from features import features
    def runtime(**kwargs):
        return Interpreter(num_threads=1,
            experimental_op_resolver_type=OpResolverType.BUILTIN_REF, **kwargs)
    OUT.mkdir(exist_ok=False)
    records = []
    for row in plan['rows']:
        wav, = [Path(item['path']) for item in row['files'] if item['path'].endswith('.wav')]
        pcm, rate = sf.read(wav, dtype='int16')
        assert rate == 16000 and pcm.ndim == 1 and len(pcm) > 0
        combined = np.concatenate((np.zeros(81600, np.int16), pcm))
        values, support, _ = prepare(features(combined), len(combined))
        _, codes, quantization = predict(values, str(MODEL), runtime)
        blocks = np.minimum(((support + 479) // 480) * 480, len(combined)) - 81600
        events = [dict(step=step, relative_sample=int(blocks[step-1]),
                       finish=bool(support[step-1] > len(combined)))
                  for step in event_steps(codes, .91796875)]
        trace = OUT / f'{row["index"]:03d}.npz'
        np.savez_compressed(trace, uint8=codes, relative_block_return=blocks, support=support)
        result = dict(index=row['index'], request=row['request'], events=events,
                      real_events=[event for event in events if event['relative_sample'] > 0],
                      prime_events=[event for event in events if event['relative_sample'] <= 0],
                      trace_sha256=sha(trace), quantization=quantization,
                      pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest())
        records.append(result)
        (OUT / f'{row["index"]:03d}.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    summary = {group: dict(clips=sum(r['request']['group'] == group for r in records),
                 activated=sum(r['request']['group'] == group and bool(r['real_events']) for r in records))
               for group in ['nyx', 'willow', 'everyday-negative']}
    (OUT / 'result.json').write_text(json.dumps(dict(lock_sha256=sha(PLAN),
        summary=summary, records=records, acceptance=False), indent=2) + '\n')


if __name__ == '__main__':
    {'lock': lock, 'run': run}[sys.argv[1]]()
