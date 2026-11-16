"""Two fixed models on all96 common-language development clips."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from collect_multispeaker_formal_v1 import prepare, predict
from uint8_event_policy_v1 import event_steps

R = Path('research/wakeword_p4/results')
DATA = Path('.local/wakeword/chatterbox-daily-language-v1')
OUT = Path('.local/wakeword/daily-language-pair-v1')
PLAN = R / '4090-daily-language-pair-plan-v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock():
    assert json.loads((DATA / 'completed.json').read_text())['requests'] == 96
    rows = [json.loads((DATA / f'{i:03d}.json').read_text()) for i in range(96)]
    models = [dict(name='c161', path='.local/wakeword/causal-source-balanced-int8-v1/int8.tflite',
                   sha256='c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81'),
              dict(name='bc01', path='.local/wakeword/c161-chatterbox-int8-v1/int8.tflite',
                   sha256='bc011e24ff04f70c3911d39b6cc027a016d68c6fd793124d777ebf5f29edefa8')]
    paths = [Path(__file__), DATA / 'completed.json',
             Path(__file__).with_name('collect_multispeaker_formal_v1.py'),
             Path(__file__).with_name('uint8_event_policy_v1.py'),
             Path('research/wakeword_p4/training/features.py')]
    for model in models:
        assert sha(model['path']) == model['sha256']
        paths.append(Path(model['path']))
    for i, row in enumerate(rows):
        assert row['index'] == i and row['request']['role'] == 'development-only'
        paths.append(DATA / f'{i:03d}.json')
        for item in row['files']:
            assert sha(item['path']) == item['sha256']
            paths.append(Path(item['path']))
    record = dict(files={str(p): sha(p) for p in paths}, rows=rows, models=models,
        versions={n: importlib.metadata.version(n) for n in ['numpy', 'soundfile', 'ai-edge-litert', 'pymicro-features']},
        threshold=.91796875, smoothing_steps=5, prime=81600,
        policy='All96 once per fixed model, continuous81600zero prime and whole audio, no reset. Native480 clocks and rawUINT8 retained. Common-language and name conflicts diagnostic only, no FAR denominator/threshold selection/training/final access. Supplied text is intended content, not verified pronunciation.')
    with PLAN.open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')


def run():
    assert subprocess.check_output(['git', 'show', 'origin/research-hye-nyx-voice-gpu:' + str(PLAN)]) == PLAN.read_bytes()
    p = json.loads(PLAN.read_text())
    for path, expected in p['files'].items():
        assert sha(path) == expected
    for name, expected in p['versions'].items():
        assert importlib.metadata.version(name) == expected
    import soundfile as sf
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    sys.path.insert(0, str(Path('research/wakeword_p4/training').resolve()))
    from features import features
    def runtime(**kwargs):
        return Interpreter(num_threads=1, experimental_op_resolver_type=OpResolverType.BUILTIN_REF, **kwargs)
    OUT.mkdir(exist_ok=False)
    records = []
    for row in p['rows']:
        wav, = [item['path'] for item in row['files'] if item['path'].endswith('.wav')]
        pcm, rate = sf.read(wav, dtype='int16')
        assert rate == 16000 and pcm.ndim == 1
        combined = np.concatenate((np.zeros(81600, np.int16), pcm))
        values, support, _ = prepare(features(combined), len(combined))
        blocks = np.minimum(((support + 479)//480)*480, len(combined)) - 81600
        record = dict(index=row['index'], request=row['request'], models={})
        for model in p['models']:
            _, codes, q = predict(values, model['path'], runtime)
            events = [dict(step=s, relative_sample=int(blocks[s-1])) for s in event_steps(codes, .91796875)]
            trace = OUT / f'{row["index"]:03d}-{model["name"]}.npz'
            np.savez_compressed(trace, uint8=codes, relative_block_return=blocks, support=support)
            record['models'][model['name']] = dict(events=events,
                real_events=[e for e in events if e['relative_sample'] > 0],
                trace_sha256=sha(trace), quantization=q)
        records.append(record)
        print(json.dumps(record), flush=True)
    (OUT / 'result.json').write_text(json.dumps(dict(plan_sha256=sha(PLAN), records=records,
        completed=True, threshold_selected=False, pronunciation_validated=False), indent=2) + '\n')


if __name__ == '__main__':
    {'lock': lock, 'run': run}[sys.argv[1]]()
