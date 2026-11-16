"""Four coherent long synthetic references from existing training-only voices."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

PLAN = Path('research/wakeword_p4/results/4090-chatterbox-training-reference-plan-v1.json')
OUT = Path('.local/wakeword/chatterbox-training-references-v1')
PARTITION = Path('.local/wakeword/hey-nyx/kokoro/generation.json')
MODEL = Path('.local/wakeword/kokoro-models/kokoro-v1.0.onnx')
VOICES = MODEL.with_name('voices-v1.0.bin')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock():
    assert sha(MODEL) == 'beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a'
    assert sha(VOICES) == 'bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d'
    splits = json.loads(PARTITION.read_text())['voice_partitions']
    chosen = [next(v for v in splits['training'] if v.startswith(prefix))
              for prefix in ['af_', 'am_', 'bf_', 'bm_']]
    assert not set(chosen) & set(splits['validation'] + splits['testing'])
    text = ('The morning light comes through the window as I prepare breakfast. '
            'Later today I plan to walk through the park, visit the library, '
            'and return home before the evening rain begins.')
    plan = dict(files={str(p): sha(p) for p in [Path(__file__), PARTITION, MODEL, VOICES]},
        voices=chosen, text=text, speed=1.0,
        versions={name: importlib.metadata.version(name) for name in
                  ['kokoro-onnx', 'onnxruntime', 'numpy', 'soundfile']},
        policy='One coherent utterance per training-only conditioning identity, no looping or concatenation. CPU ONNX preprocessing limited2threads; downstream Chatterbox synthesis and wake model training on4090. Reused synthetic identities, not new independent human speakers. Mechanical QA does not prove text correctness. All four retained, no score-based selection; fail if <=5seconds.')
    with PLAN.open('x') as stream:
        stream.write(json.dumps(plan, indent=2) + '\n')


def run():
    assert subprocess.check_output(['git', 'show',
        'origin/research-hye-nyx-voice-gpu:' + str(PLAN)]) == PLAN.read_bytes()
    plan = json.loads(PLAN.read_text())
    for name, expected in plan['versions'].items():
        assert importlib.metadata.version(name) == expected
    for path, expected in plan['files'].items():
        assert sha(Path(path)) == expected
    import numpy as np
    import onnxruntime as ort
    import soundfile as sf
    from kokoro_onnx import Kokoro
    class Session(ort.InferenceSession):
        def run(self, output_names, input_feed, run_options=None):
            return super().run(output_names,
                {**input_feed, 'speed': np.asarray([1.0], dtype=np.float32)}, run_options)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = Session(str(MODEL), options, providers=['CPUExecutionProvider'])
    engine = Kokoro.from_session(session, str(VOICES))
    OUT.mkdir(exist_ok=False)
    rows = []
    for voice in plan['voices']:
        wave, rate = engine.create(plan['text'], voice=voice, speed=1.0,
                                  lang='en-gb' if voice.startswith('b') else 'en-us')
        wave = np.asarray(wave, dtype=np.float32)
        assert wave.ndim == 1 and np.isfinite(wave).all() and len(wave) / rate > 5
        path = OUT / (voice + '.wav')
        sf.write(path, wave, rate, subtype='FLOAT')
        row = dict(voice=voice, split='training', path=str(path), sample_rate=int(rate),
                   samples=len(wave), seconds=len(wave)/rate, peak=float(np.max(np.abs(wave))),
                   sha256=sha(path), bytes=path.stat().st_size, text=plan['text'])
        rows.append(row)
        (OUT / (voice + '.json')).write_text(json.dumps(row, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    (OUT / 'result.json').write_text(json.dumps(dict(plan_sha256=sha(PLAN),
        records=rows, phonetic_review=False, role='training-conditioning-only'), indent=2) + '\n')


if __name__ == '__main__':
    {'lock': lock, 'run': run}[sys.argv[1]]()
