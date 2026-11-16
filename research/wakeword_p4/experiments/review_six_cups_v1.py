"""Independent transcript diagnostic of the flagged sentence's three seeds."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

R = Path('research/wakeword_p4/results')
PLAN = R / '4090-six-cups-asr-plan-v2.json'
DATA = Path('.local/wakeword/chatterbox-daily-language-v1')
MODEL = Path('.local/wakeword/whisper-small-en')
RESULT = R / '4090-six-cups-asr-original-v2.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def plan():
    rows = [json.loads((DATA / f'{i:03d}.json').read_text()) for i in [69, 70, 71]]
    assert all(row['request']['text'] == 'We need six cups of water.' for row in rows)
    evidence = R / '4090-daily-language-pair-result-v1.json'
    paths = [Path(__file__), evidence, Path('research/wakeword_p4/training/gpu_python.py'),
             Path('.local/wakeword/whisper-small-en-source.json')]
    paths += [p for p in MODEL.iterdir() if p.is_file()]
    for row in rows:
        for item in row['files']:
            assert sha(item['path']) == item['sha256']
            paths.append(Path(item['path']))
    record = dict(files={str(p): sha(p) for p in paths}, rows=rows,
        versions={n: importlib.metadata.version(n) for n in ['faster-whisper', 'ctranslate2', 'numpy', 'soundfile']},
        config=dict(language='en', beam_size=3, condition_on_previous_text=False,
                    temperature=0, vad_filter=False, word_timestamps=True),
        policy='All three already-consumed development seeds of flagged request, no new generation or selection. Existing pinned small.en CUDA float16, no prompt. Word timestamps and transcripts automatic, not phonetic truth. No training, threshold changes or human claim.')
    with PLAN.open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')


def run():
    assert subprocess.check_output(['git', 'show', 'origin/research-hye-nyx-voice-gpu:' + str(PLAN)]) == PLAN.read_bytes()
    assert not RESULT.exists()
    p = json.loads(PLAN.read_text())
    for path, expected in p['files'].items():
        assert sha(path) == expected
    for name, expected in p['versions'].items():
        assert importlib.metadata.version(name) == expected
    import soundfile as sf
    from faster_whisper import WhisperModel
    model = WhisperModel(str(MODEL), device='cuda', compute_type='float16', cpu_threads=2, num_workers=1)
    assert model.model.device == 'cuda'
    events = json.loads((R / '4090-daily-language-pair-result-v1.json').read_text())['records']
    rows = []
    for source in p['rows']:
        wav, = [item['path'] for item in source['files'] if item['path'].endswith('.wav')]
        pcm, rate = sf.read(wav, dtype='float32')
        assert rate == 16000 and pcm.ndim == 1
        segments, info = model.transcribe(pcm, **p['config'])
        segments = [segment._asdict() for segment in segments]
        row = dict(index=source['index'], request=source['request'], wav_sha256=sha(wav),
                   segments=segments, transcript=' '.join(s['text'] for s in segments).strip(),
                   model_events=events[source['index']]['models'])
        rows.append(row)
        print(json.dumps(row), flush=True)
    RESULT.write_text(json.dumps(dict(plan_sha256=sha(PLAN), records=rows,
        runtime_device=model.model.device, compute_type='float16', phonetic_truth=False), indent=2) + '\n')


if __name__ == '__main__':
    {'plan': plan, 'run': run}[sys.argv[1]]()
