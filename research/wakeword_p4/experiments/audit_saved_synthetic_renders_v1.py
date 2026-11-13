"""Validate consumed historical rendering fixtures; never generate or score new data."""
import importlib.metadata
import json
from pathlib import Path
import soundfile as sf
from synthetic_output_quality_v1 import sha
from synthetic_render_quality_v1 import audit_render, RIR_DELAYS, RIR_AMPLITUDES


def main():
    base = Path('.local/wakeword')
    source = base / 'slr83-development-v1'
    root = base / 'slr83-development-replay-v2'
    manifest = root / 'manifest.json'
    report = json.loads(manifest.read_text())
    assert sha(source / 'manifest.jsonl') == report['source_manifest_sha256']
    assert report['rir'] == dict(delays_samples=RIR_DELAYS, amplitudes=RIR_AMPLITUDES)
    originals = {r['sha256']: r for r in map(json.loads, (source / 'manifest.jsonl').read_text().splitlines())}
    details = []
    for row in report['records']:
        original = originals[row['source_sha256']]
        path = source / original['path']
        assert sha(path) == original['sha256']
        assert sha(root / row['path']) == row['sha256']
        dry, rate = sf.read(path, dtype='int16')
        pcm, out_rate = sf.read(root / row['path'], dtype='int16')
        assert rate == out_rate == 16000
        import hashlib
        assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest() == row['pcm_sha256']
        seed = int(hashlib.sha256((original['path'] + f":{row['variant']}:slr83-dev-v1").encode()).hexdigest()[:16], 16)
        assert seed == row['seed']
        details.append(dict(path=row['path'], **audit_render(dry, pcm, row)))
    assert len(details) == 90
    result = dict(manifest_sha256=sha(manifest), clips=len(details), passed=sum(d['passed'] for d in details),
        validator_sha256=sha(Path(__file__).with_name('synthetic_render_quality_v1.py')),
        script_sha256=sha(Path(__file__)), versions={n: importlib.metadata.version(n) for n in ['numpy', 'soundfile']},
        details=details, new_audio_generated=False, wakeword_inference=False,
        limits='Consumed historical fixtures only. Direct convolution independently checks saved FFT renders, '
        'full source/RIR boundaries, seed/noise/gain and PCM values; not new development/acceptance or phonetic truth. '
        'No source role, training data, threshold or current execution pins changed.')
    output = Path('research/wakeword_p4/results/4090-saved-synthetic-render-quality-v1.json')
    with output.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(dict(clips=result['clips'], passed=result['passed'])))


if __name__ == '__main__': main()
