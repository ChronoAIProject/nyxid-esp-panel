"""Package existing consumed PCM only; no synthesis or model inference."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import wave


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    source = Path('.local/wakeword/hey-development-replay-v1.json')
    original = source.read_bytes()
    expected = '2294fe10ba8a7a72174083e4308a3f055b6c14f4f7dc4ba8b59807c229526bc0'
    assert sha(original) == expected
    rows = json.loads(original)['records']
    assert len(rows) == 360
    payload = {'study/hey-development-replay-v1.json': original}
    selected = []
    counts = {}
    for index, row in enumerate(rows):
        if row['literal'] not in ('Hey Muse', 'Hey Willow', 'Hey Nicks'):
            continue
        path = Path(row['path'])
        data = path.read_bytes()
        assert sha(data) == row['sha256']
        with wave.open(io.BytesIO(data)) as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
            assert wav.getnframes() == row['samples']
            assert sha(wav.readframes(wav.getnframes())) == row['pcm_sha256']
        member = 'audio/' + path.name
        assert member not in payload
        payload[member] = data
        selected.append({'original_index': index, 'archive_path': member, 'original_row': row})
        counts[row['literal']] = counts.get(row['literal'], 0) + 1
    assert counts == {'Hey Willow': 36, 'Hey Nicks': 36, 'Hey Muse': 36}
    def encoded(value):
        return (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode()
    payload['study/selected-records.json'] = encoded(selected)
    manifest = {
        'schema': 'gpu-muse-consumed-pcm-handoff-v1',
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'original_manifest_sha256': expected,
        'counts': counts, 'records': len(selected),
        'limits': 'Previously consumed descriptive development PCM only. Original rows and manifest unchanged. No synthesis, model scoring, threshold selection, fresh development or acceptance access.',
        'members': {name: {'bytes': len(data), 'sha256': sha(data)} for name, data in payload.items()},
    }
    payload['manifest.json'] = encoded(manifest)
    output = Path('.local/wakeword/gpu-muse-consumed-pcm-v1.tar.gz')
    with tarfile.open(output, 'w:gz') as archive:
        for name, data in payload.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    result = {'archive': str(output), 'bytes': output.stat().st_size,
              'sha256': sha(output.read_bytes()), 'manifest': manifest}
    receipt = Path('research/wakeword_p4/results/4090-muse-consumed-pcm-handoff-v1.json')
    receipt.write_bytes(encoded(result))
    print(json.dumps({k: v for k, v in result.items() if k != 'manifest'}))


if __name__ == '__main__':
    main()
