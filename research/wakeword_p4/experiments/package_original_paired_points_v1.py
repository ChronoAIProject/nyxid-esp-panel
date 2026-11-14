"""Hand off authenticated existing diagnostic bytes; no inference or rescoring."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path('research/wakeword_p4')
SOURCE_COMMIT = '1e64ab8d7214aca8491fd7de1e0f3a998f455309'
def sha(data):
    return hashlib.sha256(data).hexdigest()
def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode()
def main():
    audit_path = ROOT / 'results/4090-hey-common-carrier-raw-scoring-independent-audit-v1.json'
    audit_bytes = audit_path.read_bytes()
    assert audit_bytes == subprocess.check_output(['git', 'show', f'{SOURCE_COMMIT}:{audit_path}'])
    audit = json.loads(audit_bytes)
    payload = {'metadata/' + audit_path.name: audit_bytes}
    sources = {'metadata/' + audit_path.name: str(audit_path)}
    for model in audit['models']:
        path = Path('.local/wakeword/hey-common-carrier-raw-development-v1/diagnostic') / (model['model'] + '.jsonl')
        data = path.read_bytes()
        assert sha(data) == model['records_sha256']
        rows = [json.loads(line) for line in data.splitlines()]
        assert len(rows) == model['points'] == 138
        assert all(len(row['records']) == model['complete_inputs_per_point'] == 372 for row in rows)
        member = 'diagnostic/' + path.name
        payload[member] = data
        sources[member] = str(path)
    manifest = {'schema': 'gpu-original-paired-points-handoff-v1', 'source_commit': SOURCE_COMMIT,
                'points': 552, 'inputs_per_point': 372,
                'limits': 'Original complete reused diagnostic bytes only. No new inference, selection, regeneration, fresh-development or acceptance access. Generated boundaries are not phonetic ground truth.',
                'members': {name: {'bytes': len(data), 'sha256': sha(data), 'source_path': sources[name], 'source_commit': SOURCE_COMMIT} for name, data in payload.items()}}
    payload['manifest.json'] = encode(manifest)
    output = Path('.local/wakeword/gpu-original-paired-points-v1.tar.gz')
    with tarfile.open(output, 'w:gz') as archive:
        for name, data in payload.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))
    receipt = {'archive': str(output), 'bytes': output.stat().st_size, 'sha256': sha(output.read_bytes()), 'manifest': manifest,
               'url': 'https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/' + output.name}
    (ROOT / 'results/4090-original-paired-points-handoff-v1.json').write_bytes(encode(receipt))
    (ROOT / 'results/gpu-original-paired-points-manifest-v1.json').write_bytes(payload['manifest.json'])
    print(json.dumps({k: v for k, v in receipt.items() if k != 'manifest'}))
if __name__ == '__main__':
    main()
