"""Recover missing training-only PCM ledgers by exact original frontend replay."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path('research/wakeword_p4/training').resolve()))
from features import features
from pymicro_features import MicroFrontend


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(row):
    corpus = Path('.local/wakeword/LibriSpeech/dev-clean')
    base = Path('.local/wakeword/features-gpu-ami-extra-v1')
    files = sorted((corpus / str(row['speaker'])).rglob('*.flac'))
    assert len(files) == row['files'] and row['split'] == 'training'
    frontend = MicroFrontend()
    stream = hashlib.sha256()
    original_samples = 0
    total_samples = 0
    arrays, sources = [], []
    for path in files:
        pcm, rate = sf.read(path, dtype='int16')
        assert rate == 16000 and pcm.ndim == 1
        digest = hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()
        sources.append(dict(path=str(path), encoded_sha256=sha(path),
                            pcm_sha256=digest, samples=len(pcm)))
        original_samples += len(pcm)
        padded = np.concatenate([pcm, np.zeros(3200, np.int16)])
        stream.update(padded.astype('<i2').tobytes())
        total_samples += len(padded)
        arrays.append(features(padded, frontend))
    actual = np.concatenate(arrays)
    expected = np.load(base / row['path'], allow_pickle=False)
    equal = bool(actual.shape == expected.shape and np.array_equal(actual, expected))
    assert original_samples / 16000 == row['source_seconds']
    assert total_samples / 16000 == row['seconds']
    assert len(actual) == row['frames']
    return dict(speaker=row['speaker'], training_feature_path=row['path'],
        original_samples=original_samples, stream_samples=total_samples,
        source_files=len(files), sources=sources,
        reconstructed_stream_pcm_sha256=stream.hexdigest(),
        stored_feature_sha256=sha(base / row['path']),
        all_frontend_values_exact=equal,
        mismatching_feature_values=(int(np.count_nonzero(actual != expected))
                                    if actual.shape == expected.shape else None))


def main():
    base = Path('.local/wakeword/features-gpu-ami-extra-v1')
    previous_path = Path('research/wakeword_p4/results/4090-new-background-training-overlap-v1.json')
    previous = json.loads(previous_path.read_text())
    assert sha(base / 'human.json') == previous['current_training_index_sha256']['human.json']
    wanted = {r['speaker'] for r in previous['human_rows_without_stream_pcm_hash']}
    rows = [r for r in json.loads((base / 'human.json').read_text())
            if r['split'] == 'training' and r['speaker'] in wanted]
    assert len(rows) == len(wanted) == 24
    output = Path('.local/wakeword/legacy-training-human-pcm-audit-v1')
    output.mkdir(exist_ok=False)
    completed = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(audit, rows):
            (output / f"{result['speaker']}.json").write_text(json.dumps(result, indent=2) + '\n')
            completed.append(result)
            print(json.dumps(dict(completed=len(completed), speaker=result['speaker'],
                exact=result['all_frontend_values_exact'])), flush=True)
    target_path = Path('.local/wakeword/prospective-background-whole-waveform-v1/manifest.json')
    targets = json.loads(target_path.read_text())['records']
    new_pcm = {r['pcm_sha256'] for r in targets}
    new_encoded = {r['sha256'] for r in targets}
    source_pcm = {s['pcm_sha256'] for r in completed for s in r['sources']}
    source_encoded = {s['encoded_sha256'] for r in completed for s in r['sources']}
    streams = {r['reconstructed_stream_pcm_sha256'] for r in completed}
    report = dict(script_sha256=sha(Path(__file__)), original_frontend_source_sha256=sha(Path('research/wakeword_p4/training/features.py')),
        prior_overlap_report_sha256=sha(previous_path), target_manifest_sha256=sha(target_path),
        speakers=len(completed), source_files=sum(r['source_files'] for r in completed),
        all_frontend_values_exact=all(r['all_frontend_values_exact'] for r in completed),
        original_source_pcm_overlap=sorted(source_pcm & new_pcm),
        original_encoded_overlap=sorted(source_encoded & new_encoded),
        reconstructed_stream_overlap=sorted(streams & new_pcm), records=completed,
        training_data_changed=False, detector_scored=False, final_audio_accessed=False,
        limits='Newly reconstructed source/stream hashes, not recovered historical same-instance hashes. '
        'Source sample ledgers and entire saved feature arrays compared exactly. Frontend is not invertible, '
        'so matching features do not uniquely prove historical raw PCM identity. No transformed/partial '
        'or person-level independence claim, no new-source scoring authorization.')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}), flush=True)


if __name__ == '__main__': main()
