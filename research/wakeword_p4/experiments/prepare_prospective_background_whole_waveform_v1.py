"""Extract all original development WAV bytes; scoring remains explicitly disabled."""
import hashlib
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

from download_prospective_background_development_v1 import EXPECTED_PLAN_SHA


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path('.local/wakeword/prospective-background-development-source-v1')
    plan_path = root / 'source-plan-original.json'
    assert sha(plan_path) == EXPECTED_PLAN_SHA
    plan = json.loads(plan_path.read_text())
    policy = Path('research/wakeword_p4/experiments/prospective_background_whole_waveform_protocol_v1.md')
    expected_shards = [r for r in plan['shards'] if r['role'] == 'development']
    audits = {}
    for entry, index in zip(expected_shards, (4, 5), strict=True):
        audit_path = root / f'test{index}-diagnostic-v2.json'
        audit = json.loads(audit_path.read_text())
        assert audit['shard'] == Path(entry['path']).name
        assert audit['source_sha256'] == EXPECTED_PLAN_SHA
        assert audit['all_replay_format_eligible'] and audit['diagnostic_only']
        assert not audit['background_qualified'] and not audit['wake_models_scored']
        shard = root / audit['shard']
        assert shard.stat().st_size == entry['size']
        assert sha(shard) == audit['shard_sha256'] == entry['sha256']
        audits[audit['shard']] = (audit_path, audit)
    output = Path('.local/wakeword/prospective-background-whole-waveform-v1')
    output.mkdir(exist_ok=False)
    (output / 'wav').mkdir()
    records = []
    try:
        for name, (_, audit) in audits.items():
            count = 0
            for index, batch in enumerate(pq.ParquetFile(root / name).iter_batches(batch_size=1)):
                row = batch.to_pylist()[0]
                expected = audit['records'][index]
                encoded = row['audio']['bytes']
                assert hashlib.sha256(encoded).hexdigest() == expected['encoded_sha256']
                filename = row['audio']['path']
                assert filename == Path(filename).name and filename.endswith('.wav')
                pcm, rate = sf.read(io.BytesIO(encoded), dtype='int16')
                assert pcm.ndim == 1 and rate == 16000 and len(pcm) == expected['frames']
                pcm_sha = hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()
                assert pcm_sha == expected['pcm16_sha256']
                destination = output / 'wav' / filename
                with destination.open('xb') as handle:
                    handle.write(encoded)
                assert sha(destination) == expected['encoded_sha256']
                records.append(dict(path='wav/' + filename, source_shard=name,
                    role='prospective-development-negative-unscored', samples=len(pcm),
                    rate=rate, seconds=len(pcm) / rate, sha256=sha(destination),
                    pcm_sha256=pcm_sha, annotation_times_valid=expected['annotation_times_valid'],
                    invalid_annotation_intervals=expected['invalid_annotation_intervals']))
                count += 1
            assert count == len(audit['records'])
        for field in ['path', 'sha256', 'pcm_sha256']:
            assert len({r[field] for r in records}) == len(records)
        hours = sum(r['samples'] for r in records) / 16000 / 3600
        assert len(records) == 42 and hours >= plan['minimum_background_hours_per_role']
        report = dict(records=records, recordings=len(records), source_hours=hours,
            plan_sha256=EXPECTED_PLAN_SHA, policy_sha256=sha(policy),
            script_sha256=sha(Path(__file__)), audits={n: sha(p) for n, (p, _) in audits.items()},
            all_encoded_wavs_preserved=True, all_decoded_samples_preserved=True,
            annotation_invalid_recordings=sum(not r['annotation_times_valid'] for r in records),
            integrity_preparation_passed=True, background_qualified=False,
            scoring_authorized=False, detector_scored=False, acceptance_audio_accessed=False,
            limits='Complete embedded mirror WAV integrity only. Invalid annotations retained; '
            'no annotation selection used. Not new scored exposure, all-training/final/person '
            'independence proof, source-qualified development or acceptance.')
        (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'records'}), flush=True)
    except Exception as error:
        (output / 'failure.json').write_text(json.dumps(dict(error_type=type(error).__name__,
            message=str(error), completed_recordings=len(records), automatic_retry=False), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
