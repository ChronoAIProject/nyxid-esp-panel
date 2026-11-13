"""Compare the new executor's unprimed path with the authenticated P4-style baseline."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime', 'data', 'model', 'saved-study', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert sha(args.model) == '04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c'
    assert sha(args.saved_study / 'report.json') == 'de27257b414ffa6eb90ce8aad5649310662e1db18bb93febfcf1f94c097184ee'
    assert sha(args.saved_study / 'scores.npz') == '61e2fbfb31fdec528d9f572b9302c6d75267a49be524689f20ba2038dcf87711'
    source_hashes = {}
    for name in ('evaluate.py', 'features.py'):
        expected = subprocess.check_output(['git', 'show', '84a6004:research/wakeword_p4/training/' + name])
        assert (args.runtime / name).read_bytes() == expected
        source_hashes[name] = sha(args.runtime / name)
    executor = subprocess.check_output(['git', 'show', '84a6004:research/wakeword_p4/experiments/score_joe_fixed_baselines_v1.py'])
    assert b'raw=features(pcm);detector.reset();scores=smooth(detector.predict(raw),5)' in executor
    assert b'nyx_times=[float((3*(j+4)+5)*.01)' in executor
    sys.path.insert(0, str(args.runtime.resolve()))
    from evaluate import Detector, event_indices, smooth
    from features import features
    baseline = json.loads((args.saved_study / 'report.json').read_text())
    saved = np.load(args.saved_study / 'scores.npz', allow_pickle=False)
    assert len(baseline['records']) == 126
    # This private diagnostic plan is written before constructing any detector.
    args.output.mkdir(parents=True)
    plan = dict(source_sha256=sha(Path(__file__)), runtime_sources=source_hashes,
        executor_sha256=hashlib.sha256(executor).hexdigest(), corpus='All126 consumed Kristin/Norman development WAVs',
        old_path='56160 zero prime, complete PCM, final padding to480; raw outputs must reproduce saved UINT8 exactly',
        new_path='Complete PCM, features() pads to160; reset, original raw predict and five-score smooth',
        threshold=.91796875, no_threshold_selection=True, new_data_access=False,
        audio_playback_or_recording=False, training=False)
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    detector = Detector(args.model)
    rows, traces = [], {}
    for index, original in enumerate(baseline['records']):
        row = original['record']
        path = args.data / row['path']
        assert sha(path) == row['sha256']
        pcm, rate = sf.read(path, dtype='int16')
        assert rate == 16000 and len(pcm) == row['samples']
        assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest() == row['pcm_sha256']
        old_raw = features(np.pad(pcm, (56160, (-len(pcm)) % 480)))
        detector.reset()
        old = detector.predict(old_raw)
        old_bytes = np.rint(old * 256).astype(np.uint8)
        np.testing.assert_array_equal(old_bytes, saved[f'output_{index}'])
        board_events = [(j + 4 + 2) * 480 - 56160 for j in event_indices(smooth(old, 5), .91796875)]
        assert board_events == original['firmware_policy']['offsets']
        new_raw = features(pcm)
        detector.reset()
        new = detector.predict(new_raw)
        new_bytes = np.rint(new * 256).astype(np.uint8)
        np.testing.assert_array_equal(new, new_bytes.astype(np.float32) / 256)
        smoothed = smooth(new, 5)
        assert len(smoothed) == ((len(pcm) + 159) // 160 - 2) // 3 - 4
        events = event_indices(smoothed, .91796875)
        feature_offsets = [(3 * (j + 4) + 5) * 160 for j in events]
        board_offsets = [(j + 4 + 2) * 480 for j in events]
        overlap = min(len(new_bytes), len(old_bytes) - 117)
        changed_scores = int(np.count_nonzero(new_bytes[:overlap] != old_bytes[117:117 + overlap]))
        traces[f'new_raw_output_{index}'] = new_bytes
        rows.append(dict(path=row['path'], voice=row['voice'], target=row['target'],
            kind=original['native_phrase_id'], condition=row['condition'],
            old_events=board_events, new_executor_feature_end_events=feature_offsets,
            new_same_policy_board_block_events=board_offsets,
            old_output_count=len(old_bytes), new_output_count=len(new_bytes),
            aligned_raw_outputs_compared=overlap, changed_aligned_uint8_outputs=changed_scores,
            old_480_pad=(-len(pcm)) % 480, new_frontend_160_pad=(-len(pcm)) % 160,
            new_last_model_block_end=(len(new_bytes) + 1) * 480,
            original_source_samples=len(pcm),
            old_frontend_sha256=hashlib.sha256(old_raw.tobytes()).hexdigest(),
            new_frontend_sha256=hashlib.sha256(new_raw.tobytes()).hexdigest()))
        if (index + 1) % 21 == 0:
            print(json.dumps(dict(completed=index + 1, planned=126)), flush=True)
    np.savez_compressed(args.output / 'scores.npz', **traces)
    changes = [r['path'] for r in rows if bool(r['old_events']) != bool(r['new_executor_feature_end_events'])]
    def counts(field):
        return {target: dict(clips=sum(r['target'] == target for r in rows),
            activated=sum(r['target'] == target and bool(r[field]) for r in rows)) for target in sorted({r['target'] for r in rows})}
    result = dict(schema='v30-consumed-lifecycle-replay-comparison-v1', plan_sha256=sha(args.output / 'plan.json'),
        scores_sha256=sha(args.output / 'scores.npz'), old126_raw_outputs_and_events_reproduced_exactly=True,
        original_counts=counts('old_events'), new_executor_counts=counts('new_executor_feature_end_events'),
        changed_activation_clips=changes, changed_aligned_uint8_outputs=sum(r['changed_aligned_uint8_outputs'] for r in rows),
        theoretical_feature_end_is_160_samples_before_board_480_block_return=True,
        versions={p: importlib.metadata.version(p) for p in ('numpy', 'ai-edge-litert', 'pymicro-features')},
        records=rows, new_data_scored=False, training=False, accepted=False,
        limits='Post-hoc comparison on consumed development only, with fixed original model and threshold. '
               'It diagnoses replay lifecycle and timestamp differences, not an alternative tuned policy, '
               'root cause, new-source quality, physical phonetic latency or model promotion.')
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('old126_raw_outputs_and_events_reproduced_exactly',
        'original_counts', 'new_executor_counts', 'changed_activation_clips', 'changed_aligned_uint8_outputs')}))


if __name__ == '__main__':
    main()
