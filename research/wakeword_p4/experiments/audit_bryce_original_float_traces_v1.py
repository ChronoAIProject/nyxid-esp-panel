"""Independently recount saved consumed126 float traces without model inference."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recount(values, source):
    # Use exact float32 values promoted to Python floats and fsum, not the
    # producer's numpy convolution or detector helper. Index raw callbacks.
    threshold = .91796875
    last, armed, offsets, means = -1000, True, [], []
    for end in range(4, len(values)):
        mean = math.fsum(float(v) for v in values[end - 4:end + 1]) / 5
        means.append(mean)
        if end - last >= 67:
            if mean < threshold / 2:
                armed = True
            if armed and mean >= threshold:
                offsets.append((end + 2) * 480 - 56160)
                armed, last = False, end
    padded_end = ((source['samples'] + 479) // 480) * 480
    callback_values = [float(v) for i, v in enumerate(values)
                       if (i + 2) * 480 > 56160]
    reference = source['generator_prefix_end_sample']
    return dict(offsets=offsets, original_threshold=threshold,
        any_source_event=any(1 <= x <= padded_end for x in offsets),
        reference_window_hit=any(reference - 1600 <= x <= reference + 12800 for x in offsets),
        events_before_source_start=sum(x < source['lead_samples'] for x in offsets),
        maximum_source_callback_raw_score=max(callback_values),
        maximum_five_mean=max(means),
        raw_output_sha256=hashlib.sha256(values.tobytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('study', 'baseline', 'protocol', 'output'):
        parser.add_argument('--' + arg, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert sha(args.protocol) == 'cb005c82f49cf4cdfc1a205a4959e6a04aee512cb24ac9b6547996e2a7f93312'
    assert sha(args.study / 'records.jsonl') == '6d8d5a6e8eb717d80384590044aadec397324ff5fa046c1791bdad8c8549508a'
    assert sha(args.study / 'scores.npz') == 'ec2d08f5f05b9fc7d86d32ae17d5ed58a839cbd3bbaf6fb2eaa12e8a9c3b1a88'
    assert sha(args.baseline / 'report.json') == 'de27257b414ffa6eb90ce8aad5649310662e1db18bb93febfcf1f94c097184ee'
    assert sha(args.baseline / 'scores.npz') == '61e2fbfb31fdec528d9f572b9302c6d75267a49be524689f20ba2038dcf87711'
    protocol = json.loads(args.protocol.read_text())
    report = json.loads((args.study / 'report.json').read_text())
    rows = [json.loads(line) for line in (args.study / 'records.jsonl').read_text().splitlines()]
    baseline = json.loads((args.baseline / 'report.json').read_text())['records']
    traces = np.load(args.study / 'scores.npz', allow_pickle=False)
    original = np.load(args.baseline / 'scores.npz', allow_pickle=False)
    arms = protocol['arms']
    assert len(rows) == len(baseline) == len(original.files) == 126
    assert len(traces.files) == 378
    assert set(traces.files) == {f'{arm}_{i}' for arm in arms for i in range(126)}
    summaries, audited = {}, []
    largest_mean_difference = 0.
    for i, row in enumerate(rows):
        prior = baseline[i]
        assert row['record'] == prior['record']
        assert row['native_phrase_id'] == prior['native_phrase_id']
        assert row['frontend_sha256'] == prior['frontend_sha256']
        new = {}
        for arm in arms:
            values = traces[f'{arm}_{i}']
            assert values.dtype == np.dtype('float32') and values.shape == (prior['output_count'],)
            assert np.isfinite(values).all() and np.all((0 <= values) & (values <= 1))
            if arm == 'retained_original_int8':
                np.testing.assert_array_equal(values, original[f'output_{i}'].astype(np.float32) / 256)
            recounted = recount(values, row['record'])
            recorded = row['arms'][arm]
            assert set(recounted) == set(recorded)
            for key in recounted:
                if key == 'maximum_five_mean':
                    difference = abs(recounted[key] - recorded[key])
                    largest_mean_difference = max(difference, largest_mean_difference)
                    assert difference <= 4e-16
                else:
                    assert recounted[key] == recorded[key], (i, arm, key)
            if arm == 'retained_original_int8':
                assert recounted['offsets'] == prior['firmware_policy']['offsets']
            new[arm] = recounted
        audited.append(dict(record=row['record'], native_phrase_id=row['native_phrase_id'], arms=new))
    for arm in arms:
        groups = {}
        for voice in ('kristin', 'norman'):
            for phrase in ('hey_nyx', 'hey_nyx_command'):
                items = [r['arms'][arm] for r in audited
                         if r['record']['voice'] == voice and r['native_phrase_id'] == phrase]
                groups[voice + '/' + phrase] = dict(clips=len(items),
                    any_source_event_clips=sum(x['any_source_event'] for x in items),
                    reference_window_hit_clips=sum(x['reference_window_hit'] for x in items),
                    raw_peak_range=[min(x['maximum_source_callback_raw_score'] for x in items),
                                    max(x['maximum_source_callback_raw_score'] for x in items)])
        positive = [r['arms'][arm] for r in audited if r['record']['target'] == 'hey_nyx']
        confusers = [r['arms'][arm] for r in audited if r['record']['target'] not in ('hey_nyx', 'hi_esp')]
        assert len(positive) == 36 and len(confusers) == 54
        summary = dict(positive_any_source_clips=sum(x['any_source_event'] for x in positive),
            positive_reference_window_hits=sum(x['reference_window_hit'] for x in positive),
            common_confuser_clips=sum(bool(x['offsets']) for x in confusers), groups=groups)
        assert summary == report['summaries'][arm]
        summaries[arm] = summary
    differences = []
    for row in audited:
        a, b, c = [row['arms'][arm] for arm in arms]
        if len({bool(x['offsets']) for x in (a, b, c)}) != 1:
            differences.append(dict(path=row['record']['path'], target=row['record']['target'],
                voice=row['record']['voice'], kind=row['native_phrase_id'],
                arms={arm: row['arms'][arm] for arm in arms}))
    result = dict(schema='mac-bryce-original-float-trace-independent-audit-v1',
        source_sha256=sha(Path(__file__)),
        artifacts_sha256={name: sha(args.study / name) for name in ('report.json', 'records.jsonl', 'scores.npz')},
        protocol_sha256=sha(args.protocol), clips=126, traces=378,
        source_record_and_frontend_hash_metadata_match_original126=True,
        retained_int8_arrays_match_original126=True, all_events_and_peak_fields_independently_recounted=True,
        all_summaries_match=True, largest_fsum_vs_convolution_peak_difference=largest_mean_difference,
        clip_kind_counts=dict(Counter(row['native_phrase_id'] for row in audited)),
        summaries=summaries, activation_classification_differences=differences,
        conclusion='The dominant Norman command miss exists in the authenticated original float graph: '
            '0/9 in both float arms and retained INT8 at the frozen operating point. Input rounding alone '
            'does not change classifications here. INT8 changes one positive and one confuser classification; '
            'it does not alone explain the dominant miss.',
        limits='Trace-only independent audit. No repeated inference, training, export, new data, playback, '
            'recording, threshold selection or acceptance. Producer checkpoint authentication and reset-repeat '
            'claims are not independently rerun by this trace audit. Consumed synthetic development does not '
            'establish phonetic truth, root cause or real-user generalization.', accepted=False)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(clips=126, traces=378, summaries_match=True,
        differences=[r['path'] for r in differences], output=str(args.output))))


if __name__ == '__main__':
    main()
