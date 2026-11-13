"""Attribute existing v30 development misses without loading audio or a model.

This is a post-hoc diagnostic of consumed development, not a new detector policy.
The original threshold, five-score mean and latch/refractory remain unchanged.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


REPORT_SHA = 'de27257b414ffa6eb90ce8aad5649310662e1db18bb93febfcf1f94c097184ee'
MANIFEST_SHA = '31adf5d810a50662591da86d1be7a50fbee6042b8e26fc8d091b4ddfb089e909'
MODEL_SHA = '04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(scores, reference):
    assert scores.dtype == np.uint8 and scores.ndim == 1
    assert len(scores) == reference['output_count']
    assert hashlib.sha256(scores.tobytes()).hexdigest() == reference['quantized_output_sha256']
    values = scores.astype(np.int64)
    sums = np.convolve(values, np.ones(5, dtype=np.int64), mode='valid')
    # Independently check every convolution sum against direct integer addition.
    assert sums.tolist() == [sum(int(x) for x in scores[i:i + 5]) for i in range(len(scores) - 4)]
    offsets, prime, armed, last = [], [], True, -1000
    for step, total in enumerate(sums, 5):
        if 2 * total < 1175 and step - last >= 67:
            armed = True
        if armed and total >= 1175 and step - last >= 67:
            armed, last = False, step
            offset = (step + 1) * 480 - 56160
            (offsets if offset > 0 else prime).append(offset)
    assert offsets == reference['firmware_policy']['offsets']
    assert prime == reference['firmware_policy']['prime_event_offsets'] == []
    assert reference['host_policy'] == reference['firmware_policy']
    # Callback offsets are the retained application's block clock, not phonetic
    # endpoint measurements or finer frontend frame-end timestamps.
    times = (np.arange(len(scores)) + 2) * 480 - 56160
    raw_max = int(values[times > 0].max())
    mean_max_sum = int(sums[times[4:] > 0].max())
    hit = bool(offsets)
    if hit:
        category = 'event'
    elif raw_max < 235:
        category = 'no_single_score_reaches_original_threshold'
    elif mean_max_sum < 1175:
        category = 'single_score_reaches_threshold_but_five_score_mean_does_not'
    else:
        category = 'five_score_crossing_without_event'
    return dict(raw_max_uint8=raw_max, five_score_max_sum_uint8=mean_max_sum,
                original_event=hit, category=category,
                source_callback_outputs_all_quantized_zero=raw_max == 0)


def summarize(rows):
    return dict(clips=len(rows), original_event_clips=sum(r['original_event'] for r in rows),
                single_score_threshold_reached=sum(r['raw_max_uint8'] >= 235 for r in rows),
                five_score_threshold_reached=sum(r['five_score_max_sum_uint8'] >= 1175 for r in rows),
                categories=dict(Counter(r['category'] for r in rows)),
                all_quantized_zero_clips=sum(r['source_callback_outputs_all_quantized_zero'] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'source-manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    report_path = args.study / 'report.json'
    assert sha(report_path) == REPORT_SHA and sha(args.source_manifest) == MANIFEST_SHA
    report = json.loads(report_path.read_text())
    manifest = json.loads(args.source_manifest.read_text())
    native = {r['key']: r for r in manifest['native_records']}
    assert report['data_manifest_sha256'] == MANIFEST_SHA and report['model_sha256'] == MODEL_SHA
    assert sha(args.study / 'scores.npz') == report['scores_sha256']
    assert report['threshold'] == 235 / 256 and not report['threshold_selected']
    assert len(report['records']) == len(manifest['records']) == 126
    rows = []
    with np.load(args.study / 'scores.npz', allow_pickle=False) as traces:
        assert set(traces.files) == {f'output_{i}' for i in range(126)}
        for index, (reference, source) in enumerate(zip(report['records'], manifest['records'], strict=True)):
            assert reference['record'] == source
            assert reference['prime_samples'] == 56160
            result = analyze(traces[f'output_{index}'], reference)
            phrase = native[source['source_key']]
            target = source['target']
            group = ('hey_nyx' if target == 'hey_nyx' else 'opposite_hi_esp'
                     if target == 'hi_esp' else 'common_confuser')
            rows.append(dict(path=source['path'], source_key=source['source_key'],
                             voice=source['voice'], condition=source['condition'], group=group,
                             phrase=phrase['id'],
                             generator_prefix_duration_ms=phrase['generator_prefix_end_sample'] * 1000 / phrase['source_rate'],
                             **result))
    positives = [r for r in rows if r['group'] == 'hey_nyx']
    output = dict(
        schema='bryce-v30-saved-score-miss-diagnostic-v1',
        source_sha256=sha(Path(__file__)), report_sha256=REPORT_SHA,
        data_manifest_sha256=MANIFEST_SHA, model_sha256=MODEL_SHA,
        saved_scores_sha256=report['scores_sha256'],
        all_126_saved_event_sequences_reproduced=True,
        threshold_uint8=235, five_score_threshold_sum_uint8=1175,
        threshold=235 / 256, smoothing_steps=5, refractory_steps=67,
        by_group={g: summarize([r for r in rows if r['group'] == g]) for g in sorted({r['group'] for r in rows})},
        positive_by_voice_and_phrase={v + '/' + p: summarize([r for r in positives if r['voice'] == v and r['phrase'] == p])
                                      for v in ('kristin', 'norman') for p in ('hey_nyx', 'hey_nyx_command')},
        records=rows, model_inference=False, audio_loaded=False, threshold_search=False,
        detector_policy_changed=False, training=False, final_data_accessed=False,
        accepted=None,
        limits=[
            'Post-hoc analysis of already consumed synthetic development, not an independent experiment.',
            'Single-score maxima are a miss-attribution diagnostic, not a proposed operating point. No new background policy was evaluated.',
            'A zero UINT8 output is a quantization-floor observation, not proof of zero float probability, silence or an audio capture defect.',
            'Bare and command utterances have different native synthesis durations and acoustics. They are not a controlled identical-prefix/suffix intervention.',
            'Generator token durations are not verified human phonetic boundaries; three render variants share each native source.',
            'No current fixed/per_draw candidate, official Hi ESP, P4, acoustic or long-call inference was performed.'
        ])
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(by_group=output['by_group'], positive_by_voice_and_phrase=output['positive_by_voice_and_phrase'])))


if __name__ == '__main__':
    main()
