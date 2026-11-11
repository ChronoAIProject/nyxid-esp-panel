"""Audit saved v30 traces, source identities and critical regression counts.

This is saved-output accounting, not independent inference or pronunciation.
The firmware-policy event check uses integer arithmetic on quantized bytes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zlib

import numpy as np
import soundfile as sf

from features import features


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trigger_steps(sums, threshold_sum):
    last, armed, steps = -1000, True, []
    for step, total in enumerate(sums, 5):
        if total*2 < threshold_sum and step-last >= 67:
            armed = True
        if armed and total >= threshold_sum and step-last >= 67:
            last, armed = step, False
            steps.append(step)
    return steps


def verify_group(records, target, summary):
    esp = {'Hi SP.', 'Hi ESB.', 'Hi ASP.', 'Hi EST.', 'I see ESP.', 'High speed.'}
    nyx = {row['text'] for row in records if row['target'] == 'none' and row['text'].startswith('Hey ')}
    definitions = {
        'positives_any_event': [row for row in records if row['target'] == target],
        'ordinary_and_near_word_negatives': [row for row in records if row['target'] == 'none'],
        'shared_negative_texts': [row for row in records if row['target'] == 'none' and row['text'] not in esp],
        'shared_ordinary_sentences': [row for row in records if row['target'] == 'none' and row['text'] not in esp|nyx],
        'nyx_specific_near_words': [row for row in records if row['target'] == 'none' and row['text'] in nyx],
        'hi_esp_specific_near_words_unvalidated_pronunciation': [row for row in records if row['target'] == 'none' and row['text'] in esp],
        'other_wake_phrase': [row for row in records if row['target'] not in (target, 'none')],
    }
    def counts(group):
        sizes = [len(row['result']['offsets']) for row in group]
        return {'clips': len(group), 'clips_with_event': sum(size > 0 for size in sizes), 'events': sum(sizes)}
    for key, selected in definitions.items():
        assert summary[key] == counts(selected), key
    positives = definitions['positives_any_event']
    window_hits = sum(any(-1600 <= at-row['keyword_end'] <= 12800
                          for at in row['result']['offsets']) for row in positives)
    assert summary['positives_in_estimated_end_window'] == {
        'clips': len(positives), 'detected': window_hits, 'window_ms': [-100, 800]}
    negative = definitions['ordinary_and_near_word_negatives']
    assert summary['negative_events_by_text'] == {
        text: counts([row for row in negative if row['text'] == text])
        for text in sorted({row['text'] for row in negative})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sources', 'artifacts', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    root = args.artifacts
    result = json.loads((root/'result.json').read_bytes())
    plan = json.loads((root/'plan.json').read_bytes())
    assert sha(root/'plan.json') == result['plan_sha256']
    assert sha(root/'source.py') == result['source_sha256']
    assert sha(args.sources/'manifest.jsonl') == result['manifest_sha256'] == plan['manifest_sha256']
    assert sha(args.sources/'baseline.json') == result['baseline_sha256'] == plan['baseline_sha256']
    manifest = [json.loads(line) for line in (args.sources/'manifest.jsonl').read_text().splitlines()]
    baseline = json.loads((args.sources/'baseline.json').read_bytes())
    assert len(manifest) == len(baseline['clips']) == len(result['source_details']) == 350
    assert result['accepted'] is False and result['all_reset_replays_exact']
    qfile = np.load(root/'scores.npz', allow_pickle=False)
    assert set(qfile.files) == {f'output_{i}' for i in range(350)}
    rows_host, rows_device, differences, values = [], [], [], 0
    for index, (item, row, official) in enumerate(zip(manifest, result['source_details'], baseline['clips'], strict=True)):
        assert all(row[k] == official[k] == v for k, v in item.items())
        path = args.sources/item['path']
        assert sha(path) == item['sha256']
        pcm, rate = sf.read(path, dtype='int16')
        assert rate == 16000 and pcm.ndim == 1 and len(pcm) == item['samples']
        assert zlib.crc32(pcm.astype('<i2').tobytes()) == official['result']['crc32']
        pad = (-len(pcm)) % 480
        assert row['prime_samples'] == plan['v30_prime_samples'] == 56160 and row['tail_zero_samples'] == pad
        front = features(np.pad(pcm, (56160, pad)).astype('<i2'))
        assert list(front.shape) == row['frontend_shape'] and str(front.dtype) == row['frontend_dtype']
        assert hashlib.sha256(front.tobytes()).hexdigest() == row['frontend_sha256']
        q = qfile[f'output_{index}']
        assert q.dtype == np.uint8 and q.ndim == 1
        assert len(q) == (56160+len(pcm)+pad)//480-1 == row['output_count']
        assert hashlib.sha256(q.tobytes()).hexdigest() == row['quantized_output_sha256']
        # Separate exact-integer firmware reference and the original float64
        # convolution. Neither imports the producer's event/smoothing helpers.
        totals = [sum(map(int, q[at:at+5])) for at in range(len(q)-4)]
        integer = trigger_steps(totals, 5*235)
        host_values = np.convolve(q.astype(np.float64)/256, np.ones(5)/5, mode='valid')
        host = trigger_steps(host_values, 235/256)
        for key, steps in [('host_convolution', host), ('firmware_arithmetic', integer)]:
            offsets = [(step+1)*480-56160 for step in steps]
            assert row[key] == {'events': sum(at > 0 for at in offsets),
                'offsets': [at for at in offsets if at > 0], 'one_based_score_steps': steps,
                'prime_event_offsets': [at for at in offsets if at <= 0]}
        assert row['policies_have_identical_events'] == (host == integer)
        if host != integer:
            differences.append(item['path'])
        rows_host.append({**item, 'result': row['host_convolution']})
        rows_device.append({**item, 'result': row['firmware_arithmetic']})
        values += len(q)
    assert differences == result['host_device_policy_differences']
    for key, rows, target in [('v30_host_convolution', rows_host, 'hey_nyx'),
                               ('v30_firmware_arithmetic', rows_device, 'hey_nyx'),
                               ('hiesp_retained_p4', baseline['clips'], 'hi_esp')]:
        verify_group(rows, target, result[key]['all'])
        for cohort, summary in result[key]['by_voice_and_variant'].items():
            voice, variant = cohort.split('/variant-')
            verify_group([r for r in rows if r['voice'] == voice and r['variant'] == int(variant)], target, summary)
    audit = {'scope': __doc__, 'passed': True, 'sources': 350, 'quantized_outputs_verified': values,
        'all_source_crcs_and_frontend_hashes_match': True,
        'all_integer_and_host_events_and_critical_cohort_counts_reconstruct': True,
        'host_device_event_differences': len(differences),
        'result_sha256': sha(root/'result.json'), 'scores_sha256': sha(root/'scores.npz'),
        'auditor_sha256': sha(Path(__file__)), 'accepted': False,
        'limits': 'Checks saved-output accounting and recomputes frontend bytes. Does not independently infer the model, reprove the original reset run, validate pronunciation, or establish MCU/acoustic/latency acceptance.'}
    with args.output.open('x') as out:
        out.write(json.dumps(audit, indent=2)+'\n')
    shutil.copyfile(__file__, root/'auditor-source.py')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
