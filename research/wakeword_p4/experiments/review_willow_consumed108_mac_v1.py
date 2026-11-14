"""Check the new reducer using retained native Willow records, without hardware."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import run_willow_consumed108_mac_v1 as runner


def encode(records):
    return b'cut historical boot prefix' + b'\r\n'.join(k.encode() + b' ' + json.dumps(v).encode() for k, v in records) + b'\r\n'


def check(path):
    raw = path.read_bytes()
    original = json.loads(raw)
    published = json.loads((runner.REPO / 'research/wakeword_p4/results/official-hey-consumed-speech-grid-native-v1.json').read_text())
    assert runner.sha(raw) == published['original_report_sha256']
    source_rows = original['clips'][:12]
    assert len(source_rows) == 12 and all(r['reference_threshold_f32_hex'] is None for r in source_rows)
    ready = original['firmware_ready']
    baud = 921600
    records = [('WAKE_BENCH_READY', ready), ('WAKE_BENCH_BAUD', dict(baud=baud)), ('WAKE_BENCH_READY', ready)]
    rows = []
    for row in source_rows:
        source = {k: v for k, v in row.items() if k not in ('initialization', 'raw_events', 'result', 'threshold')}
        source['literal'] = row['text']
        rows.append(dict(original=source, pcm_crc32=row['result']['crc32']))
        for _ in range(2):
            clip_id = sum(k == 'WAKE_BENCH_RESULT' for k, _ in records)
            initialization, result, events = deepcopy(row['initialization']), deepcopy(row['result']), deepcopy(row['raw_events'])
            initialization['id'] = result['id'] = clip_id
            for event in events:
                event['id'] = clip_id
            records.append(('WAKE_BENCH_CLIP_READY', initialization))
            records.extend(('WAKE_BENCH_EVENT', e) for e in events)
            records.append(('WAKE_BENCH_RESULT', result))
    records.extend([('WAKE_BENCH_BAUD', dict(baud=baud)), ('WAKE_BENCH_READY', ready)])
    references = dict(uart_baud=baud, fixtures=dict(rows=rows, selected_literals=list(dict.fromkeys(r['original']['literal'] for r in rows))))
    result = runner.reduce_capture(encode(records), references)
    assert len(result['cases']) == 24 and result['all_repeats_exact']
    assert all(c['result']['samples'] == source_rows[i // 2]['samples'] for i, c in enumerate(result['cases']))
    def rejected(records, refs=references):
        try:
            runner.reduce_capture(encode(records), refs)
        except (AssertionError, ValueError, KeyError, RuntimeError):
            return True
        return False
    checks = []
    wrong = deepcopy(records); wrong[0][1]['engine'] = 'wn10_himuse'; checks.append(rejected(wrong))
    wrong = deepcopy(records); next(v for k, v in wrong if k == 'WAKE_BENCH_CLIP_READY')['control']['getter_after_prime_f32_hex'] = '3f1b367a'; checks.append(rejected(wrong))
    wrong = deepcopy(records); next(v for k, v in wrong if k == 'WAKE_BENCH_RESULT')['crc32'] ^= 1; checks.append(rejected(wrong))
    wrong = deepcopy(records); next(v for k, v in wrong if k == 'WAKE_BENCH_EVENT')['sample_end'] += 1; checks.append(rejected(wrong))
    checks.append(rejected(deepcopy(records[:-1])))
    wrong = deepcopy(records); wrong.append(('WAKE_BENCH_BAUD', dict(baud=baud))); checks.append(rejected(wrong))
    wrong = deepcopy(references); wrong['fixtures']['rows'][0]['original']['samples'] += 1; checks.append(rejected(records, wrong))
    assert len(checks) == 7 and all(checks)
    return dict(schema='willow-consumed108-reducer-review-v1',
        source_sha256=runner.sha(Path(__file__).read_bytes()), runner_sha256=runner.sha(Path(runner.__file__).read_bytes()),
        original_native_report_sha256=runner.sha(raw), original_native_rows_used=12,
        constructed_repeated_cases=24, constructed_malformed_inputs_rejected=len(checks), all_passed=True,
        device_operated=False, new_inference=False, audio_accessed=False,
        limits='Constructed transport/reducer checks from authenticated old native records, not new Semaine input inference or recognition evidence.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-native-report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = check(args.original_native_report)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
