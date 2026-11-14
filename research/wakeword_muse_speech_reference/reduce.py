"""Reduce complete Muse UART independently of the hardware capture process."""
import argparse
import importlib.util
import json
from pathlib import Path
from fixtures import sha

spec = importlib.util.spec_from_file_location('muse_baseline_protocol', Path(__file__).resolve().parent.parent / 'wakeword_baseline/run.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
validate_ready = baseline.validate_ready
validate_control_initialization = baseline.validate_control_initialization
validate_control_result = baseline.validate_control_result
CONTROL_FIELD = baseline.CONTROL_FIELD

DEFAULT_BITS = '3f1b367a'
PREFIXES = ('WAKE_BENCH_READY', 'WAKE_BENCH_BAUD', 'WAKE_BENCH_CLIP_READY', 'WAKE_BENCH_EVENT', 'WAKE_BENCH_RESULT')


def messages(raw):
    assert raw.endswith(b'\n'), 'Incomplete final UART line'
    result, started = [], False
    for line in raw.splitlines():
        assert not any(marker in line for marker in (b'WAKE_BENCH_ERROR', b'Guru Meditation', b'abort()')), 'Firmware failure'
        if not started:
            start = line.find(b'WAKE_BENCH_READY ')
            if start >= 0:
                line = line[start:]
                started = True
            else:
                assert b'WAKE_BENCH_' not in line, 'Unexpected protocol before initial READY'
                continue
        assert b'WAKE_BENCH_' not in line or line.startswith(b'WAKE_BENCH_'), 'Displaced protocol after initial sync'
        if line.startswith(b'WAKE_BENCH_'):
            key, text = line.decode('ascii').split(' ', 1)
            assert key in PREFIXES
            result.append((key, json.loads(text)))
    return result


def reduce_capture(raw, references):
    records = messages(raw)
    index = 0
    def take(kind):
        nonlocal index
        assert index < len(records) and records[index][0] == kind, (index, kind)
        value = records[index][1]
        index += 1
        return value
    ready = take('WAKE_BENCH_READY')
    validate_ready(ready, 'wn10_himuse', True, True)
    assert ready['expanded_model_bytes'] == 430762
    if references['uart_baud'] != 115200:
        assert take('WAKE_BENCH_BAUD') == dict(baud=references['uart_baud'])
        assert take('WAKE_BENCH_READY') == ready
    cases, repeats = [], []
    for entry in references['fixtures']['rows']:
        current = []
        for repeat in range(2):
            clip_id = len(cases)
            row = dict(entry['original'], **{CONTROL_FIELD: None})
            prepared = take('WAKE_BENCH_CLIP_READY')
            validate_control_initialization(prepared, row, clip_id)
            control = prepared['control']
            assert all(control[key] == DEFAULT_BITS for key in ('default_getter_f32_hex', 'getter_after_set_f32_hex', 'getter_after_prime_f32_hex'))
            events = []
            while index < len(records) and records[index][0] == 'WAKE_BENCH_EVENT':
                events.append(take('WAKE_BENCH_EVENT'))
            result = take('WAKE_BENCH_RESULT')
            validate_control_result(result, events, row, clip_id, 512)
            assert result['crc32'] == entry['pcm_crc32']
            assert 0 < result['max_us'] <= result['total_us'] <= result['calls'] * result['max_us']
            for key in ('heap_before_create', 'heap_after_create', 'heap_after_prime'):
                heap = prepared[key]
                assert set(heap) == {'internal_free', 'psram_free', 'internal_largest', 'psram_largest'}
                assert all(type(n) is int and n >= 0 for n in heap.values())
            case = dict(id=clip_id, repeat=repeat, original=entry['original'], initialization=prepared, events=events, result=result)
            current.append(case)
            cases.append(case)
        def logical(case):
            return dict(prime=case['initialization']['control']['prime_returns'],
                        threshold=case['initialization']['control']['getter_after_prime_f32_hex'],
                        result={k: v for k, v in case['result'].items() if k not in ('id', 'total_us', 'max_us')},
                        events=[{k: v for k, v in event.items() if k != 'id'} for event in case['events']])
        repeats.append(logical(current[0]) == logical(current[1]))
    assert take('WAKE_BENCH_BAUD') == dict(baud=references['uart_baud'])
    assert take('WAKE_BENCH_READY') == ready
    assert index == len(records), 'Unexpected trailing protocol records'
    assert len(cases) == 216 and len(repeats) == 108
    groups = []
    for literal in references['fixtures']['selected_literals']:
        for repeat in range(2):
            selected = [case for case in cases if case['original']['literal'] == literal and case['repeat'] == repeat]
            groups.append(dict(literal=literal, repeat=repeat, clips=len(selected), clips_with_events=sum(bool(case['events']) for case in selected),
                               total_events=sum(len(case['events']) for case in selected)))
    return dict(schema='muse-consumed108-uart-reduction-v1', capture_sha256=sha(raw),
        reference_binding_sha256=sha(json.dumps(references, sort_keys=True).encode()),
        ready=ready, cases=cases, repeat_exact=repeats, all_repeats_exact=all(repeats), groups=groups,
        input_invocations=sum(case['result']['calls'] for case in cases), prime_invocations=216 * 63,
        max_detect_us=max(case['result']['max_us'] for case in cases),
        setter_called=False, threshold_selected=False, candidate_qualified=False,
        limits='DEFAULT-only complete consumed synthetic input. Event counts describe full-clip presence, not complete first-utterance delivery, background FAR or mature-model parity.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'references', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = reduce_capture(args.capture.read_bytes(), json.loads(args.references.read_text()))
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('input_invocations', 'all_repeats_exact', 'max_detect_us', 'groups')}))


if __name__ == '__main__':
    main()
