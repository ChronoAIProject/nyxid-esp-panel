"""Compare all retained DEFAULT native events on identical consumed input bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import run_willow_consumed108_mac_v1 as native

BASE = 'research/wakeword_p4/results/'
FIELDS = ('sha256', 'pcm_sha256', 'samples', 'source_id', 'target_id', 'variant', 'carrier', 'length_scale', 'literal')
BOUNDARIES = ('generated_prefix_end_sample', 'full_rir_prefix_end_sample', 'generated_container_end_sample', 'full_rir_container_end_sample')


def compare(args):
    commit = subprocess.check_output(['git', 'rev-parse', args.gpu_ref], text=True).strip()
    inputs = {}
    def read(name):
        data = subprocess.check_output(['git', 'show', commit + ':' + BASE + name])
        inputs[BASE + name] = native.sha(data)
        return json.loads(data)
    calibration = read('4090-original-official-development-calibration-v1.json')
    execution = read('4090-hey-official-development-execution-v1.json')
    assert calibration['execution_sha256'] == inputs[BASE + '4090-hey-official-development-execution-v1.json']
    default, = [p for p in calibration['curves']['willow'] if p['request'] == 'DEFAULT']
    assert default['actual_float32_bits'] == native.DEFAULT_BITS
    references = json.loads(args.protocol.read_text())
    assert references['schema'] == 'willow-consumed108-frozen-protocol-v1'
    reduction = native.reduce_capture(args.capture.read_bytes(), references)
    qemu_inputs = {(r['source_id'], r['variant']): r for r in execution['records'] if r['phase'] == 'synthetic'}
    qemu_events = {(r['source_id'], r['variant']): r for r in default['synthetic']['records']}
    assert len(qemu_inputs) == len(qemu_events) == 360
    rows = references['fixtures']['rows']
    assert len(rows) == 108 and len(reduction['cases']) == 216
    matched = []
    for index, entry in enumerate(rows):
        row = entry['original']
        key = row['source_id'], row['variant']
        assert all(row[k] == qemu_inputs[key][k] for k in FIELDS)
        old = qemu_events[key]
        assert all(old[k] == row[k] for k in ('source_id', 'target_id', 'variant', 'carrier', 'length_scale'))
        expected = [(e['call'], e['sample_end'], e['result']) for e in old['events']]
        for event in old['events']:
            assert event['sample_end'] == min(event['call'] * 512, row['samples'])
            assert all(event[b + '_signed_ms'] == (event['sample_end'] - row[b]) / 16 for b in BOUNDARIES)
        for repeat in range(2):
            case = reduction['cases'][index * 2 + repeat]
            assert case['original'] == row and case['repeat'] == repeat
            actual = [(e['call'], e['sample_end'], e['result']) for e in case['events']]
            matched.append(dict(source_id=key[0], variant=key[1], literal=row['literal'], carrier=row['carrier'],
                length_scale=row['length_scale'], repeat=repeat, samples=row['samples'], pcm_sha256=row['pcm_sha256'],
                qemu_events=expected, p4_events=actual, event_coordinates_equal=actual == expected,
                p4_prime_positive_events=sum(v > 0 for v in case['initialization']['control']['prime_returns'])))
    groups = []
    for literal in references['fixtures']['selected_literals']:
        for repeat in range(2):
            selected = [r for r in matched if r['literal'] == literal and r['repeat'] == repeat]
            groups.append(dict(literal=literal, repeat=repeat, clips=len(selected),
                qemu_clips_with_events=sum(bool(r['qemu_events']) for r in selected),
                p4_clips_with_events=sum(bool(r['p4_events']) for r in selected),
                coordinate_disagreements=sum(not r['event_coordinates_equal'] for r in selected)))
    return dict(schema='willow-consumed108-p4-qemu-event-comparison-v1', gpu_commit=commit,
        source_sha256=native.sha(Path(__file__).read_bytes()), gpu_input_sha256=inputs,
        protocol_sha256=native.sha(args.protocol.read_bytes()), capture_sha256=native.sha(args.capture.read_bytes()),
        reduction_sha256=native.sha(json.dumps(reduction, sort_keys=True).encode()),
        same_input_wav_pcm_metadata_verified=108, native_executions=216, all_matched_cases=matched, groups=groups,
        all_native_repeats_exact=reduction['all_repeats_exact'],
        all_event_coordinates_equal=all(r['event_coordinates_equal'] for r in matched),
        p4_input_invocations=reduction['input_invocations'], p4_prime_invocations=reduction['prime_invocations'],
        p4_max_detect_us=reduction['max_detect_us'], threshold_selected=False, candidate_qualified=False,
        limits=['All108 identical consumed inputs; two P4 repeats are not independent speakers.',
                'Compares DEFAULT positive event coordinates only, not all underlying logits/native return bytes or arbitrary streams.',
                'Same catalog name and actual getter, separate architecture-specific builds; this does not independently prove identical model weight bytes between architectures.',
                'Generated support is not phonetic truth. No background FAR, new-source quality, acoustic wake delay, full-panel resources or voice lifecycle acceptance.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ref', required=True)
    for name in ('protocol', 'capture', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    result = compare(args)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('all_event_coordinates_equal', 'all_native_repeats_exact', 'p4_max_detect_us', 'groups')}))


if __name__ == '__main__':
    main()
