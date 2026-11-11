"""Summarize the completed silent v30/Hi ESP probe from retained evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from audit_runtime_selftest import audit
from measure_idle_headroom import capture_audit, TIMING_FIELDS, TIMING_LINE


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(root, name, value):
    with (root/name).open('x') as out:
        out.write(json.dumps(value, indent=2)+'\n')


def summarize_arm(root, name, directory, report_name):
    runner = json.loads((root/directory/'report.json').read_bytes())
    report = json.loads((root/report_name).read_bytes())
    assert runner['restored'] and runner['acceptance_exit_code'] == 0
    assert report['completed'] and report['capture']['passed']
    assert report['app_sha256'] == runner['experimental_elf_sha256']
    assert not report['speaker_playback'] and not report['microphone_pcm_access']
    assert report['measure_seconds'] == 120 and len(report['samples']) == 121
    raw = (root/directory/'uart-private.log').read_bytes()
    recorded = [dict(zip(TIMING_FIELDS, map(int, row))) for row in TIMING_LINE.findall(raw)]
    windows = report['capture']['windows']
    assert sum(recorded[i:i+len(windows)] == windows for i in range(len(recorded))) == 1
    counts = ''.join('panel.wake.timing: '+' '.join(f'{k}={row[k]}' for k in TIMING_FIELDS)
                     +'\n' for row in windows).encode()
    assert capture_audit(counts, report['capture']['backlog_bound_bytes'],
                         report['capture']['minimum_seconds']) == report['capture']
    with (root/(name+'-counts-only.txt')).open('xb') as out:
        out.write(counts)
    # The first retained window establishes the RX counter origin. Its interval
    # can include earlier activity and is excluded from steady timing totals.
    steady = windows[1:]
    wall = sum(row['wall_us'] for row in steady)
    elapsed = sum(row['detect_us'] for row in steady)
    calls = sum(row['calls'] for row in steady)
    intervals = report['capture']['intervals']
    assert wall/1e6 == report['capture']['wall_seconds']
    assert len({row['pcm_samples']//row['calls'] for row in steady}) == 1
    return {
        'app_elf_sha256': report['app_sha256'],
        'app_bin_sha256': runner['experimental_bin_sha256'],
        'app_bin_bytes': (root/(name+'-nyxid-esp-panel.bin')).stat().st_size,
        'restored_original_app_service_and_listener': runner['restored'],
        'flash_baud': runner['flash_baud'],
        'idle_seconds_requested': report['measure_seconds'],
        'status_samples': len(report['samples']),
        'steady_timing': {
            'complete_windows': len(steady), 'calls': calls,
            'block_samples': steady[0]['pcm_samples']//steady[0]['calls'],
            'wall_seconds': wall/1e6, 'detector_elapsed_seconds': elapsed/1e6,
            'detector_elapsed_wall_fraction': elapsed/wall,
            'mean_call_elapsed_ms': elapsed/calls/1000,
            'max_call_elapsed_ms': max(row['max_us'] for row in steady)/1000,
            'min_stack_free_bytes': min(row['stack_free'] for row in steady),
            'first_retained_window_excluded': True,
        },
        'heap_summary_bytes': report['heap_summary_bytes'],
        'capture': {
            'passed': True, 'input_seconds': report['capture']['input_seconds'],
            'first_retained_rx_overflows': windows[0]['rx_overflows'],
            'new_rx_overflows': sum(row['new_overflows'] for row in intervals),
            'cumulative_rx_read_balance_range_bytes': [
                min(row['unaccounted_rx_bytes_since_start'] for row in intervals),
                max(row['unaccounted_rx_bytes_since_start'] for row in intervals)],
            'backlog_bound_bytes': report['capture']['backlog_bound_bytes'],
        },
        'evidence_sha256': {'idle_report': sha(root/report_name),
            'runner_report': sha(root/directory/'report.json'),
            'private_final_uart': sha(root/directory/'uart-private.log'),
            'public_counts': sha(root/(name+'-counts-only.txt'))},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args(); root = args.artifacts.resolve()
    repo = Path(__file__).resolve().parents[2]
    outputs = ('result.json', 'v30-counts-only.txt', 'hiesp-counts-only.txt',
               'selftest-markers-only.txt', 'selftest-verdict-sanitized.json')
    if any((root/name).exists() for name in outputs):
        parser.error('Use fresh summary outputs; retained evidence is never overwritten')
    sdk = sha(root/'v30-sdkconfig')
    assert sdk == sha(root/'hiesp-sdkconfig')
    original = yaml.safe_load((repo/'dependencies.lock').read_bytes())['dependencies']
    resolved = [yaml.safe_load((root/name).read_bytes())['dependencies'] for name in
                ('resolved-dependencies-private.lock', 'resolved-dependencies-final-private.lock')]
    assert set(original) == set(resolved[0]) == set(resolved[1])
    def identities(deps):
        return {k: (v.get('version'), v.get('component_hash')) for k, v in deps.items()}
    assert identities(original) == identities(resolved[0]) == identities(resolved[1])
    fixture = json.loads((root/'v30-prime-selftest.json').read_bytes())
    verdict = json.loads((root/'selftest-verdict-final.json').read_bytes())
    raw = (root/'board/uart-private.log').read_bytes()
    assert verdict['uart_sha256'] == hashlib.sha256(raw).hexdigest()
    verified = audit(raw.decode(errors='replace'), fixture)
    assert all(verdict[k] == v for k, v in verified.items())
    # Publish only the fields recognized and checked by the auditor.
    markers = ('panel.wake.research: model='+verdict['model_sha256']+'\n'
               + ''.join(f'panel.wake.research: activation step={step}\n'
                         for step in verdict['activation_steps'])
               + f"exact-PCM selftest outputs={verdict['outputs']} mismatches=0 passed=true\n")
    assert audit(markers, fixture) == verified
    with (root/'selftest-markers-only.txt').open('x') as out:
        out.write(markers)
    save(root, 'selftest-verdict-sanitized.json', {**verified,
         'sanitized_markers_sha256': sha(root/'selftest-markers-only.txt'),
         'original_final_uart_sha256': verdict['uart_sha256'],
         'normalization': 'Reconstructed only audited model, event-step and verdict fields; no raw UART text.'})
    arms = {name: summarize_arm(root, name, directory, report) for name, directory, report in
            (('v30', 'board', 'headroom-private.json'),
             ('hiesp', 'board-hiesp', 'hiesp-headroom-private.json'))}
    diagnostic = json.loads((root/'diagnostic-training-source-results.json').read_bytes())
    result = {'format': 'nyx-gpu-v30-p4-probe-v1', 'accepted': False,
        'scope': 'Silent full-app numeric self-test and matched steady idle resources; no acoustic acceptance.',
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'summarizer_sha256': sha(Path(__file__)),
        'model_binding': json.loads((root/'v30-build-manifest.json').read_bytes()),
        'selftest': verified,
        'training_only_diagnostic': {'sources': len(diagnostic['sources']),
            'varied_output_sources': sum(row['unique_output_bytes'] > 1 for row in diagnostic['sources']),
            'reset_reproducible': all(row['reset_reproducible'] for row in diagnostic['sources']),
            'first_source_rule': 'First two lexicographic positive training/variant-0 paths per engine, frozen before scoring.',
            'recognition_acceptance': False, 'rejected_legacy_source_all_outputs_zero': True},
        'build_comparison': {'identical_sdkconfig_sha256': sdk,
            'dependency_component_count': len(original), 'all_dependency_versions_and_component_hashes_match': True,
            'same_raw_idle_input_and_timing_instrumentation': True,
            'calculator_only_generated_ninja_warning_exception': '-Wno-error=format-truncation'},
        'arms': arms,
        'speaker_playback': False, 'microphone_pcm_recording': False,
        'production_board_touched': False,
        'limits': [
            'Detect elapsed time includes scheduling; it is neither CPU utilization nor acoustic latency.',
            'Different detector block sizes require time-normalized comparisons.',
            'Startup loss is not accepted as zero: initial overflow counts differ and v30 runs its diagnostic during startup.',
            'The single synthetic numeric fixture and reused training-source diagnostics are not recall/false-wake tests.',
            'Heap minima are idle observations, not proof of active-call resource capacity.',
            'Real-user/acoustic recognition, word-end latency, continuous room negatives and active long-call/echo/reconnect comparison remain open.',
        ]}
    save(root, 'result.json', result)
    print(json.dumps({'accepted': False, 'arms': {k: v['steady_timing'] for k, v in arms.items()}}))


if __name__ == '__main__':
    main()
