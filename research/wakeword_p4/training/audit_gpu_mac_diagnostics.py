"""Check pinned GPU diagnostic reports against the original saved Mac traces.

This verifies report identities and accounting. Linux arrays, float inference
and the identity of floating weights are not independently verified here.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def event_steps(values, threshold):
    armed, last, found = True, -1000, []
    for start in range(len(values)-4):
        mean = sum(int(v) for v in values[start:start+5])/1280
        step = start+5
        if mean < threshold/2 and step-last >= 67:
            armed = True
        if armed and mean >= threshold and step-last >= 67:
            found.append(step)
            armed, last = False, step
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-commit', required=True)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(['git', 'rev-parse', '--verify', args.gpu_commit+'^{commit}'],
                                     cwd=repo, text=True).strip()

    def read(relative):
        return subprocess.check_output(['git', 'show', commit+':research/wakeword_p4/'+relative], cwd=repo)

    cpu_raw = read('results/4090-mac-cpu-replay-v1.json')
    float_raw = read('results/4090-mac-float-int8-v1.json')
    cpu, floating = json.loads(cpu_raw), json.loads(float_raw)
    assert sha(read('experiments/replay_mac_cpu_v1.py')) == cpu['script_sha256']
    assert sha(read('experiments/mac_float_int8_v1.py')) == floating['script_sha256']
    study = args.study
    assert sha((study/'result.json').read_bytes()) == (
        '47ccac0cfd8d9b3809aa62e6b9f8232e70ea1d12655959648a4067b3a608c57b')
    assert sha((study/'scores.npz').read_bytes()) == (
        '374a399faa3f2acfc0603ef358767d08844f811f1d59f595c7fb7c90dacd8f29')
    mac = json.loads((study/'result.json').read_bytes())
    assert cpu['reference_result_sha256'] == sha((study/'result.json').read_bytes())
    assert cpu['reference_scores_sha256'] == sha((study/'scores.npz').read_bytes())
    assert cpu['model_sha256'] == floating['model_sha256'] == mac['model_sha256']
    assert floating['training_sha256'] == mac['training_sha256']
    assert floating['int8_scores_sha256'] == cpu['scores_sha256']
    assert len(mac['source_details']) == len(cpu['details']) == len(floating['details']) == 350
    assert floating['threshold'] == mac['v30_threshold'] == .91796875
    assert floating['smoothing_steps'] == mac['smoothing_steps'] == 5
    summary, transitions, frontend_diffs, output_diffs, time_details = {}, [], [], [], []
    max_mean_differences = []
    outputs = 0
    with np.load(study/'scores.npz', allow_pickle=False) as saved:
        for i, (m, c, f) in enumerate(zip(mac['source_details'], cpu['details'], floating['details'], strict=True)):
            for key in ('path', 'voice', 'target', 'text', 'variant', 'samples', 'keyword_end', 'sha256', 'scope'):
                assert f[key] == m[key], (i, key)
            assert m['path'] == c['path']
            values = saved[f'output_{i}']
            assert values.dtype == np.uint8 and values.ndim == 1
            assert sha(values.tobytes()) == m['quantized_output_sha256']
            assert c['output_count'] == f['output_count'] == len(values)
            outputs += len(values)
            assert c['frontend_shape'] == m['frontend_shape']
            assert c['frontend_dtype'] == m['frontend_dtype'] == 'uint16'
            same_frontend = c['frontend_sha256'] == m['frontend_sha256']
            same_output = c['output_sha256'] == m['quantized_output_sha256']
            assert c['frontend_exact'] == same_frontend and c['output_exact'] == same_output
            if not same_frontend:
                frontend_diffs.append(m['path'])
            if not same_output:
                output_diffs.append({'path': m['path'], 'reported_first_difference': c['first_output_difference']})
            steps = event_steps(values, floating['threshold'])
            assert steps == m['firmware_arithmetic']['one_based_score_steps']
            assert steps == f['int8_events'], m['path']
            qmean = max(sum(int(v) for v in values[k:k+5])/1280 for k in range(len(values)-4))
            if abs(qmean-f['int8_max_five_mean']) >= 1e-14:
                assert not same_output, m['path']
                max_mean_differences.append({'path': m['path'], 'mac': qmean,
                    'linux_reported': f['int8_max_five_mean']})
            correct = lambda s: 480*(s+1)-56160
            # The GPU script used 480*s rather than 480*(s+1) to filter
            # source events. Check whether that changes this actual result.
            qs = [s for s in steps if correct(s) > 0]
            fs = [s for s in f['float_events'] if correct(s) > 0]
            assert qs == f['int8_source_events'] and fs == f['float_source_events'], m['path']
            assert all(5 <= s <= len(values) for s in f['float_events'])
            assert f['mae'] >= 0 and f['max_error'] >= f['mae']
            assert np.isfinite([f['mae'], f['max_error'], f['float_max_five_mean']]).all()
            cohort = summary.setdefault(f['target'], Counter())
            cohort.update(clips=1, float_trigger_clips=bool(fs), int8_trigger_clips=bool(qs),
                          float_only=bool(fs) and not bool(qs), int8_only=bool(qs) and not bool(fs))
            if bool(fs) != bool(qs):
                transitions.append({'path': m['path'], 'target': m['target'],
                    'float_steps': fs, 'int8_steps': qs,
                    'float_max_five_mean': f['float_max_five_mean'], 'int8_max_five_mean': qmean})
            if f['float_events'] or steps:
                time_details.append({'path': m['path'],
                    'reported_float_steps': f['float_events'],
                    'reported_float_return_offsets': [correct(s) for s in f['float_events']],
                    'mac_verified_int8_steps': steps,
                    'mac_verified_int8_return_offsets': [correct(s) for s in steps]})
    summary = {key: dict(value) for key, value in summary.items()}
    assert summary == floating['summary']
    assert outputs == cpu['outputs'] == 94940
    result = {
        'scope': __doc__, 'gpu_commit': commit,
        'cpu_report_sha256': sha(cpu_raw), 'float_report_sha256': sha(float_raw),
        'source_sha256': sha(Path(__file__).read_bytes()),
        'sources_checked': 350, 'mac_output_bytes_checked': outputs,
        'model_sha256': mac['model_sha256'], 'summary_recomputed_from_report_rows': summary,
        'frontend_hash_differences': frontend_diffs, 'output_hash_differences': output_diffs,
        'all_reported_int8_events_match_mac_recomputed': True,
        'reported_linux_vs_mac_max_five_mean_differences': max_mean_differences,
        'identical_output_streams_with_verified_max_five_mean': 350-len(output_diffs),
        'source_filter_off_by_one_in_gpu_script_changes_these_events': False,
        'correct_return_offset_formula': '480*(one_based_score_step+1)-56160',
        'float_int8_trigger_transitions': transitions, 'event_return_offsets': time_details,
        'limits': [
            'No Linux score archive, original floating SavedModel or float trace was executed or verified by this audit.',
            'Float events/errors are reported by the pinned GPU result; summary consistency is not independent float inference.',
            'Cross-platform frontend element differences and the reported two differing output bytes require original arrays.',
            'Consumed synthetic intent labels do not establish real pronunciation, acoustic latency, FAR or acceptance.',
        ],
    }
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('gpu_commit', 'sources_checked',
        'mac_output_bytes_checked', 'summary_recomputed_from_report_rows', 'float_int8_trigger_transitions')}))


if __name__ == '__main__':
    main()
