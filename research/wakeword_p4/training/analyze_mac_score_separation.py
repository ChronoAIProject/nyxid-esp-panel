"""Compute necessary scalar-score separation bounds without choosing a threshold."""
import argparse
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def core_identity(repo, commit, path, name):
    raw = subprocess.check_output(['git', 'show', commit+':'+path], cwd=repo)
    node, = [node for node in ast.parse(raw).body if getattr(node, 'name', None) == name]
    return {'file_sha256': digest(raw),
            'core_ast_sha256': digest(ast.dump(node, include_attributes=False).encode())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'plan', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); root = args.artifacts
    plan = json.loads(args.plan.read_bytes())
    raw_result = (root/'result.json').read_bytes()
    assert digest(raw_result) == plan['full_result_sha256']
    assert digest((root/'scores.npz').read_bytes()) == plan['scores_sha256']
    report = json.loads(raw_result)
    assert report['accepted'] is False and report['sources'] == plan['sources'] == 350
    schema = plan.get('report_schema', 'v30-original-v1')
    assert schema in ('v30-original-v1', 'candidate-mac-v1')
    if schema == 'v30-original-v1':
        assert report['prime_samples'] == plan['prime_samples'] == 56160
    else:
        assert all(row['prime_samples'] == plan['prime_samples'] == 56160
                   for row in report['source_details'])
        assert report['threshold'] == plan['threshold']
    output_hash_field = 'output_sha256' if schema == 'candidate-mac-v1' else 'quantized_output_sha256'
    threshold_sum = Fraction.from_float(float(np.float32(plan['threshold'])))*1280
    assert plan['block_samples'] == 480 and plan['score_window'] == 5
    qfile = np.load(root/'scores.npz', allow_pickle=False)
    assert set(qfile.files) == {f'output_{i}' for i in range(350)}
    rows, prime_max = [], 0
    for index, source in enumerate(report['source_details']):
        q = qfile[f'output_{index}']
        assert q.dtype == np.uint8 and q.ndim == 1 and len(q) == source['output_count']
        assert digest(q.tobytes()) == source[output_hash_field]
        sums = np.asarray([sum(map(int, q[at:at+5])) for at in range(len(q)-4)], dtype=np.int64)
        host = np.convolve(q.astype(np.float64)/256, np.ones(5)/5, mode='valid')
        # First five-score mean ends at raw score step 5, returned by PCM block 6.
        ends = (np.arange(len(sums))+6)*480-56160
        assert np.any(ends <= 0) and np.any(ends > 0)
        prime_max = max(prime_max, int(sums[ends <= 0].max()))
        source_sum = int(sums[ends > 0].max())
        source_host = float(host[ends > 0].max())
        assert (source_sum >= threshold_sum) == bool(source['firmware_arithmetic']['events'])
        assert (source_host >= plan['threshold']) == bool(source['host_convolution']['events'])
        rows.append({k: source[k] for k in ('path', 'voice', 'variant', 'text', 'target', 'sha256')})
        rows[-1].update(max_five_score_sum=source_sum, denominator=1280,
                        host_convolution_max=source_host)
    assert prime_max == 0, 'A nonzero prime needs separate latch analysis; do not use this certificate'
    positive = [row for row in rows if row['target'] == 'hey_nyx']
    confuser = [row for row in rows if row['target'] == 'none' and row['text'].startswith('Hey ')]
    esp = {'Hi SP.', 'Hi ESB.', 'Hi ASP.', 'Hi EST.', 'I see ESP.', 'High speed.'}
    ordinary = [row for row in rows if row['target'] == 'none'
                and not row['text'].startswith('Hey ') and row['text'] not in esp]
    assert (len(positive), len(confuser), len(ordinary)) == (50, 90, 100)
    highest = max(row['max_five_score_sum'] for row in confuser)
    highest_host = max(row['host_convolution_max'] for row in confuser)
    firmware_cap = sum(row['max_five_score_sum'] > highest for row in positive)
    host_cap = sum(row['host_convolution_max'] > highest_host for row in positive)
    repo = Path(__file__).resolve().parents[3]
    core = {}
    for filename, symbol in [('features.py', 'features'), ('evaluate.py', 'Detector')]:
        path = 'research/wakeword_p4/training/'+filename
        identities = {track: core_identity(repo, plan[track+'_commit'], path, symbol)
                      for track in ('mac', 'gpu')}
        core[symbol] = {**identities, 'same_core_ast':
                        identities['mac']['core_ast_sha256'] == identities['gpu']['core_ast_sha256']}
    nonzero = sum(row['max_five_score_sum'] > 0 for row in positive)
    result = {'scope': plan['scope'], 'accepted': False,
        'plan_sha256': digest(args.plan.read_bytes()), 'source_sha256': digest(Path(__file__).read_bytes()),
        'input_result_sha256': plan['full_result_sha256'], 'input_scores_sha256': plan['scores_sha256'],
        'sources': len(rows), 'all_saved_output_hashes_verified': True,
        'all_frozen_threshold_event_booleans_reconstructed': True, 'prime_max_sum': prime_max,
        'current_firmware_positive_hits': sum(row['max_five_score_sum'] >= threshold_sum for row in positive),
        'current_firmware_confuser_false_clips': sum(row['max_five_score_sum'] >= threshold_sum for row in confuser),
        'score_separation': {
            'highest_confuser_five_score_sum': highest, 'denominator': 1280,
            'positive_upper_bound_at_zero_confuser_false_clips': firmware_cap,
            'positive_ties_with_highest_confuser': sum(row['max_five_score_sum'] == highest for row in positive),
            'host_highest_confuser_average': highest_host,
            'host_positive_upper_bound_at_zero_confuser_false_clips': host_cap,
            'positive_sources_with_nonzero_score': nonzero,
            'positive_sources_with_all_zero_scores': len(positive)-nonzero,
            'descriptive_reference_hits': plan['descriptive_hiesp_reference_hits'],
            'positive_threshold_cannot_reach_reference_even_ignoring_false_clips': nonzero < plan['descriptive_hiesp_reference_hits'],
            'scalar_threshold_cannot_reach_reference_at_zero_confuser_false_clips': firmware_cap < plan['descriptive_hiesp_reference_hits'],
        },
        'worst_confuser_witnesses': [row for row in confuser if row['max_five_score_sum'] == highest],
        'positive_max_sum_histogram': dict(sorted(Counter(row['max_five_score_sum'] for row in positive).items())),
        'by_voice': {voice: {'sources': len(selected := [row for row in positive if row['voice'] == voice]),
            'all_zero_score_sources': sum(row['max_five_score_sum'] == 0 for row in selected),
            'min_max_sum': min(row['max_five_score_sum'] for row in selected),
            'median_max_sum': float(np.median([row['max_five_score_sum'] for row in selected])),
            'max_max_sum': max(row['max_five_score_sum'] for row in selected)}
            for voice in sorted({row['voice'] for row in positive})},
        'branch_inference_core': core, 'limits': plan['limits'], 'source_scores': rows}
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.plan, args.output/'plan.json')
    shutil.copyfile(__file__, args.output/'source.py')
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'score_separation': result['score_separation'], 'same_branch_core':
                      {name: value['same_core_ast'] for name, value in core.items()}}))


if __name__ == '__main__':
    main()
