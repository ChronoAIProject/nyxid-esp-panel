"""Reconcile v41 training, fixed joint gate and matched training diagnostics.

Reads committed reports/source only. Does not load model weights, saved score
arrays or audio, and does not access a board or change experimental selection.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess

from review_gpu_v40_reports import poisson_upper95


BASE = 'research/wakeword_p4/'
CONTROLS = '''context_frames initial_weights_sha256 classifier_initial_sha256
batchnorm_initial seed recipe parameters steps architecture architecture_args
full_source five_score effective_training_batch_size batch_size negative_weight
prefix_negatives phone_binary spec_augment label_smoothing ambient_sampling
ambient_sampling_coverage source_exponent validation_window_phase
features_index_sha256 feature_metadata_sha256 selection'''.split()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--training-revision', default='df9e63c')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Fresh output required')
    revision, training_revision = [subprocess.check_output(
        ['git', 'rev-parse', value+'^{commit}'], text=True).strip()
        for value in (args.revision, args.training_revision)]
    hashes = {}

    def read(path, ref=None):
        ref = ref or revision
        data = subprocess.check_output(['git', 'show', ref+':'+BASE+path])
        hashes[ref+':'+path] = hashlib.sha256(data).hexdigest()
        return json.loads(data) if path.endswith('.json') else data

    def report(suffix):
        return read('results/4090-v41-'+suffix+'.json')

    training = report('training')
    baseline = read('results/4090-v40-training.json')
    export = report('export-audit-v1')
    calibration = report('reused-development-exploratory-calibration')
    rejection = report('rejection-receipt-v1')
    phone = report('phone-development-historical-threshold')
    vctk = report('vctk-development-historical-threshold')
    background = report('voxconverse-development')
    support = report('training-support-v1')
    float_history = report('float-history-v1')
    support40 = read('results/4090-v40-training-support-v1.json')
    comparison = read('results/4090-v40-v41-training-support-comparison-v1.json')
    model = training['model_sha256']
    assert all(row['model_sha256'] == model for row in
               (export, calibration, rejection, phone, vctk, background, support, float_history))
    assert all(training[key] == baseline[key] for key in CONTROLS)
    coverage, previous = dict(training['complete_coverage']), dict(baseline['complete_coverage'])
    timeline = coverage.pop('negative_timeline')
    coverage.pop('source_sha256'); previous.pop('source_sha256')
    assert coverage == previous
    assert timeline == dict(seed=137726,
        policy='Uniform integer end150..len(raw),24 candidate negatives only',
        authenticated_variants=1995)
    training_sha = hashes[revision+':results/4090-v41-training.json']
    assert export['training_sha256'] == support['training_sha256'] == training_sha
    assert training['batchnorm_initial'] == training['batchnorm_selected']
    assert all(training[key] == export[key] for key in
               ('steps', 'selected_step', 'parameters', 'model_sha256', 'model_bytes', 'seconds', 'gpu_details'))
    assert export['all_saved_weights_finite'] and export['frozen_bn_unchanged']
    assert not export['candidate_promoted']
    for name in ('train_stream.py', 'complete_coverage.py'):
        data = read('training/'+name, training_revision)
        assert hashlib.sha256(data).hexdigest() == training['source_files_sha256'][name]
    for path, expected in (
            ('training/audit_stream_export.py', export['source_sha256']),
            ('training/calibrate_phone_development.py', calibration['source_sha256']),
            ('training/evaluate.py', calibration['detector_source_sha256']),
            ('experiments/v41_training_support_probe.py', support['script_sha256']),
            ('experiments/v41_float_history_probe.py', float_history['script_sha256'])):
        assert hashlib.sha256(read(path)).hexdigest() == expected
    assert calibration['synthetic_report_sha256'] == [
        hashes[revision+':results/4090-v41-phone-development-historical-threshold.json'],
        hashes[revision+':results/4090-v41-vctk-development-historical-threshold.json']]
    assert calibration['background_report_sha256'] == hashes[
        revision+':results/4090-v41-voxconverse-development.json']

    # Reconstruct each float checkpoint's original rank, preserving stable ties.
    assert [row['step'] for row in training['history']] == list(range(5000, 120001, 5000))
    best, chosen, eligible_steps = None, None, []
    for row in training['history']:
        eligible = [p for p in row['grid'] if p['ambient_faph'] <= .5
                    and p['max_source_false_rate'] <= .01]
        if eligible:
            point = max(eligible, key=lambda p: (p['macro_recall'], -p['ambient_faph'], -p['max_source_false_rate']))
            rank = (1, point['macro_recall'], -point['ambient_faph'], -point['max_source_false_rate'])
            eligible_steps.append(row['step'])
        else:
            point = min(row['grid'], key=lambda p: (p['ambient_faph'], p['max_source_false_rate'], -p['macro_recall']))
            rank = (0, -point['ambient_faph'], -point['max_source_false_rate'], point['macro_recall'])
        assert row['selected'] == point
        if best is None or rank > best:
            best, chosen = rank, row['step']
    assert chosen == training['selected_step'] == 25000

    grid = sorted(set([.5, .7, .8, .9, .95, .97, .98, .985, .99, .992, .994]
                      + [.5+i/256 for i in range(128)]))
    assert [row['threshold'] for row in calibration['grid']] == grid
    for row in calibration['grid']:
        groups = list(row['groups'].values())
        macro = sum(g['triggered']/g['positives'] for g in groups)/len(groups)
        upper = poisson_upper95(row['background_events'], row['background_hours'])
        assert math.isclose(macro, row['synthetic_macro_clip_trigger'], abs_tol=1e-14)
        assert math.isclose(upper, row['background_upper_95'], abs_tol=1e-12)
        eligible = (upper <= .5 and macro >= .48125
                    and all(not g['false_activation_clips'] and not g['pre_generation_positive_events']
                            for g in groups))
        assert row['eligible'] == eligible
    assert sum(row['eligible'] for row in calibration['grid']) == rejection['eligible_points'] == 0
    assert len(grid) == rejection['grid_points'] == 138
    assert max(row['synthetic_macro_clip_trigger'] for row in calibration['grid']) == rejection['max_macro']
    assert min(sum(g['false_activation_clips'] for g in row['groups'].values())
               for row in calibration['grid']) == rejection['minimum_negative_false_clips'] == 13
    assert calibration['selected'] is calibration['threshold'] is None
    assert calibration['viable_operating_point'] is False
    events = sum(row['false_events'] for row in background['speakers'])
    hours = sum(row['source_seconds'] for row in background['speakers'])/3600
    assert events == rejection['background_events'] == 1
    assert math.isclose(hours, rejection['background_hours'], abs_tol=1e-12)
    upper = poisson_upper95(events, hours)
    assert math.isclose(upper, rejection['background_upper95'], abs_tol=1e-12)

    historical = {}
    for name, paired in [('kokoro', phone), ('vctk', vctk)]:
        rows = paired['records']
        target = [r for r in rows if r['record']['target'] == 'hey_nyx']
        common = [r for r in rows if r['record']['target'] not in ('hey_nyx', 'hi_esp')]
        cross = [r for r in rows if r['record']['target'] == 'hi_esp']
        historical[name] = dict(
            target_any=[sum(bool(r['nyx']['event_seconds']) for r in target), len(target)],
            common_false=[sum(bool(r['nyx']['event_seconds']) for r in common), len(common)],
            cross_false=sum(bool(r['nyx']['event_seconds']) for r in cross))
    assert historical == dict(kokoro=dict(target_any=[9, 16], common_false=[3, 48], cross_false=0),
                              vctk=dict(target_any=[26, 60], common_false=[10, 180], cross_false=0))

    assert [r['pack_index'] for r in support['details']] == [r['pack_index'] for r in support40['details']]
    assert len({r['pack_index'] for r in support['details']}) == comparison['matched_indices'] == 6597
    summaries, transitions = {}, {}
    for version, data in [('v40', support40), ('v41', support)]:
        assert comparison['reports_sha256'][version] == hashes[
            revision+':results/4090-'+version+'-training-support-v1.json']
        summaries[version] = {str(label): {str(threshold): {
            'window_float': sum(r['window_float'] >= threshold for r in data['details'] if r['label'] == label),
            'full_int8_max_five': sum(r['full_int8_max_five'] >= threshold for r in data['details'] if r['label'] == label)
        } for threshold in (.5, .99609375)} for label in (0, 1)}
    for threshold in (.5, .99609375):
        counts = Counter()
        for before, after in zip(support40['details'], support['details'], strict=True):
            assert all(before[key] == after[key] for key in ('pack_index', 'engine', 'phrase', 'label', 'variant'))
            if after['label'] == 0:
                change = f"{int(before['full_int8_max_five'] >= threshold)}->{int(after['full_int8_max_five'] >= threshold)}"
                counts[after['engine']+'/'+after['phrase']+'/'+change] += 1
        transitions[str(threshold)] = dict(counts)
    assert summaries == comparison['summaries']
    assert transitions == comparison['negative_full_int8_transitions']
    assert float_history['prior_report_sha256'] == hashes[
        revision+':results/4090-v41-training-support-v1.json']
    assert float_history['weights_sha256'] == export['weights_sha256']
    assert float_history['all_window_variables_unchanged'] and float_history['optimizer_updates'] == 0
    assert len(float_history['details']) == len(support['details'])
    for old, new in zip(support['details'], float_history['details'], strict=True):
        assert all(new[key] == value for key, value in old.items())
    float_summary = {}
    for label in (0, 1):
        rows = [row for row in float_history['details'] if row['label'] == label]
        float_summary[str(label)] = dict(examples=len(rows))
        for threshold in (.5, .99609375):
            float_summary[str(label)][str(threshold)] = {
                key: sum(row[key] >= threshold for row in rows)
                for key in ('window_float', 'float_full_max_five', 'full_int8_max_five')}
    assert float_summary == float_history['summary']
    first_by_group = {}
    for row in float_history['details']:
        first_by_group.setdefault((row['engine'], 'positive' if row['label'] else row['phrase']), row['pack_index'])
    witnesses = float_history['sequence_stream_witnesses']
    assert [row['pack_index'] for row in witnesses] == list(first_by_group.values())
    assert len(witnesses) == 20
    witness_error = max(row['max_sequence_stream_error'] for row in witnesses)
    assert calibration['validation_only'] and calibration['final_data_used'] is False
    assert not any(rejection[key] for key in ('candidate_promoted', 'new_development_scored', 'final_scored'))
    result = dict(
        scope='Independent committed-report reconciliation; original v41 rejection retained.', accepted=False,
        gpu_revision=revision, training_revision=training_revision,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(__file__).with_name('review_gpu_v40_reports.py').read_bytes()).hexdigest(),
        input_sha256=hashes, model_sha256=model, control_fields_exact=CONTROLS,
        coverage_changed_only_by_declared_negative_timeline=True,
        all_24_checkpoint_rankings_recomputed=True, selected_step=chosen,
        internal_float_eligible_steps=eligible_steps,
        gate=dict(points=len(grid), eligible=0, minimum_negative_false_clips=13,
                  maximum_macro=rejection['max_macro'], background_events=events,
                  background_hours=hours, background_upper95=upper),
        historical=historical, matched_training_summaries=summaries,
        negative_full_int8_transitions=transitions,
        same_weights_float_summary=float_summary,
        actual_float_stream_witnesses=len(witnesses), maximum_sequence_stream_residual=witness_error,
        limits=[
            'Report/source identities and calculations only; no raw model weights, saved score arrays or GPU process independently verified.',
            'Training max-five counts are descriptive, not latched-event recognition or new acceptance.',
            'Training distributions and selected checkpoints differ; no isolated causal or phonetic claim.',
            'Shared float sequence has 20 actual exported-stream witnesses, not all6597 actual streams; nonzero residual retained.',
            'No new/final data, model inference, audio I/O, board operation, threshold selection or training.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('model_sha256', 'selected_step', 'gate', 'historical')}, indent=2))


if __name__ == '__main__':
    main()
