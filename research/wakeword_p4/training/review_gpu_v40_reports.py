"""Independently reconcile committed v40 reports, without loading any model/audio.

Read GPU evidence from an explicit git revision. Optionally authenticate the
unchanged public training pack; this checks metadata coverage, not score traces.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


BASE = 'research/wakeword_p4/'
MODEL = '0b48cd96e8837b27c2df73c709c3360da9f06079f61eb9464922bae438bb98e4'
CONTROL_FIELDS = '''context_frames initial_weights_sha256 classifier_initial_sha256
batchnorm_initial batchnorm_selected seed recipe parameters steps architecture_args
full_source prefix_negatives negative_weight effective_training_batch_size
ambient_sampling'''.split()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def poisson_upper95(events, hours):
    """Invert P(Poisson(mu) <= observed)=.05 without scipy/chi2."""
    lo, hi = 0., max(10., 2. * (events + 1))
    for _ in range(100):
        mu = (lo + hi) / 2
        cdf = math.exp(-mu) * sum(mu ** k / math.factorial(k)
                                 for k in range(events + 1))
        if cdf > .05:
            lo = mu
        else:
            hi = mu
    return (lo + hi) / (2 * hours)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--training-revision', default='4974646',
                        help='Completed v40 source revision, before later sampler changes')
    parser.add_argument('--pack', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Fresh output required')
    revision = subprocess.check_output(
        ['git', 'rev-parse', args.revision + '^{commit}'], text=True).strip()
    training_revision = subprocess.check_output(
        ['git', 'rev-parse', args.training_revision + '^{commit}'], text=True).strip()
    input_hashes = {}

    def read(relative, source_revision=None):
        ref = source_revision or revision
        data = subprocess.check_output(['git', 'show', ref + ':' + BASE + relative])
        input_hashes[(ref + ':' if source_revision else '') + relative] = digest(data)
        return json.loads(data) if relative.endswith('.json') else data

    def report(suffix):
        return read('results/4090-v40-' + suffix + '.json')

    training = report('training')
    export = report('export-audit')
    rejection = report('rejection-v1')
    calibration = report('reused-development-exploratory-calibration-v1')
    phone = report('phone-development-v1')
    vctk = report('vctk-development-v1')
    background = report('voxconverse-development-v1')
    support = report('training-support-v1')
    history = report('float-history-v1')
    localization = report('negative-stream-localization-v1')
    v37 = read('results/4090-v37-training.json')
    for evidence in [training, export, rejection, calibration, phone, vctk,
                     background, support, history, localization]:
        assert evidence['model_sha256'] == MODEL
    training_hash = input_hashes['results/4090-v40-training.json']
    assert export['training_sha256'] == support['training_sha256'] == training_hash
    assert history['prior_report_sha256'] == input_hashes['results/4090-v40-training-support-v1.json']
    assert history['weights_sha256'] == export['weights_sha256']
    assert localization['prior_report_sha256'] == input_hashes['results/4090-v40-float-history-v1.json']
    assert all(training[key] == v37[key] for key in CONTROL_FIELDS)
    assert training['five_score'] is None
    assert training['complete_coverage']['seed'] == 127726
    assert training['selected_step'] == export['selected_step'] == 45000
    assert export['candidate_promoted'] is False
    assert history['optimizer_updates'] == 0 and history['all_window_variables_unchanged']
    source_bindings = {
        'training/audit_stream_export.py': (training_revision, export['source_sha256']),
        'training/complete_coverage.py': (training_revision, training['complete_coverage']['source_sha256']),
        'experiments/v40_training_support_probe.py': (revision, support['script_sha256']),
        'experiments/v40_float_history_probe.py': (revision, history['script_sha256']),
        'experiments/v40_negative_stream_localization.py': (revision, localization['script_sha256']),
    }
    for path, (ref, expected) in source_bindings.items():
        assert digest(read(path, ref)) == expected, path
    assert calibration['synthetic_report_sha256'] == [
        input_hashes['results/4090-v40-phone-development-v1.json'],
        input_hashes['results/4090-v40-vctk-development-v1.json']]
    assert calibration['background_report_sha256'] == input_hashes['results/4090-v40-voxconverse-development-v1.json']
    assert phone['threshold'] == vctk['threshold'] == background['threshold'] == rejection['historical_threshold']
    assert phone['calibration_sha256'] == vctk['calibration_sha256'] == background['frozen_report_sha256']

    expected_grid = sorted(set([.50, .70, .80, .90, .95, .97, .98,
                               .985, .99, .992, .994] + [.5 + i/256 for i in range(128)]))
    assert [row['threshold'] for row in calibration['grid']] == expected_grid
    for row in calibration['grid']:
        groups = list(row['groups'].values())
        macro = sum(g['triggered']/g['positives'] for g in groups) / len(groups)
        assert math.isclose(macro, row['synthetic_macro_clip_trigger'], abs_tol=1e-14)
        upper = poisson_upper95(row['background_events'], row['background_hours'])
        assert math.isclose(upper, row['background_upper_95'], abs_tol=1e-12)
        eligible = (upper <= .5 and macro >= .48125
                    and all(g['false_activation_clips'] == 0
                            and g['pre_generation_positive_events'] == 0 for g in groups))
        assert row['eligible'] == eligible
    eligible_count = sum(row['eligible'] for row in calibration['grid'])
    maximum_macro = max(row['synthetic_macro_clip_trigger'] for row in calibration['grid'])
    minimum_false = min(sum(g['false_activation_clips'] for g in row['groups'].values())
                        for row in calibration['grid'])
    assert (len(expected_grid), eligible_count, maximum_macro, minimum_false) == (
        rejection['fixed_grid_points'], rejection['eligible_points'],
        rejection['max_grid_synthetic_macro'], rejection['min_grid_negative_false_clips'])
    assert eligible_count == 0 and calibration['selected'] is calibration['threshold'] is None
    assert calibration['viable_operating_point'] is False
    assert sum(r['false_events'] for r in background['speakers']) == rejection['vox_events'] == 0
    hours = sum(r['source_seconds'] for r in background['speakers']) / 3600
    assert math.isclose(hours, background['source_hours'], abs_tol=1e-12)
    assert math.isclose(poisson_upper95(0, hours), rejection['vox_upper95'], abs_tol=1e-12)

    historical = {}
    for name, paired, expected_count in [('kokoro', phone, 80), ('vctk', vctk, 300)]:
        assert len(paired['records']) == expected_count
        rows = paired['records']
        target = [r for r in rows if r['record']['target'] == 'hey_nyx']
        common = [r for r in rows if r['record']['target'] not in ('hey_nyx', 'hi_esp')]
        cross = [r for r in rows if r['record']['target'] == 'hi_esp']
        counts = dict(target_any=[sum(bool(r['nyx']['event_seconds']) for r in target), len(target)],
                      common_false=[sum(bool(r['nyx']['event_seconds']) for r in common), len(common)],
                      cross_false=sum(bool(r['nyx']['event_seconds']) for r in cross))
        assert counts == rejection[name]
        false_groups = Counter((r['record']['intended_phrase'], r['record']['voice'])
                               for r in common + cross if r['nyx']['event_seconds'])
        historical[name] = dict(**counts, false_by_intended_phrase_voice=[
            dict(phrase=k[0], voice=k[1], clips=n) for k, n in sorted(false_groups.items())])

    indices = [row['pack_index'] for row in support['details']]
    assert len(indices) == len(set(indices)) == 6597
    assert indices == [row['pack_index'] for row in history['details']]
    for old, new in zip(support['details'], history['details']):
        assert all(old[key] == new[key] for key in old)
    transitions = {}
    grouped = defaultdict(Counter)
    for label in (0, 1):
        rows = [r for r in history['details'] if r['label'] == label]
        summary = {'examples': len(rows)}
        for threshold in (.5, .99609375):
            counts = {key: sum(r[key] >= threshold for r in rows)
                      for key in ('window_float', 'float_full_max_five', 'full_int8_max_five')}
            summary[str(threshold)] = counts
            pairs = Counter((r['float_full_max_five'] >= threshold,
                             r['full_int8_max_five'] >= threshold) for r in rows)
            transitions[f'label={label}/threshold={threshold}'] = {
                'both_high': pairs[True, True], 'float_only': pairs[True, False],
                'int8_only': pairs[False, True], 'both_low': pairs[False, False]}
        assert summary == history['summary'][str(label)]
    for row in history['details']:
        if row['label'] == 0:
            key = (row['engine'], row['phrase'], row['variant'])
            grouped[key]['examples'] += 1
            for model, field in [('float', 'float_full_max_five'), ('int8', 'full_int8_max_five')]:
                for threshold in (.5, .99609375):
                    grouped[key][f'{model}_high_at_{threshold}'] += int(row[field] >= threshold)

    expected_union = [r for r in history['details'] if r['label'] == 0 and
                      (r['float_full_max_five'] >= .5 or r['full_int8_max_five'] >= .5)]
    actual_rows = localization['details']
    assert len(actual_rows) == len(expected_union) == localization['selection_count'] == 118
    assert localization['optimizer_updates'] == 0 and localization['all_window_variables_unchanged']
    for expected, actual in zip(expected_union, actual_rows):
        assert all(actual[key] == expected[key] for key in expected)
        assert actual['peak_minus_crop_end_frames'] == (
            actual['peak_five_end_feature_exclusive'] - actual['training_crop_end'])
        assert actual['peak_five_end_feature_exclusive'] % 3 == 0
    localized_summary = {}
    for threshold in (.5, .99609375):
        high = [r for r in actual_rows if r['actual_stream_max_five'] >= threshold]
        assert len(high) == localization['summary'][str(threshold)]
        shifts = [r['peak_minus_crop_end_frames'] for r in high]
        localized_summary[str(threshold)] = dict(
            actual_high=len(high), later=sum(n > 0 for n in shifts),
            earlier=sum(n < 0 for n in shifts), equal=shifts.count(0),
            median_processing_frame_shift=statistics.median(shifts),
            phrases=dict(sorted(Counter(r['phrase'] for r in high).items())),
            sequence_actual_classification_differences=sum(
                (r['float_full_max_five'] >= threshold) !=
                (r['actual_stream_max_five'] >= threshold) for r in actual_rows))
    actual_max_residual = max(r['sequence_stream_max_error'] for r in actual_rows)
    assert actual_max_residual == localization['max_sequence_stream_error']

    pack_check = None
    if args.pack:
        for name, expected in training['complete_coverage']['input_sha256'].items():
            assert digest((args.pack/name).read_bytes()) == expected
        sources = json.loads((args.pack/'sources.json').read_text())
        records = json.loads((args.pack/'records.json').read_text())
        source_hashes = [s['source_sha256'] for s in sources]
        assert len(source_hashes) == len(set(source_hashes)) == 2594
        groups, excluded, expected_indices = defaultdict(lambda: defaultdict(list)), Counter(), []
        for i, source in enumerate(sources):
            meta = source['aliases'][0]
            task = 'positive' if meta['label'] == 1 else meta['phrase']
            if task not in ('positive', 'hey nick', 'hey next', 'hey mix'):
                excluded[task] += 1
                continue
            identity = json.dumps(sorted(meta['speakers'], key=str), sort_keys=True)
            groups[meta['engine']+'/'+task][identity].append(i)
            expected_indices.extend(range(3*i, 3*i+3))
        assert expected_indices == indices
        actual_groups = {key: dict(identities=len(identities),
                                  canonical_sources=sum(map(len, identities.values())))
                         for key, identities in groups.items()}
        assert actual_groups == training['complete_coverage']['groups']
        assert dict(excluded) == training['complete_coverage']['excluded_canonical_by_phrase']
        for row in history['details']:
            meta = records[row['pack_index']]
            assert (row['engine'], row['label'], row['variant'], row['phrase']) == (
                meta['source_group'], meta['label'], meta['variant'],
                sources[meta['source_index']]['aliases'][0]['phrase'])
        for row in actual_rows:
            assert row['training_crop_end'] == records[row['pack_index']]['feature_crop_end']
        pack_check = dict(all_four_pinned_file_hashes_match=True, canonical_sources=len(sources),
                          eligible_variants=len(indices), groups=actual_groups,
                          excluded_canonical_by_phrase=dict(excluded))
    result = dict(
        scope='Independent aggregate/report and optional original-pack metadata reconciliation; no inference or raw-score revalidation.',
        gpu_revision=revision, input_sha256=input_hashes, model_sha256=MODEL,
        export_binds_training_bytes=True, v37_control_fields_equal=CONTROL_FIELDS,
        source_hash_bindings_verified={path: dict(revision=ref, sha256=expected)
                                       for path, (ref, expected) in source_bindings.items()},
        fixed_grid_points=len(expected_grid), eligible_points=eligible_count,
        maximum_grid_macro=maximum_macro, minimum_grid_negative_false_clips=minimum_false,
        zero_event_poisson_upper95=poisson_upper95(0, hours), historical=historical,
        training_support_summary=history['summary'], full_history_transitions=transitions,
        negative_groups=[dict(engine=k[0], phrase=k[1], variant=k[2], **v)
                         for k, v in sorted(grouped.items())],
        actual_float_streaming_witnesses=len(history['sequence_stream_witnesses']),
        maximum_reported_sequence_streaming_residual=max(
            r['max_sequence_stream_error'] for r in history['sequence_stream_witnesses']),
        selected_negative_actual_streaming=dict(
            selection='Union of prior sequence-float or INT8 max-five >= .5, training negatives only',
            variants=len(actual_rows), maximum_reported_residual=actual_max_residual,
            summary=localized_summary),
        original_pack_check=pack_check,
        decision='Reject v40; retain v30 as a frozen research reference, without a parity claim.',
        limits=[
            'No GPU session, model bytes, weights, saved score arrays, audio, board, new development or final set accessed.',
            'Full float sequence is a shared offline graph. Actual streaming covers 20 initial witnesses plus a selected 118-negative union, not all 6597 variants; observed residuals are not all-source bounds.',
            'Peak shifts compare the final processing frame of five scores to the stored crop end, not a phonetic boundary or isolated failure cause.',
            'Maximum five-score values are not latched-event accuracy. Different historical thresholds do not establish matched-FAR improvement.',
            'Intended phrase/voice metadata is not phonetic truth or independent human identity.',
        ])
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('gpu_revision', 'eligible_points',
          'minimum_grid_negative_false_clips', 'full_history_transitions', 'decision')}, indent=2))


if __name__ == '__main__':
    main()
