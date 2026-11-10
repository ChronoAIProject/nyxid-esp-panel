"""Prespecified auxiliary-only source-energy alignment, with matched training.

Only standalone training positives share the existing short-negative RMS
anchor. Command-followed positives, all main labels and all audio stay intact.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import wave

import numpy as np
from audit_ctc_endpoints import standalone
from lexical_auxiliary import rms_end


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aligned_record(record, texts, activity_end, frames):
    if record['class'] != 0 or not all(standalone(text) for text in texts):
        return dict(record)
    low = int(np.ceil((2.4 + activity_end + .03) / .08))
    high = int(np.ceil((2.4 + activity_end + .32) / .08))
    if not 25 <= low <= high <= frames:
        raise ValueError('New auxiliary bounds exceed warmed original features')
    return record | {'minimum_end': low, 'maximum_end': high}


def prepare(args):
    if args.output.exists() or args.report.exists():
        raise ValueError('Use fresh manifest and report paths')
    plan = json.loads(args.plan.read_text())
    for name, expected in plan['scripts'].items():
        if sha(Path(__file__).parent / name) != expected:
            raise ValueError('Frozen preparation source differs')
    jobs_path = args.data_root / 'head-features-tts-v1/jobs-private.json'
    index_path = args.data_root / 'head-features-tts-v1/index.jsonl'
    for path, key in [(args.manifest, 'original_manifest_sha256'), (args.metadata, 'metadata_sha256'),
                      (jobs_path, 'source_jobs_sha256'), (index_path, 'feature_index_sha256')]:
        if sha(path) != plan[key]:
            raise ValueError('Frozen original source/feature identity differs')
    original = json.loads(args.manifest.read_text())
    metadata, heldout = defaultdict(list), set()
    for row in json.loads(args.metadata.read_text()):
        if row['split'] != 'training':
            heldout.add(row['sha256'])
        elif row['variant'] == 0:
            metadata[row['sha256']].append(row)
    rows = {row['id']: row for row in map(json.loads, index_path.read_text().splitlines())}
    jobs = {row['id']: row for row in json.loads(jobs_path.read_text())}
    activity, standalone_sources, details, records = {}, set(), [], []
    for record in original['records']:
        row, job = rows[record['id']], jobs[record['id']]
        source = record['source_sha256']
        aliases = metadata[source]
        if (source in heldout or not aliases or row['split'] != 'training' or row['kind'] != 'tts'
                or job['split'] != 'training' or job['source_sha256'] != source
                or any(row[key] != record[key] for key in
                       ('source_sha256', 'features_sha256', 'engine', 'label', 'variant'))
                or any((alias['label'], alias['engine']) != (record['label'], record['engine']) for alias in aliases)):
            raise ValueError('Original auxiliary source roles differ')
        texts = [alias['text'] for alias in aliases]
        eligible = record['class'] == 0 and all(standalone(text) for text in texts)
        if eligible:
            standalone_sources.add(source)
            if source not in activity:
                path = Path(job['source'])
                if sha(path) != source:
                    raise ValueError('Original source WAV differs')
                with wave.open(str(path), 'rb') as wav:
                    if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                        raise ValueError('Expected unchanged16kHz mono PCM16')
                    activity[source] = rms_end(np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2'))
        replacement = aligned_record(record, texts, activity.get(source), row['frames'])
        records.append(replacement)
        if replacement != record:
            details.append({'id': record['id'], 'source_sha256': source, 'engine': record['engine'],
                            'variant': record['variant'], 'old_bounds': [record['minimum_end'], record['maximum_end']],
                            'new_bounds': [replacement['minimum_end'], replacement['maximum_end']]})
    if len(standalone_sources) != plan['expected_standalone_training_sources']:
        raise ValueError('Frozen standalone-source coverage differs')
    result = {**original, 'scope': __doc__, 'records': records,
              'parent_auxiliary_manifest_sha256': sha(args.manifest),
              'preparation_plan_sha256': sha(args.plan),
              'positive_anchor_policy': 'Standalone-only original last10ms RMS frame >=max(peak*.01,1), plus2.4s lead; command-followed original markers retained. Auxiliary metadata only, not phonetic labels.'}
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2) + '\n')
    groups = defaultdict(list)
    for detail in details:
        groups[detail['engine']].append(detail)
    report = {'scope': __doc__, 'plan_sha256': sha(args.plan), 'manifest_sha256': sha(args.output),
              'original_manifest_sha256': sha(args.manifest), 'original_records': len(original['records']),
              'standalone_training_sources': len(standalone_sources), 'changed_feature_views': len(details),
              'changed_unique_sources': len({row['source_sha256'] for row in details}),
              'changed_cohorts': [{'engine': engine, 'views': len(values),
                                    'sources': len({value['source_sha256'] for value in values}),
                                    'minimum_end_delta_frames_05_50_95': np.quantile(
                                        [value['new_bounds'][0] - value['old_bounds'][0] for value in values], [.05, .5, .95]).tolist()}
                                   for engine, values in sorted(groups.items())],
              'negative_records_identical': all(left == right for left, right in zip(original['records'], records) if left['class'] != 0),
              'command_positive_records_identical': all(left == right for left, right in zip(original['records'], records)
                                                        if left['class'] == 0 and left['source_sha256'] not in standalone_sources),
              'source_roles_order_ids_classes_and_features_identical': all(
                  {k: v for k, v in left.items() if k not in ('minimum_end', 'maximum_end')} ==
                  {k: v for k, v in right.items() if k not in ('minimum_end', 'maximum_end')}
                  for left, right in zip(original['records'], records)),
              'audio_speed_changed': False, 'speaker_playback': False, 'microphone_access': False,
              'detector_loaded': False, 'accepted': False}
    with args.report.open('x') as out:
        out.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def compare(args):
    from compare_lexical_auxiliary import read_arm
    control, _, left = read_arm(args.control)
    treatment, _, right = read_arm(args.treatment)
    if ({k: v for k, v in control['plan'].items() if k != 'auxiliary_manifest_sha256'} !=
            {k: v for k, v in treatment['plan'].items() if k != 'auxiliary_manifest_sha256'}):
        raise ValueError('Training plans differ beyond auxiliary anchor manifest')
    for key in ('main_batches_sha256', 'auxiliary_source_batches_sha256', 'initial_weights_sha256',
                'feature_inputs', 'parameters', 'training_sources', 'training_frames'):
        if control[key] != treatment[key]:
            raise ValueError('Matched source/class/main-training evidence differs: ' + key)
    if (left['representatives_sha256'] != right['representatives_sha256']
            or left['export']['ops'] != right['export']['ops']
            or left['export']['model_bytes'] != right['export']['model_bytes']):
        raise ValueError('Matched PTQ or exported binary graph differs')
    reference = json.loads(args.reference_complete.read_text())
    if (control['checkpoints'][0]['model_sha256'] != reference['checkpoints'][0]['model_sha256']
            or control['main_batches_sha256'] != reference['main_batches_sha256']
            or control['auxiliary_batches_sha256'] != reference['auxiliary_batches_sha256']):
        raise ValueError('New control does not exactly reproduce retained CE-only control')
    for side, complete in [(left, control), (right, treatment)]:
        side['auxiliary_source_batches_sha256'] = complete['auxiliary_source_batches_sha256']
        side['auxiliary_manifest_sha256'] = complete['auxiliary_manifest_sha256']
    a, b = left['confidence_and_confuser_qualified'], right['confidence_and_confuser_qualified']
    result = {'scope': 'Matched auxiliary-anchor-only ablation on reused original validation; no physical/final acceptance',
              'script_sha256': sha(Path(__file__)), 'control': left, 'treatment': right,
              'same_main_ptq_auxiliary_source_and_class_draws': True,
              'control_model_and_main_auxiliary_batches_exactly_reproduced': True,
              'reference_complete_sha256': sha(args.reference_complete),
              'joint_qualified_macro_difference': b['macro_recall'] - a['macro_recall'] if a and b else None,
              'speaker_playback': False, 'audio_speed_changed': False, 'accepted': False,
              'limits': ['Energy anchors are uncertain auxiliary metadata; source PCM, main labels and evaluation windows stay unchanged.',
                         'Repeated synthetic source views and reused validation are not independent speakers or Hi ESP parity.',
                         'No development/final scoring, P4 resource/numeric or full voice stability acceptance.']}
    with args.report.open('x') as out:
        out.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'control': a, 'treatment': b, 'difference': result['joint_qualified_macro_difference']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    preparation = sub.add_parser('prepare')
    for name in ('data-root', 'manifest', 'metadata', 'plan', 'output', 'report'):
        preparation.add_argument('--' + name, type=Path, required=True)
    comparison = sub.add_parser('compare')
    for name in ('control', 'treatment', 'reference-complete', 'report'):
        comparison.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else compare)(args)
