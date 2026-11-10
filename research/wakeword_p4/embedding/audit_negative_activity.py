"""Audit original training-negative energy without scoring or dropping sources."""
import argparse
from collections import Counter
import json
from pathlib import Path
import wave

import numpy as np

from negative_activity import sha


def energy_statistics(pcm):
    if pcm.ndim != 1 or not len(pcm):
        raise ValueError('Expected nonempty mono PCM')
    values = pcm.astype(np.float64)
    blocks = np.pad(values, (0, (-len(values)) % 160)).reshape(-1, 160)
    rms = np.sqrt(np.mean(blocks ** 2, axis=1))
    threshold = max(1., float(rms.max()) * .01)
    active = np.flatnonzero(rms >= threshold)
    return {'samples': len(pcm), 'seconds': len(pcm) / 16000,
            'nonzero_samples': int(np.count_nonzero(pcm)),
            'sample_peak_absolute': int(np.max(np.abs(values))),
            'whole_source_rms': float(np.sqrt(np.mean(values ** 2))),
            'peak_10ms_rms': float(rms.max()), 'activity_threshold': threshold,
            'active_blocks': len(active),
            'activity_start_seconds': int(active[0]) * .01 if len(active) else None,
            'activity_stop_seconds': min(len(pcm), (int(active[-1]) + 1) * 160) / 16000 if len(active) else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'plan', 'failure', 'output', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise ValueError('Use fresh audit paths')
    plan = json.loads(args.plan.read_text())
    if sha(Path(__file__).with_name('negative_activity.py')) != plan['preparation_script_sha256']:
        raise ValueError('Audit requires original frozen v1 energy rule')
    failure = json.loads(args.failure.read_text())
    if failure['returncode'] != 1 or 'Silent source has no sampling interval' not in failure['stderr']:
        raise ValueError('Expected preserved v1 preparation failure')
    rows, jobs, heldout, positives = [], {}, set(), set()
    for directory, binding in plan['feature_inputs'].items():
        root = args.data_root / directory
        for name, expected in binding.items():
            if sha(root / name) != expected:
                raise ValueError('Frozen feature/source input differs')
        records = json.loads((root / 'jobs-private.json').read_text())
        jobs[directory] = {row['id']: row for row in records}
        if len(jobs[directory]) != len(records):
            raise ValueError('Ambiguous original source identity')
        for line in (root / 'index.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['split'] != 'training':
                heldout.add(row['source_sha256'])
            elif row['kind'] == 'tts' and not row['label']:
                rows.append(row | {'_feature_directory': directory})
            elif row['label']:
                positives.add(row['source_sha256'])
    sources = {}
    for row in rows:
        source = row['source_sha256']
        job = jobs[row['_feature_directory']][row['id']]
        if source in heldout or source in positives or any(row[k] != job[k] for k in
                ('source_sha256', 'engine', 'variant', 'split', 'kind', 'label')):
            raise ValueError('Training-negative role or original identity differs')
        if source not in sources:
            path = Path(job['source'])
            if sha(path) != source:
                raise ValueError('Original source WAV differs')
            with wave.open(str(path), 'rb') as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                    raise ValueError('Expected original 16kHz mono PCM16')
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
            sources[source] = {'source_sha256': source, **energy_statistics(pcm), 'views': []}
        stats = sources[source]
        if abs(stats['seconds'] - row['source_seconds']) > 1e-9:
            raise ValueError('Original source timebase differs')
        stats['views'].append({k: row[k] for k in
            ('id', '_feature_directory', 'features_sha256', 'engine', 'variant', 'frames')})
    affected = [row for row in sources.values() if row['active_blocks'] == 0]
    manifest = {'scope': __doc__, 'script_sha256': sha(Path(__file__)),
                'preparation_plan_sha256': sha(args.plan), 'v1_failure_sha256': sha(args.failure),
                'sources': sorted(sources.values(), key=lambda row: row['source_sha256']),
                'all_source_hashes_checked': True, 'all_sources_training_negative_only': True,
                'heldout_and_positive_disjoint': True, 'source_audio_unchanged': True,
                'speaker_playback': False, 'microphone_access': False, 'accepted': False}
    with args.output.open('x') as out:
        out.write(json.dumps(manifest, indent=2) + '\n')
    cohorts = Counter((view['engine'], view['_feature_directory']) for source in affected for view in source['views'])
    report = {k: v for k, v in manifest.items() if k != 'sources'}
    report.update(manifest_sha256=sha(args.output), original_sources=len(sources), feature_views=len(rows),
                  undefined_activity_sources=len(affected),
                  undefined_activity_views=sum(len(row['views']) for row in affected),
                  all_zero_sources=sum(row['nonzero_samples'] == 0 for row in affected),
                  nonzero_below_rms_floor_sources=sum(row['nonzero_samples'] != 0 for row in affected),
                  affected_cohorts=[{'engine': e, 'feature_directory': d, 'views': n} for (e, d), n in sorted(cohorts.items())],
                  affected_sources=[{k: v for k, v in row.items() if k != 'views'} |
                                    {'feature_views': len(row['views'])} for row in sorted(affected, key=lambda r: r['source_sha256'])],
                  limits=['No detector inference, text/pronunciation judgment, data filtering, label changes or fitting.',
                          'Undefined activity means no 10ms block reaches max(peak_RMS*.01,1); not necessarily all-zero PCM.',
                          'A follow-up policy must be frozen explicitly; v1 remains a failed preparation, not a training result.'])
    with args.report.open('x') as out:
        out.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('affected_sources', 'limits')}), flush=True)


if __name__ == '__main__':
    main()
