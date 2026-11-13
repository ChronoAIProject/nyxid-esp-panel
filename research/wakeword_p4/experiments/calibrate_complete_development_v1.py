"""Score complete collected traces; never run a model or decode audio.

Run only after the producer admits the completed candidates and new development
under its separate scoring contract. A raw-collection lock alone is not that
admission. The same published lock must bind this consumer and target mapping.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np

from collect_complete_development_v1 import SCHEMA, sha, require_bound
from complete_feature_stream_v1 import POLICY
from complete_development_metrics_v1 import (
    GRID, validate_clips, synthetic_metrics, background_metrics, eligible, choose,
)

MIN_BACKGROUND_HOURS = 8


def read_execution(path, published_ref):
    path = Path(path)
    relative = path.resolve().relative_to(Path.cwd())
    published = subprocess.check_output(['git', 'show', published_ref+':'+str(relative)])
    if published != path.read_bytes():
        raise ValueError('Execution must match its published bytes')
    lock = json.loads(published)
    if lock['schema'] != SCHEMA or lock['tail_policy'] != POLICY:
        raise ValueError('Wrong complete collector schema or tail policy')
    if lock['authorizes'] != 'development-raw-collection-only':
        raise ValueError('Expected original raw collection execution')
    for name in ('numpy', 'scipy'):
        if importlib.metadata.version(name) != lock['versions'][name]:
            raise ValueError('Changed scoring dependency: '+name)
    here = Path(__file__).resolve().parent
    for name in ('calibrate_complete_development_v1.py', 'complete_development_metrics_v1.py',
                 'hey_development_metrics_v1.py', 'uint8_event_policy_v1.py',
                 'collect_complete_development_v1.py', 'complete_feature_stream_v1.py'):
        require_bound(here/name, lock['files'])
    return lock


def read_groups(lock, execution_sha, model_id):
    model = lock['models'][model_id]
    if model['target_id'] not in ('hey_willow', 'hey_nyx'):
        raise ValueError('Explicit execution-bound custom target_id required')
    base = Path(model['output'])
    report_path = base/'report.json'
    report = json.loads(report_path.read_text())
    if (report['schema'] != 'complete-development-stream-report-v1'
            or report['model_id'] != model_id or report['model'] != model
            or report['execution_sha256'] != execution_sha or report['tail_policy'] != POLICY
            or report['threshold_selected'] is not False or report['acceptance_scored'] is not False
            or report['candidate_promoted'] is not False):
        raise ValueError('Collector report differs from frozen execution or scope')
    manifests = {}; groups = {}
    for phase in ('synthetic', 'background'):
        material = lock['materials'][phase]
        if material['role'] != 'new-development':
            raise ValueError('This selector requires new development; reused data is regression only')
        manifest_path = Path(material['path'])
        require_bound(manifest_path, lock['files'])
        if sha(manifest_path) != material['sha256']:
            raise ValueError('Changed expected material manifest')
        manifests[phase] = json.loads(manifest_path.read_text())
        rows = manifests[phase]['records']
        observed = report['phases'][phase]
        records_path = base/phase/'records.jsonl'
        receipts = [json.loads(line) for line in records_path.read_text().splitlines()]
        if (observed['material'] != material or observed['records'] != receipts
                or sha(records_path) != observed['records_sha256']
                or [r['record'] for r in receipts] != rows):
            raise ValueError('Incomplete or changed collector records')
        pairs = []
        for index, receipt in enumerate(receipts):
            trace = Path(receipt['trace_path'])
            if (trace.resolve() != (base/phase/f'{index:06}.npz').resolve()
                    or sha(trace) != receipt['trace_sha256']
                    or receipt['reset_repeat_exact'] is not True):
                raise ValueError('Changed trace path/hash or missing reset equality')
            with np.load(trace, allow_pickle=False) as arrays:
                if set(arrays.files) != {'raw', 'sample_end'}:
                    raise ValueError('Expected complete raw and sample clocks')
                raw, clocks = arrays['raw'], arrays['sample_end']
            row = receipt['record']; samples = row['samples']
            if type(samples) is not int or samples <= 0:
                raise ValueError('Invalid original sample count')
            frames = max(0, (samples+159)//160-2)
            calls = (frames+2)//3
            expected_clocks = np.minimum((3*np.arange(1, calls+1, dtype=np.int64)+2)*160, samples)
            tail = receipt['tail']
            if (receipt['raw_steps'] != calls or raw.dtype != np.uint8 or raw.shape != (calls,)
                    or clocks.dtype != np.int64 or not np.array_equal(clocks, expected_clocks)
                    or any(tail[key] != value for key,value in dict(policy=POLICY,
                        original_samples=samples, original_feature_frames=frames,
                        final_feature_padding=(-frames)%3, input_feature_frames=frames+(-frames)%3,
                        raw_steps=calls).items())):
                raise ValueError('Incomplete raw steps, sample clocks or tail metadata')
            pairs.append((row, raw))
        validate_clips(pairs, rows)
        groups[phase] = pairs
    return groups, manifests, report_path


def score(groups, manifests, model):
    hours = sum(r['samples'] for r,_ in groups['background'])/16000/3600
    if hours < MIN_BACKGROUND_HOURS:
        raise ValueError('Need at least eight whole original background hours')
    references = model.get('fixed_reference_thresholds', [])
    if not isinstance(references, list) or len(set(references)) != len(references):
        raise ValueError('Invalid execution-bound reference thresholds')
    points = []
    for threshold in sorted(set(GRID+references)):
        synthetic = synthetic_metrics(groups['synthetic'], model['target_id'], threshold,
            manifests['synthetic']['records'], manifests['synthetic']['standalone_carriers'], details=True)
        background = background_metrics(groups['background'], threshold, manifests['background']['records'])
        points.append(dict(threshold=threshold, synthetic=synthetic,
            background_events=background['events'], background_hours=background['hours'],
            background_original_samples=background['original_samples'],
            background_upper95=background['upper95'], background_sources=background['sources'],
            registered_grid=threshold in GRID, eligible=eligible(synthetic, background['upper95'])))
    selected = choose([p for p in points if p['registered_grid']])
    return dict(grid=points, selected=selected, threshold=selected['threshold'] if selected else None,
        eligible_points=sum(p['eligible'] for p in points if p['registered_grid']),
        fixed_reference_points=[p for p in points if p['threshold'] in references])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-lock', type=Path, required=True)
    parser.add_argument('--published-ref', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    lock = read_execution(args.execution_lock, args.published_ref)
    if args.model not in lock['models']:
        parser.error('Model not listed in frozen execution')
    output = Path(lock['models'][args.model]['output'])/'calibration.json'
    if output.exists():
        raise FileExistsError('Preserve original calibration; no overwrite')
    groups, manifests, report_path = read_groups(lock, sha(args.execution_lock), args.model)
    result = dict(schema='complete-development-calibration-v1', model_id=args.model,
        target_id=lock['models'][args.model]['target_id'], model=lock['models'][args.model],
        execution_sha256=sha(args.execution_lock), collector_report_sha256=sha(report_path),
        materials=lock['materials'], **score(groups, manifests, lock['models'][args.model]),
        candidate_promoted=False, acceptance_authorized=False,
        limits='New-development threshold selection under the producer scoring contract only. Generated boundaries are not human phoneme truth; nominal Poisson FAR ignores clustering. No model/audio/frontend executed, no independent acceptance or P4 parity established. Raw-collection execution alone is not candidate/source admission.')
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(model=args.model, threshold=result['threshold'], eligible_points=result['eligible_points'])))


if __name__ == '__main__':
    main()
