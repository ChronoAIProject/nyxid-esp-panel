"""Replay training-only adapters, then run the frozen occurrence comparison."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from audit_compact_study import require, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'repository', 'features', 'export', 'training-study'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    repo, root = args.repository.resolve(), args.artifacts.resolve()
    export, features, parent = args.export.resolve(), args.features.resolve(), args.training_study.resolve()
    relative = 'research/wakeword_p4/training/plans/compact-positive-occurrence-v1.json'
    plan_path = repo/relative
    plan = json.loads(plan_path.read_bytes())
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    require(subprocess.check_output(['git', 'show', revision+':'+relative], cwd=repo)
            == plan_path.read_bytes(), 'Plan is not committed')
    require(sha(export/'plan.json') == plan['export_plan_sha256']
            and sha(export/'result.json') == plan['export_result_sha256']
            and sha(export/'compact_dual.tflite') == plan['model_sha256'], 'Export differs')
    require(sha(parent/'plan.json') == plan['parent_study_plan_sha256']
            and sha(parent/'result-summary.json') == plan['parent_study_result_sha256']
            and sha(parent/'input-audit.json') == plan['parent_input_audit_sha256'], 'Parent study differs')
    root.mkdir(parents=True, exist_ok=False)
    source, post = root/'source', root/'auditing'
    source.mkdir(); post.mkdir()
    for path in (repo/'research/wakeword_p4/training').glob('*.py'):
        shutil.copyfile(path, source/path.name)
    for name, digest in plan['postprocessing_source_sha256'].items():
        path = repo/'research/handoff'/name
        require(sha(path) == digest, 'Audit source differs: '+name)
        shutil.copyfile(path, post/name)
    for name, digest in plan['source_sha256'].items():
        require(sha(source/name) == digest, 'Inference source differs: '+name)
    shutil.copyfile(plan_path, root/'plan.json')
    shutil.copyfile(parent/'background-inputs.json', root/'background-inputs.json')
    shutil.copyfile(Path(__file__), root/'launcher.py')
    sys.path.insert(0, str(post))
    from background_inputs import inventory
    require(sha(root/'background-inputs.json') == plan['background_inputs_sha256']
            and inventory(features) == json.loads((root/'background-inputs.json').read_bytes()),
            'Actual background bytes differ')
    require(sha(features/'features.bin') == plan['feature_data_sha256'], 'Scalar feature bytes differ')
    for name, digest in plan['feature_index_sha256'].items():
        require(sha(features/name) == digest, 'Feature index differs: '+name)
    flags = {'PYTHONHASHSEED': '0', 'TF_ENABLE_ONEDNN_OPTS': '0',
             'TF_CPP_MIN_LOG_LEVEL': '2', 'PYTHONUNBUFFERED': '1'}
    os.environ.update(flags)
    command = [sys.executable, str(source/'evaluate_compact_sequence.py'),
        '--plan', str(root/'plan.json'), '--export', str(export),
        '--features', str(features), '--output', str(root/'evaluation')]
    audit_command = [sys.executable, str(post/'audit_compact_inference.py'),
        '--artifacts', str(root), '--repository', str(repo), '--export', str(export),
        '--features', str(features), '--output', str(root/'input-result-audit.json'),
        '--plan-repository-path', relative]
    launch = {'checkout_revision': revision, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': sha(root/'plan.json'), 'command': command, 'audit_command': audit_command,
        'source_sha256': {p.name: sha(p) for p in source.glob('*.py')},
        'launcher_sha256': sha(root/'launcher.py'), 'environment_flags': flags,
        'runtime': {'python': sys.version, **{k: version(k) for k in ['numpy', 'scipy', 'ai-edge-litert']}}}
    (root/'launch.json').write_text(json.dumps(launch, indent=2)+'\n')
    sys.path.insert(0, str(source))
    from keyword_occurrence import OccurrenceDetector, log_probability
    from audit_keyword_occurrence import probability_domain
    rows = json.loads((features/'features.json').read_bytes())
    index = {(r['path'], r['variant']): r for r in rows}
    reserved = {r['sha256'] for r in rows if r['split'] != 'training'}
    raw = np.memmap(features/'features.bin', mode='r', dtype='<u2').reshape(-1, 40)
    detector = OccurrenceDetector(export/'compact_dual.tflite')
    traces = root/'adapter-traces'
    traces.mkdir()
    reports = []
    for number, item in enumerate([*plan['adapter_training_sources'], plan['long_training_source']]):
        row = index[item['path'], item['variant']]
        require(row['split'] == 'training' and row['sha256'] not in reserved
                and all(row[k] == item[k] for k in item if k != 'cohort'), 'Adapter training source differs')
        selected = raw[row['offset']:row['offset']+row['frames']]
        detector.reset()
        p, logits = detector.raw_outputs(selected)
        probability = np.exp(log_probability(logits))
        independent = probability_domain(logits)
        delta = float(np.max(np.abs(probability-independent)))
        require(delta <= 1e-12, 'Independent occurrence probability differs')
        detector.reset()
        chunks = [detector.raw_outputs(selected[at:at+768]) for at in range(0, len(selected)-2, 768)]
        require(np.array_equal(p, np.concatenate([v[0] for v in chunks]))
                and np.array_equal(logits, np.concatenate([v[1] for v in chunks])), 'Raw chunk adapter differs')
        detector.reset()
        chunk_p, chunk_product = detector.predict(selected)
        require(np.array_equal(p, chunk_p) and np.array_equal(p*probability, chunk_product),
                'Occurrence adapter differs')
        path = traces/f'trace-{number:03d}.npz'
        np.savez_compressed(path, scalar=p, logits=logits, occurrence=probability, product=p*probability)
        reports.append({'source': item, 'array': path.relative_to(root).as_posix(), 'sha256': sha(path),
            'steps': len(p), 'crosses_256_step_chunk': len(p) > 256, 'all_logits_scalar_and_product_exact': True,
            'maximum_independent_occurrence_delta': delta})
    require(len(reports) == 41 and reports[-1]['crosses_256_step_chunk'], 'Missing actual long-source gate')
    (root/'adapter-replay.json').write_text(json.dumps({'plan_sha256': sha(root/'plan.json'),
        'model_sha256': plan['model_sha256'], 'passed': True, 'sources': reports}, indent=2)+'\n')
    print(json.dumps({'phase': 'adapter_passed', 'sources': len(reports),
        'steps': sum(r['steps'] for r in reports), 'maximum_independent_occurrence_delta':
        max(r['maximum_independent_occurrence_delta'] for r in reports)}), flush=True)
    for name, cmd in (('evaluation', command), ('audit', audit_command)):
        with (root/(name+'.log')).open('xb') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((root/'evaluation/result.json').read_bytes())
    print(json.dumps({'phase': 'complete', 'advance': result['advance'],
        'strict_gate': result['strict_gate']}), flush=True)


if __name__ == '__main__':
    main()
