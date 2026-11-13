"""Collect complete raw development streams under a published execution lock.

No historical model directory, model byte size, source count or threshold is
embedded. This collector does not perform calibration or open acceptance.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from complete_feature_stream_v1 import POLICY, predict_complete_features

SCHEMA = 'complete-development-stream-execution-v1'
PACKAGES = ('numpy', 'scipy', 'soundfile', 'ai-edge-litert', 'pymicro-features')
ROLES = ('new-development', 'reused-development')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require_bound(path, files):
    path = Path(path)
    candidates = {str(path), str(path.resolve())}
    try:
        candidates.add(str(path.resolve().relative_to(Path.cwd())))
    except ValueError:
        pass
    expected = [files[key] for key in candidates if key in files]
    if not expected:
        # Filesystem aliases (e.g. macOS /var and /private/var) may use a
        # different absolute spelling while naming exactly the same file.
        canonical = path.resolve()
        expected = [value for key, value in files.items()
                    if Path(key).resolve() == canonical]
    if not expected or any(value != sha(path) for value in expected):
        raise ValueError('Missing or changed execution-bound file: '+str(path))


def runtime_identity_files():
    """Use when preparing the lock; native frontend lives outside its wrapper."""
    paths = set()
    for name in ('ai_edge_litert', 'pymicro_features', 'micro_features_cpp'):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError('Missing runtime/frontend: '+name)
        path = Path(spec.origin)
        if spec.submodule_search_locations:
            paths.update(p for p in path.parent.rglob('*')
                         if p.is_file() and p.suffix in ('.py', '.so', '.pyd', '.dylib'))
        else:
            paths.add(path)
    return sorted(paths)


def source_paths(row, phase):
    if phase == 'synthetic':
        return Path(row['path']), row['sha256']
    return Path(row['wav_path']), row['wav_sha256']


def validate_materials(lock):
    manifests = {}
    for phase in ('synthetic', 'background'):
        material = lock['materials'][phase]
        if material['role'] not in ROLES:
            raise ValueError('Only explicitly labelled development materials allowed')
        path = Path(material['path'])
        require_bound(path, lock['files'])
        if sha(path) != material['sha256']:
            raise ValueError('Material manifest hash differs from execution')
        manifest = json.loads(path.read_text())
        rows = manifest['records']
        if not isinstance(rows, list) or not rows:
            raise ValueError('Empty development phase')
        for row in rows:
            if type(row['samples']) is not int or row['samples'] <= 0:
                raise ValueError('Invalid original sample count')
            wav, expected = source_paths(row, phase)
            require_bound(wav, lock['files'])
            if sha(wav) != expected:
                raise ValueError('Source differs from frozen manifest')
            if phase == 'background' and 'frontend_path' in row:
                cached = Path(row['frontend_path'])
                require_bound(cached, lock['files'])
                if sha(cached) != row['frontend_sha256']:
                    raise ValueError('Cached full frontend differs from manifest')
        manifests[phase] = rows
    return manifests


def collect_model(lock, lock_sha256, model_id, manifests, detector, frontend, read_pcm):
    """Runtime dependencies are explicit so constructed tests need no model."""
    model = lock['models'][model_id]
    dest = Path(model['output'])
    dest.mkdir(parents=True, exist_ok=False)
    phases = {}
    for phase in ('synthetic', 'background'):
        group = dest/phase
        group.mkdir()
        results = []
        with (group/'records.jsonl').open('x') as log:
            for index, row in enumerate(manifests[phase]):
                wav, expected = source_paths(row, phase)
                if sha(wav) != expected:
                    raise ValueError('Source changed after execution admission')
                pcm, rate = read_pcm(wav)
                if pcm.dtype != np.int16 or pcm.ndim != 1 or rate != 16000 or len(pcm) != row['samples']:
                    raise ValueError('Source must be the full original 16kHz mono PCM16')
                if hashlib.sha256(pcm.astype('<i2', copy=False).tobytes()).hexdigest() != row['pcm_sha256']:
                    raise ValueError('Decoded PCM differs from frozen manifest')
                if phase == 'background' and 'frontend_path' in row:
                    if sha(row['frontend_path']) != row['frontend_sha256']:
                        raise ValueError('Cached frontend changed after admission')
                    raw = np.load(row['frontend_path'], allow_pickle=False)
                else:
                    raw = frontend(pcm)
                detector.reset()
                first, clocks, tail = predict_complete_features(detector, raw, row['samples'])
                detector.reset()
                second, second_clocks, second_tail = predict_complete_features(detector, raw, row['samples'])
                np.testing.assert_array_equal(first, second)
                np.testing.assert_array_equal(clocks, second_clocks)
                if tail != second_tail:
                    raise ValueError('Reset-repeat tail metadata differs')
                trace = group/f'{index:06}.npz'
                with trace.open('xb') as output:
                    np.savez(output, raw=first, sample_end=clocks)
                record = dict(record=row, trace_path=str(trace), trace_sha256=sha(trace),
                    raw_steps=len(first), reset_repeat_exact=True, tail=tail)
                results.append(record)
                log.write(json.dumps(record, sort_keys=True)+'\n')
                log.flush()
                print(json.dumps(dict(model=model_id, phase=phase, completed=index+1)), flush=True)
        phases[phase] = dict(material=lock['materials'][phase], records=results,
            records_sha256=sha(group/'records.jsonl'))
    report = dict(schema='complete-development-stream-report-v1', model_id=model_id,
        model=model, execution_sha256=lock_sha256, tail_policy=POLICY, phases=phases,
        threshold_selected=False, acceptance_scored=False, candidate_promoted=False,
        limits='Development raw collection only; explicit new/reused roles do not prove source independence. Complete traces and reset equality do not establish recognition/FAR or trained numeric/P4 acceptance.')
    with (dest/'report.json').open('x') as output:
        output.write(json.dumps(report, indent=2)+'\n')
    return report


def load_execution(path, published_ref):
    relative = path.resolve().relative_to(Path.cwd())
    published = subprocess.check_output(['git', 'show', published_ref+':'+str(relative)])
    if published != path.read_bytes():
        raise ValueError('Execution lock must match its published bytes')
    lock = json.loads(published)
    if lock['schema'] != SCHEMA or lock['tail_policy'] != POLICY:
        raise ValueError('Wrong complete-development execution schema/policy')
    if lock.get('authorizes') != 'development-raw-collection-only':
        raise ValueError('Wrong execution scope')
    for path, expected in lock['files'].items():
        if sha(path) != expected:
            raise ValueError('Changed execution file: '+path)
    for package in PACKAGES:
        if importlib.metadata.version(package) != lock['versions'][package]:
            raise ValueError('Changed runtime version: '+package)
    here = Path(__file__).resolve().parent
    for source in (Path(__file__), here/'complete_feature_stream_v1.py',
                   here/'uint8_stream_detector_v1.py', here.parent/'training/features.py'):
        require_bound(source, lock['files'])
    for source in runtime_identity_files():
        require_bound(source, lock['files'])
    for model in lock['models'].values():
        require_bound(model['path'], lock['files'])
        if sha(model['path']) != model['sha256'] or Path(model['path']).stat().st_size != model['bytes']:
            raise ValueError('Changed frozen model')
    return lock, hashlib.sha256(published).hexdigest(), validate_materials(lock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-lock', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--published-ref', default='origin/research-hye-nyx-voice-gpu')
    args = parser.parse_args()
    lock, execution_sha, manifests = load_execution(args.execution_lock, args.published_ref)
    if args.model not in lock['models']:
        parser.error('Model not listed in execution lock')
    if Path(lock['models'][args.model]['output']).exists():
        raise ValueError('Keep existing/partial output; use the registered fresh directory')
    # Only after complete byte/material admission import actual inference/audio IO.
    import soundfile as sf
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'training'))
    from features import features
    from uint8_stream_detector_v1 import Uint8StreamDetector
    detector = Uint8StreamDetector(lock['models'][args.model]['path'])
    collect_model(lock, execution_sha, args.model, manifests, detector, features,
                  lambda path: sf.read(path, dtype='int16'))


if __name__ == '__main__':
    main()
