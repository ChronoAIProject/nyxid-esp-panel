"""Bind fresh development raw collection only after completed numerical admission."""
import importlib.metadata
import json
from pathlib import Path

from collect_complete_development_v1 import SCHEMA, POLICY, PACKAGES, runtime_identity_files, validate_materials
from run_hey_broad_trained_quantization_v1 import ROOT, published
from plan_hey_broad_trained_quantization_v1 import sha


def main():
    reports = ROOT/'results'
    admission_path = reports/'4090-hey-broad-bounded-consistency-numerical-admission-v1.json'
    published(admission_path)
    admission = json.loads(admission_path.read_text())
    assert admission['scope'] == 'numerical-only-fixed-treatment'
    assert admission['assessment']['condition'] == 'bounded_logit_consistency'
    assert admission['assessment']['all_original_gates_passed']
    assert admission['assessment']['screen_cases'] == admission['assessment']['reference_cases'] == 270
    assert set(admission['models']) == {'hey_willow', 'hey_nyx'}
    for path, expected in admission['files'].items():
        assert sha(path) == expected, path
    for model in admission['models'].values():
        assert sha(model['path']) == model['sha256']
        assert sha(model['weights_path']) == model['weights_sha256']
    folders = {t: Path(m['weights_path']).parent for t, m in admission['models'].items()}
    receipt_path = reports/'4090-jenny-development-replay-manifest-v1.json'
    receipt = json.loads(receipt_path.read_text())
    synthetic = Path(receipt['path'])
    assert sha(synthetic) == receipt['sha256']
    for path, expected in receipt['files'].items():
        assert sha(path) == expected
    preparation = reports/'4090-jenny-earnings-complete-preparation-v1.json'
    prep = json.loads(preparation.read_text())
    background = Path(prep['manifest_path'])
    assert sha(background) == prep['manifest_sha256']
    history_path = reports/'4090-jenny-earnings-history-overlap-v1.json'
    history = json.loads(history_path.read_text())
    assert history['manifest_sha256'] == sha(background)
    assert history['no_exact_encoded_or_pcm_hash_matches'] and not history['matches']
    assert sha(history['inventory_path']) == history['inventory_sha256']
    frontend_path = reports/'4090-jenny-background-complete-frontend-audit-v1.json'
    frontend = json.loads(frontend_path.read_text())
    assert frontend['manifest_sha256'] == sha(background)
    assert frontend['history_report_sha256'] == sha(history_path)
    assert frontend['all_complete_arrays_exact'] and len(frontend['cases']) == 12
    native_path = reports/'4090-jenny-development-independent-native-audit-v1.json'
    native = json.loads(native_path.read_text())
    assert native['rows'] == 132 and native['all_native_process_quality_pass']
    render_path = reports/'4090-jenny-development-render-independent-audit-v1.json'
    render = json.loads(render_path.read_text())
    assert render['independently_qualified'] == 396 and not render['failures']
    paths = [Path(__file__).resolve().relative_to(Path.cwd()), admission_path, receipt_path,
             synthetic, preparation, background, history_path, frontend_path, native_path, render_path]
    paths += [Path(p) for p in receipt['files']]
    paths += [Path(p) for p in admission['files']]
    paths += [Path(history['inventory_path'])]
    here = Path(__file__).parent
    paths += [here/n for n in ('collect_complete_development_v1.py', 'complete_feature_stream_v1.py',
                               'uint8_stream_detector_v1.py', 'run_hey_broad_trained_quantization_v1.py',
                               'plan_hey_broad_trained_quantization_v1.py',
                               'calibrate_complete_development_v1.py',
                               'complete_development_metrics_v1.py',
                               'hey_development_metrics_v1.py', 'uint8_event_policy_v1.py')]
    paths += [ROOT/'training/features.py'] + runtime_identity_files()
    models = {}
    for target, folder in folders.items():
        path = Path(admission['models'][target]['path'])
        models[target] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size, target_id=target,
                             output=str(Path('.local/wakeword/jenny-bounded-consistency-complete-raw-v1')/target))
        paths += [path, folder/'final.weights.h5', folder/'training.json',
                  folder/'actual-batch-schedule.jsonl', folder/'initial-inventory.json',
                  folder/'final-inventory.json', folder/'final-optimizer.npz']
    frozen = Path('.local/wakeword/model-gpu-v30-ami-extra/hey_nyx.tflite')
    assert frozen.stat().st_size == 54440
    assert sha(frozen) == '04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c'
    models['v30'] = dict(path=str(frozen), sha256=sha(frozen), bytes=54440, target_id='hey_nyx',
                         fixed_reference_thresholds=[0.91796875],
                         output='.local/wakeword/jenny-bounded-consistency-complete-raw-v1/v30')
    paths.append(frozen)
    for manifest, phase in ((synthetic, 'synthetic'), (background, 'background')):
        rows = json.loads(manifest.read_text())['records']
        assert len(rows) == (396 if phase == 'synthetic' else 12)
        for row in rows:
            paths.append(Path(row['path'] if phase == 'synthetic' else row['wav_path']))
            if 'frontend_path' in row:
                paths.append(Path(row['frontend_path']))
    result = dict(schema=SCHEMA, tail_policy=POLICY, authorizes='development-raw-collection-only',
                  files={str(p): sha(p) for p in sorted(set(paths))}, models=models,
                  versions={p: importlib.metadata.version(p) for p in PACKAGES},
                  materials={phase: dict(path=str(p), sha256=sha(p), role='new-development')
                             for phase, p in (('synthetic', synthetic), ('background', background))},
                  producer_admission='Fresh development only, bounded exact-history and saved synthesis quality admitted. Original preparation reports remain unchanged. Fixed prospectively selected bounded_logit_consistency only; separate numerical receipt authenticated. Raw collection only; no calibration authorization, independent acceptance or promotion.',
                  limitations='Single author-direct Jenny voice; different export bytes and bounded history do not prove speaker/pretraining independence. No protected final sources opened. Official reference execution requires a separate lock for identical PCM.')
    validate_materials(result)
    output = reports/'4090-jenny-bounded-consistency-complete-development-execution-v1.json'
    with output.open('x') as destination:
        destination.write(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
