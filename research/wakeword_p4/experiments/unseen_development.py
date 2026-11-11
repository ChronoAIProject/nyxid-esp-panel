"""Run fixed unseen development only after completed eligible candidate regression."""
import argparse
import re
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE = Path('.local/wakeword')
SCRIPTS = Path('research/wakeword_p4/training')
RESULTS = Path('research/wakeword_p4/results')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(script, *args):
    subprocess.run([sys.executable, str(SCRIPTS/script), *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--check-only', action='store_true', help='Verify old-stage prerequisites without reading/scoring unseen data')
    args = parser.parse_args()
    version = args.candidate
    if not re.fullmatch(r'v[0-9]+', version):
        parser.error('candidate must be v followed by digits')
    model_dir = BASE/f'model-gpu-{version}-ami-extra'
    model = model_dir/'hey_nyx.tflite'
    training = json.loads((model_dir/'training.json').read_text())
    digest = sha(model)
    assert training['steps'] == 120000 and training['model_sha256'] == digest
    audit = json.loads((BASE/f'{version}-export-audit.json').read_text())
    probe = json.loads((BASE/f'{version}-streaming-probe.json').read_text())
    regression = json.loads((BASE/f'{version}-reused-development-exploratory-calibration-v1.json').read_text())
    assert audit['model_sha256'] == probe['model_sha256'] == regression['model_sha256'] == digest
    assert audit['all_saved_weights_finite'] and audit['frozen_bn_unchanged']
    assert probe['reset_reproducibility_passed']
    assert audit['training_sha256'] == sha(model_dir/'training.json')
    assert regression['synthetic_report_sha256'] == [
        sha(BASE/f'{version}-phone-development-historical-threshold-v1/report.json'),
        sha(BASE/f'{version}-vctk-development-historical-threshold-v1/report.json')]
    assert regression['background_report_sha256'] == sha(BASE/f'{version}-voxconverse-development-v1/report.json')
    if not regression['viable_operating_point']:
        raise RuntimeError(f'Existing {version} development has no eligible point; preserve unseen data, review rejection first')
    if regression['final_data_used'] or not regression['validation_only'] or len(regression['grid']) != 138:
        raise RuntimeError('Wrong old-stage calibration scope')
    selected = regression['selected']
    if (not selected['eligible'] or selected['background_upper_95'] > .5
            or selected['synthetic_macro_clip_trigger'] < .48125
            or any(g['false_activation_clips'] or g['pre_generation_positive_events']
                   for g in selected['groups'].values())):
        raise RuntimeError('Invalid old-stage selected point')
    if args.check_only:
        print(json.dumps(dict(candidate=version, old_stage_passed=True,
                              unseen_scored=False, model_sha256=digest)), flush=True)
        return
    # No waits/restarts/publication/promotion. Any missing/partial output stops.
    manifest = BASE/'slr83-development-replay-v2/manifest.json'
    background = BASE/'voxconverse-new-development-features-v1'
    assert sha(manifest) == '88cfa446880e8a562927221d552a544bb72f204af071cd6867c7e1284db80415'
    assert sha(background/'manifest.json') == 'c65b0e566384bc3c54d36d4766d6e7abed731e01263354f5d05a924a454b2170'
    v30 = BASE/'model-gpu-v30-ami-extra/hey_nyx.tflite'
    calibration30 = RESULTS/'4090-v30-expanded-calibration.json'
    candidate_calibration = BASE/f'eval-{version}-ami-extra-validation/report.json'
    qemu = BASE/'qemu-xtensa-release/qemu/bin/qemu-system-xtensa'
    elf = BASE/'wakenet-esp32-replay-build/nyx_wakenet_esp32_replay.elf'
    flash = BASE/'wakenet-esp32-replay-build/qemu-flash.bin'
    nm = BASE/'idf-tools/tools/xtensa-esp-elf/esp-14.2.0_20260121/xtensa-esp-elf/bin/xtensa-esp32-elf-nm'
    common = ['--qemu', qemu, '--elf', elf, '--flash', flash, '--nm', nm]
    assert sha(v30) == '04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c'
    paired = Path('/tmp/nyx-slr-dev-v30-v1')
    reserved = [paired, BASE/'slr83-development-official-reports-v1.json',
                Path('/tmp/nyx-vox-new-esp-v1')]
    for name in ['v30', version]:
        reserved.extend([BASE/(name+'-slr83-development-v1'),
                         BASE/(name+'-voxconverse-new-development-v1')])
    if any(path.exists() for path in reserved):
        raise RuntimeError('Unseen stage output already exists; review original run rather than rescore')
    run('compare_paired_validation.py', '--manifest', manifest, '--nyx-model', v30,
        '--nyx-calibration', calibration30, *common, '--output', paired)
    official = BASE/'slr83-development-official-reports-v1.json'
    with official.open('x') as f:
        json.dump([json.loads((paired/str(i)/'report.json').read_text()) for i in range(90)], f, indent=2)
        f.write('\n')
    scored = {}
    for run_version, path, calibration in [('v30', v30, calibration30), (version, model, candidate_calibration)]:
        root = BASE/(run_version+'-slr83-development-v1')
        extra = ['--baseline', paired/'report.json'] if run_version == 'v30' else []
        run('score_phone_development.py', '--manifest', manifest, '--model', path,
            '--calibration', calibration, '--official-reports', official, '--output', root, *extra)
        scored[run_version] = root
        run('synthetic_presentation.py', '--scores', root, '--manifest', manifest,
            '--output', BASE/(run_version+'-slr83-presentation-historical-threshold-v1.json'))
        run('evaluate_frozen_human.py', '--model', path, '--frozen-report', calibration,
            '--features', background, '--output', BASE/(run_version+'-voxconverse-new-development-v1'),
            '--corpus-name', 'VoxConverse dev2/dev3 separate development stage', '--identity-unit', 'recording')
    selected = BASE/f'{version}-slr83-new-development-calibration-v1.json'
    run('calibrate_slr83_development.py', '--manifest', manifest,
        '--synthetic', scored[f'{version}'], '--baseline', scored['v30'],
        '--background', BASE/f'{version}-voxconverse-new-development-v1', '--output', selected)
    if json.loads(selected.read_text())['viable_operating_point']:
        run('synthetic_presentation.py', '--scores', scored[f'{version}'], '--manifest', manifest,
            '--threshold-calibration', selected, '--output', BASE/f'{version}-slr83-presentation-selected-threshold-v1.json')
    run('replay_voxconverse_official.py', '--manifest', background/'manifest.json',
        *common, '--output', '/tmp/nyx-vox-new-esp-v1')
    print('Unseen development completed; review evidence, no final-data access or automatic promotion', flush=True)


if __name__ == '__main__':
    main()
