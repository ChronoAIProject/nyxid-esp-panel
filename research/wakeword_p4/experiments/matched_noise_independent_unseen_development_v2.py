"""Run fixed unseen development only after completed eligible matched-noise-independent-formal-v1 regression."""
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
    model_dir = BASE/'model-matched-noise-independent-formal-v1'
    model = model_dir/'hey_nyx.tflite'
    training = json.loads((model_dir/'training.json').read_text())
    digest = sha(model)
    assert training['steps'] == 120000 and training['model_sha256'] == digest
    audit = json.loads((BASE/'matched-noise-independent-formal-v1-export-audit.json').read_text())
    probe = json.loads((BASE/'matched-noise-independent-formal-v1-streaming-probe.json').read_text())
    regression = json.loads((BASE/'matched-noise-independent-formal-v1-reused-development-exploratory-calibration-v1.json').read_text())
    assert audit['model_sha256'] == probe['model_sha256'] == regression['model_sha256'] == digest
    assert audit['all_saved_weights_finite'] and audit['frozen_bn_unchanged']
    assert probe['reset_reproducibility_passed']
    if not regression['viable_operating_point']:
        raise RuntimeError('Existing matched-noise-independent-formal-v1 development has no eligible point; preserve unseen data, review rejection first')
    assert digest == '83cc9b613062cc706c67736aa6ae500c89d69bbd50546137d211ecf281c00cca'
    assert probe['reset_reproducibility_passed']
    # Require all outputs fresh before consuming any reserved development scores.
    fresh = ['matched-noise-unseen-v30-official-paired-v2',
             'slr83-development-official-reports-v1.json',
             'v30-slr83-development-v1', 'v30-voxconverse-new-development-v1',
             'v30-slr83-presentation-historical-threshold-v1.json',
             'matched-noise-unseen-official-background-v1',
             'matched-noise-independent-formal-v1-slr83-development-v1',
             'matched-noise-independent-formal-v1-voxconverse-new-development-v1',
             'matched-noise-independent-formal-v1-slr83-new-development-calibration-v1.json',
             'matched-noise-independent-formal-v1-slr83-presentation-historical-threshold-v1.json',
             'matched-noise-independent-formal-v1-slr83-presentation-selected-threshold-v1.json']
    assert not Path('/tmp/nyx-mn-slr-v2').exists()
    assert not Path('/tmp/nyx-mn-vox-v2').exists()
    for name in fresh:
        assert not (BASE/name).exists(), name
    # No waits/restarts/publication/promotion. Any missing/partial output stops.
    manifest = BASE/'slr83-development-replay-v2/manifest.json'
    background = BASE/'voxconverse-new-development-features-v1'
    assert sha(manifest) == '88cfa446880e8a562927221d552a544bb72f204af071cd6867c7e1284db80415'
    assert sha(background/'manifest.json') == 'c65b0e566384bc3c54d36d4766d6e7abed731e01263354f5d05a924a454b2170'
    v30 = BASE/'model-gpu-v30-ami-extra/hey_nyx.tflite'
    calibration30 = RESULTS/'4090-v30-expanded-calibration.json'
    calibration36 = BASE/'eval-matched-noise-independent-formal-v1-ami-extra-validation/report.json'
    qemu = BASE/'qemu-xtensa-release/qemu/bin/qemu-system-xtensa'
    elf = BASE/'wakenet-esp32-replay-build/nyx_wakenet_esp32_replay.elf'
    flash = BASE/'wakenet-esp32-replay-build/qemu-flash.bin'
    nm = BASE/'idf-tools/tools/xtensa-esp-elf/esp-14.2.0_20260121/xtensa-esp-elf/bin/xtensa-esp32-elf-nm'
    common = ['--qemu', qemu, '--elf', elf, '--flash', flash, '--nm', nm]
    paired = Path('/tmp/nyx-mn-slr-v2')
    run('compare_paired_validation.py', '--manifest', manifest, '--nyx-model', v30,
        '--nyx-calibration', calibration30, *common, '--output', paired)
    official = BASE/'slr83-development-official-reports-v1.json'
    with official.open('x') as f:
        json.dump([json.loads((paired/str(i)/'report.json').read_text()) for i in range(90)], f, indent=2)
        f.write('\n')
    scored = {}
    for version, path, calibration in [('v30', v30, calibration30), ('matched-noise-independent-formal-v1', model, calibration36)]:
        root = BASE/(version+'-slr83-development-v1')
        extra = ['--baseline', paired/'report.json'] if version == 'v30' else []
        run('score_phone_development.py', '--manifest', manifest, '--model', path,
            '--calibration', calibration, '--official-reports', official, '--output', root, *extra)
        scored[version] = root
        run('synthetic_presentation.py', '--scores', root, '--manifest', manifest,
            '--output', BASE/(version+'-slr83-presentation-historical-threshold-v1.json'))
        run('evaluate_frozen_human.py', '--model', path, '--frozen-report', calibration,
            '--features', background, '--output', BASE/(version+'-voxconverse-new-development-v1'),
            '--corpus-name', 'VoxConverse dev2/dev3 separate development stage', '--identity-unit', 'recording')
    selected = BASE/'matched-noise-independent-formal-v1-slr83-new-development-calibration-v1.json'
    run('calibrate_slr83_development.py', '--manifest', manifest,
        '--synthetic', scored['matched-noise-independent-formal-v1'], '--baseline', scored['v30'],
        '--background', BASE/'matched-noise-independent-formal-v1-voxconverse-new-development-v1', '--output', selected)
    if json.loads(selected.read_text())['viable_operating_point']:
        run('synthetic_presentation.py', '--scores', scored['matched-noise-independent-formal-v1'], '--manifest', manifest,
            '--threshold-calibration', selected, '--output', BASE/'matched-noise-independent-formal-v1-slr83-presentation-selected-threshold-v1.json')
    run('replay_voxconverse_official.py', '--manifest', background/'manifest.json',
        *common, '--output', '/tmp/nyx-mn-vox-v2')
    print('Unseen development completed; review evidence, no final-data access or automatic promotion', flush=True)


if __name__ == '__main__':
    main()
