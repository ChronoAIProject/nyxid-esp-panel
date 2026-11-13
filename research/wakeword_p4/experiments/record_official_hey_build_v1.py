"""Verify isolated P4 reference build outputs; no compilation or device I/O.

--build-root contains project/, build-hiesp/, build-willow/
and both build logs. --host-check is check_official_hey_reference_v1.py output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


REPO = Path(__file__).resolve().parents[3]
SOURCE_FILES = ['research/wakeword_baseline/' + name for name in (
    'CMakeLists.txt', 'main/CMakeLists.txt', 'main/baseline.c',
    'main/idf_component.yml', 'sdkconfig.defaults', 'dependencies.lock')]
SOURCE_FILES.append('tools/embed_wake_model.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--host-check', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    root = args.build_root
    sources = {name: sha(root / 'project' / name) for name in SOURCE_FILES}
    for name, digest in sources.items():
        assert sha(REPO / name) == digest
    host = json.loads((args.host_check / 'report.json').read_text())
    products = {}
    for selection, engine, symbol, factory in [
            ('hiesp', 'wn9_hiesp', 'baseline_hi_esp', 0),
            ('willow', 'wn9_heywillow_tts', 'baseline_hey_willow', 1)]:
        build = root / ('build-' + selection)
        log = root / ('build-' + selection + '.log')
        lines = log.read_text().splitlines()
        assert 'Project build complete. To flash, run:' in lines
        config = json.loads((build / 'config/sdkconfig.json').read_text())
        expected = dict(IDF_TARGET='esp32p4', IDF_INIT_VERSION='5.5.4',
            ESP32P4_SELECTS_REV_LESS_V3=True, ESP32P4_REV_MIN_100=True,
            ESP_DEFAULT_CPU_FREQ_MHZ=360, SPIRAM=True, SPIRAM_SPEED=200,
            ESPTOOLPY_FLASHSIZE='16MB')
        assert all(config[key] == value for key, value in expected.items())
        assert f'BASELINE_WAKE_MODEL:STRING={engine}' in (build / 'CMakeCache.txt').read_text().splitlines()
        description = json.loads((build / 'project_description.json').read_text())
        compile_entry = next(row for row in json.loads((build / 'compile_commands.json').read_text())
                             if row['file'].endswith('/main/baseline.c'))
        command = shlex.split(compile_entry['command'])
        defines = [part for part in command if part.startswith(('-DBASELINE_', '-DIDF_VER='))]
        for part in [f'-DBASELINE_MODEL_NAME="{engine}"',
                     f'-DBASELINE_USE_MODEL_FACTORY={factory}',
                     f'-DBASELINE_PACKED_START="_binary_{symbol}_bin_start"',
                     f'-DBASELINE_PACKED_END="_binary_{symbol}_bin_end"', '-DIDF_VER="v5.5.4"']:
            assert part in defines
        compiler = Path(command[0])
        version = subprocess.check_output([str(compiler), '--version'], text=True).splitlines()[0]
        idf_version = subprocess.check_output(['git', '-C', description['idf_path'],
                                              'describe', '--always', '--dirty'], text=True).strip()
        assert idf_version == 'v5.5.4'
        binary = build / 'nyx_wake_baseline.bin'
        elf = build / 'nyx_wake_baseline.elf'
        packed = args.host_check / (engine + '.bin')
        embedded = build / 'esp-idf/main' / ('baseline-' + ('hi-esp' if not factory else 'hey-willow') + '.bin')
        assert packed.read_bytes() == embedded.read_bytes()
        assert sha(packed) == host['products'][engine]['compressed_sha256']
        assert binary.read_bytes().count(packed.read_bytes()) == 1
        nm = compiler.with_name('riscv32-esp-elf-nm')
        symbols = subprocess.check_output([str(nm), '--defined-only', str(elf)], text=True)
        needed = [f'_binary_{symbol}_bin_start', f'_binary_{symbol}_bin_end']
        if factory:
            needed.append('esp_wn_handle_from_name')
        assert set(needed) <= {line.split()[-1] for line in symbols.splitlines() if line.strip()}
        products[engine] = dict(build_completed=True, configuration=expected, defines=defines,
            idf_version=idf_version, compiler=version, compiler_sha256=sha(compiler),
            files={path.name: dict(bytes=path.stat().st_size, sha256=sha(path))
                   for path in (binary, elf, embedded)}, log_sha256=sha(log),
            warning_count=sum('warning:' in line.lower() for line in lines),
            warning_lines=[line for line in lines if 'warning:' in line.lower()],
            compiler_warning_count=sum('warning:' in line for line in lines),
            exact_host_checked_model_occurrences_in_app=1, linked_symbols_checked=needed)
    report = dict(schema='official-hey-p4-build-v1', sources=sources, products=products,
        recorder_sha256=sha(Path(__file__)),
        scope='Compilation/linking and embedded byte identity only; not P4 runtime compatibility or quality.',
        inference_performed=False, device_operated=False, audio_io_performed=False,
        candidate_scored=False, reserved_acceptance_accessed=False, accepted=False)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
