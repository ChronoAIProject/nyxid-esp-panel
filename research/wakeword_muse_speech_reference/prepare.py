"""Authenticate the separate Muse complete-PCM build before fixture execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import struct
import sys
import zlib

PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT.parent / 'wakeword_muse_reference'))
from pack_model import PREPARED_TO_SOURCE, SDK_HELPERS


def sha(data):
    return hashlib.sha256(data).hexdigest()


def collect(root, original, inventory):
    project, build = root / 'project', root / 'build'
    source_files = ['CMakeLists.txt', 'main/CMakeLists.txt', 'main/idf_component.yml',
                    'sdkconfig.defaults', 'dependencies.lock', 'partitions.csv']
    sources = {}
    for name in source_files:
        data = (PROJECT / name).read_bytes()
        assert data == (project / name).read_bytes(), name
        sources['research/wakeword_muse_speech_reference/' + name] = sha(data)
    for name in ('wakeword_baseline/main/baseline.c', 'wakeword_muse_reference/main/layout.cpp',
                 'wakeword_muse_reference/pack_model.py'):
        data = (REPO / 'research' / name).read_bytes()
        assert data == (root / name).read_bytes(), name
        sources['research/' + name] = sha(data)
    for name in ('README.md', 'prepare.py'):
        sources['research/wakeword_muse_speech_reference/' + name] = sha((PROJECT / name).read_bytes())
    assert (PROJECT / 'dependencies.lock').read_bytes() == (PROJECT.parent / 'wakeword_muse_reference/dependencies.lock').read_bytes()
    sr = project / 'managed_components/espressif__esp-sr'
    for name, entry in inventory['files'].items():
        assert sha((sr / name).read_bytes()) == entry['installed_sha256'], name
    for name, digest in SDK_HELPERS.items():
        assert sha((sr / 'model' / name).read_bytes()) == digest
    dl = project / 'managed_components/espressif__esp-dl'
    assert sha((dl / 'dl/model/include/dl_model_base.hpp').read_bytes()) == original['dl_model_header_sha256']
    packed = (build / 'esp-idf/main/muse-model.bin').read_bytes()
    assert sha(packed) == original['compressed_model_sha256']
    expanded = zlib.decompress(packed[4:])
    assert len(expanded) == struct.unpack_from('<I', packed)[0] == original['expanded_model_bytes'] == 430762
    assert sha(expanded) == original['expanded_model_sha256']
    number, name, count = struct.unpack_from('<I32sI', expanded)
    assert number == 1 and name.rstrip(b'\0') == b'wn10_himuse' and count == 2
    cursor, members = 120, {}
    for i in range(count):
        name, offset, size = struct.unpack_from('<32sII', expanded, 40 + i * 40)
        name = name.rstrip(b'\0').decode('ascii')
        assert name in PREPARED_TO_SOURCE and name not in members and offset == cursor
        data = expanded[offset:offset + size]
        assert len(data) == size and data == (sr / 'model/wakenet_model/wn10_himuse' / PREPARED_TO_SOURCE[name]).read_bytes()
        members[name] = dict(bytes=size, sha256=sha(data), source_member=PREPARED_TO_SOURCE[name])
        cursor += size
    assert cursor == len(expanded) and members == original['model_members']
    config = json.loads((build / 'config/sdkconfig.json').read_text())
    assert all(config[key] == value for key, value in original['configuration'].items())
    commands = json.loads((build / 'compile_commands.json').read_text())
    baseline = next(row for row in commands if row['file'].endswith('/main/baseline.c'))
    layout = next(row for row in commands if row['file'].endswith('/main/layout.cpp'))
    for row, name in ((baseline, 'wakeword_baseline/main/baseline.c'), (layout, 'wakeword_muse_reference/main/layout.cpp')):
        assert Path(row['file']).read_bytes() == (REPO / 'research' / name).read_bytes()
    flags = shlex.split(baseline['command'])
    required = ('BASELINE_MODEL_NAME="wn10_himuse"', 'BASELINE_USE_MODEL_FACTORY=1',
                'BASELINE_THRESHOLD_CONTROL=1', 'BASELINE_REPORT_RESOURCES=1', 'BASELINE_LONG_STREAMS=0')
    assert all('-D' + definition in flags for definition in required)
    compiler = Path(flags[0])
    assert sha(compiler.read_bytes()) == original['compiler_sha256']
    elf = build / 'nyx_muse_speech_reference.elf'
    app = (build / 'nyx_muse_speech_reference.bin').read_bytes()
    assert app[0] == 0xe9 and struct.unpack_from('<I', app, 32)[0] == 0xabcd5432
    assert app[176:208].hex() == sha(elf.read_bytes())
    assert app.count(packed) == 1
    symbols = subprocess.check_output([str(compiler.with_name('riscv32-esp-elf-nm')), '--defined-only', str(elf)], text=True)
    assert set(original['linked_symbols']) <= {line.split()[-1] for line in symbols.splitlines() if line.strip()}
    log = (root / 'build-private.log').read_text()
    assert 'Project build complete. To flash, run:' in log and 'warning:' not in log.lower()
    return dict(schema='official-muse-consumed-speech-build-v1', sources=sources,
        model_members=members, compressed_model_sha256=sha(packed), expanded_model_sha256=sha(expanded),
        expanded_model_bytes=len(expanded), firmware_sha256=sha(app), firmware_bytes=len(app), elf_sha256=sha(elf.read_bytes()),
        preparation_target='esp32p4', dl_model_abi_bytes=144, configuration=original['configuration'],
        compiler_sha256=sha(compiler.read_bytes()), compile_definitions=list(required), compiler_warnings=0,
        same_model_and_abi_as_original_zero_probe=True, original_baseline_source_unchanged=True,
        device_operated=False, fresh_development_accessed=False, acceptance_accessed=False, candidate_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build-root', 'zero-build-report', 'inventory', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.build_root, json.loads(args.zero_build_report.read_text()), json.loads(args.inventory.read_text()))
    report['original_zero_build_report_sha256'] = sha(args.zero_build_report.read_bytes())
    report['inventory_sha256'] = sha(args.inventory.read_bytes())
    args.output.mkdir(exist_ok=False)
    (args.output / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(firmware_bytes=report['firmware_bytes'], firmware_sha256=report['firmware_sha256'],
                         same_original_model=True, compiled=True, native_inference=False)))


if __name__ == '__main__':
    main()
