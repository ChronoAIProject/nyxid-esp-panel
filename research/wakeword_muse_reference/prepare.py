"""Authenticate the Muse build and freeze its zero-input protocol before use."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import struct
import subprocess
import zlib
import yaml

from pack_model import PREPARED_TO_SOURCE, SDK_HELPERS

PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]
SOURCES = ('CMakeLists.txt', 'main/CMakeLists.txt', 'main/idf_component.yml',
           'main/muse.c', 'main/layout.cpp', 'sdkconfig.defaults', 'dependencies.lock', 'partitions.csv', 'pack_model.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    copied = args.build_root / 'project'
    build = args.build_root / 'build'
    source_hashes = {}
    for name in SOURCES:
        assert (copied / name).read_bytes() == (PROJECT / name).read_bytes(), name
        source_hashes['research/wakeword_muse_reference/' + name] = sha(PROJECT / name)
    for name in ('run.py', 'prepare.py', 'audit.py', 'README.md'):
        source_hashes['research/wakeword_muse_reference/' + name] = sha(PROJECT / name)
    inventory = json.loads(args.inventory.read_text())
    sr = copied / 'managed_components/espressif__esp-sr'
    for name, entry in inventory['files'].items():
        assert sha(sr / name) == entry['installed_sha256'], name
    helpers = {}
    for name, digest in SDK_HELPERS.items():
        assert sha(sr / 'model' / name) == digest, name
        helpers[name] = dict(sha256=digest,
            url=f"https://raw.githubusercontent.com/espressif/esp-sr/{inventory['upstream_commit']}/model/{name}")
    lock = yaml.safe_load((copied / 'dependencies.lock').read_text())
    components = {name: dict(version=entry['version'], component_hash=entry['component_hash'])
                  for name, entry in lock['dependencies'].items() if name != 'idf'}
    assert components['espressif/esp-dl']['version'] == '3.3.14'
    dl = copied / 'managed_components/espressif__esp-dl'
    dl_manifest = yaml.safe_load((dl / 'idf_component.yml').read_text())
    assert dl_manifest['version'] == '3.3.14'
    layout_entry = next(e for e in json.loads((build / 'compile_commands.json').read_text())
                        if e['file'].endswith('/main/layout.cpp'))
    assert Path(layout_entry['file']).read_bytes() == (PROJECT / 'main/layout.cpp').read_bytes()

    embedded = build / 'esp-idf/main/muse-model.bin'
    compressed = embedded.read_bytes()
    size = struct.unpack_from('<I', compressed)[0]
    expanded = zlib.decompress(compressed[4:])
    assert len(expanded) == size and 1024 <= size <= 1024 * 1024
    number, name, count = struct.unpack_from('<I32sI', expanded)
    assert number == 1 and name.rstrip(b'\0') == b'wn10_himuse' and count == 2
    cursor = 40 + count * 40
    members = {}
    for i in range(count):
        name, offset, length = struct.unpack_from('<32sII', expanded, 40 + i * 40)
        name = name.rstrip(b'\0').decode('ascii')
        assert name in PREPARED_TO_SOURCE and name not in members and offset == cursor
        data = expanded[offset:offset + length]
        original = PREPARED_TO_SOURCE[name]
        assert len(data) == length and data == (sr / 'model/wakenet_model/wn10_himuse' / original).read_bytes()
        members[name] = dict(bytes=length, sha256=hashlib.sha256(data).hexdigest(), source_member=original)
        cursor += length
    assert cursor == len(expanded)
    assert set(members) == set(PREPARED_TO_SOURCE)
    config = json.loads((build / 'config/sdkconfig.json').read_text())
    required = dict(IDF_TARGET='esp32p4', IDF_INIT_VERSION='5.5.4',
        ESP32P4_SELECTS_REV_LESS_V3=True, ESP32P4_REV_MIN_100=True,
        ESP_DEFAULT_CPU_FREQ_MHZ=360, SPIRAM=True, SPIRAM_SPEED=200,
        ESPTOOLPY_FLASHSIZE='16MB', PARTITION_TABLE_CUSTOM=True)
    assert all(config[k] == v for k, v in required.items())
    entry = next(e for e in json.loads((build / 'compile_commands.json').read_text())
                 if e['file'].endswith('/main/muse.c'))
    compiler = Path(shlex.split(entry['command'])[0])
    version = subprocess.check_output([str(compiler), '--version'], text=True).splitlines()[0]
    symbols = subprocess.check_output([str(compiler.with_name('riscv32-esp-elf-nm')),
                                      '--defined-only', str(build / 'nyx_muse_reference.elf')], text=True)
    expected_symbols = {'esp_wn_handle_from_name', '_binary_muse_model_bin_start', '_binary_muse_model_bin_end'}
    assert expected_symbols <= {line.split()[-1] for line in symbols.splitlines() if line.strip()}
    log = (args.build_root / 'build-private.log').read_text()
    assert 'Project build complete. To flash, run:' in log and 'warning:' not in log.lower()
    app = build / 'nyx_muse_reference.bin'
    assert app.read_bytes().count(compressed) == 1
    report = dict(schema='official-muse-zero-build-v3', sources=source_hashes,
        inventory_sha256=sha(args.inventory), esp_sr='2.6.1',
        upstream_commit=inventory['upstream_commit'], preparation_helpers=helpers,
        preparation_target='esp32p4', preparation_api='copy_wn10_model then pack_models',
        raw_inventory_members=4, model_members=members,
        components=components, dl_model_abi_bytes=144,
        esp_dl_upstream_commit=dl_manifest['repository_info']['commit_sha'],
        dl_model_header_sha256=sha(dl / 'dl/model/include/dl_model_base.hpp'),
        dl_model_abi_compile_assertion=True,
        expanded_model_bytes=size, expanded_model_sha256=hashlib.sha256(expanded).hexdigest(),
        compressed_model_bytes=len(compressed), compressed_model_sha256=sha(embedded),
        firmware_bytes=app.stat().st_size, firmware_sha256=sha(app),
        elf_sha256=sha(build / 'nyx_muse_reference.elf'), configuration=required,
        compiler=version, compiler_sha256=sha(compiler), linked_symbols=sorted(expected_symbols),
        exact_compressed_model_occurrences_in_app=1, compiler_warnings=0,
        device_operated=False, audio_io_performed=False, recognition_accepted=False)
    report_path = args.output / 'build.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    plan = dict(schema='official-muse-zero-plan-v3', sources=source_hashes,
        build_report_sha256=sha(report_path), firmware_sha256=report['firmware_sha256'],
        engine='wn10_himuse', mode='DET_MODE_95', setter_called=False,
        preparation_target='esp32p4', expanded_model_bytes=size,
        sample_rate=16000, channels=1, chunk='actual get_samp_chunksize; require 2..4096',
        cases=['1', 'chunk-1', 'chunk+1', '16000', '16001'], prime_calls_per_case=63,
        lifecycle='Per case create, query, allocate zeros, 63 zero prime calls, whole input with final-only padding, free/destroy. Model bytes loaded once; no mid-stream reset.',
        observations='All raw prime/input returns, every word name/default/postprime getter bits, CRC of original unpadded zero samples, actual calls, timings and phase heap/probe buffer bytes.',
        restore='Only verified development MAC/active slot. Exact app and restore binary SHA; fresh partition-table read. Guarded finally restores app/slot/service/agent/listener, then independent HTTP/no-owner check. app-only is not voice acceptance.',
        decision='Preserve initialization/failure and every return. Report positive zero events rather than reinterpret as recognition. No threshold or candidate selection.',
        scope='Initialization, actual interface and zero-input resources only; no speaker, microphone, TTS, speech/FAR, parity or reserved acceptance.')
    (args.output / 'protocol.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(firmware_bytes=report['firmware_bytes'], firmware_sha256=report['firmware_sha256'],
                         expanded_model_bytes=size, compiled=True, native_inference=False)))


if __name__ == '__main__':
    main()
