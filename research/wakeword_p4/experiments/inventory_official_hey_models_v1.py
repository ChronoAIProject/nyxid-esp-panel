"""Authenticate installed reference artifacts against immutable ESP-SR sources.

Pack each exact model with the original upstream tool and independently unpack
every member. No inference, speech generation, audio I/O or device operation.
Requires PyYAML. Run from the repository root with a fresh --output directory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import urllib.request

import yaml


COMMIT = 'b9dc43a329a68de2f72147808fc9ea29d2ace71e'
COMPONENT_HASH = '7ba1e97d446440faa96a7d98fbda7dbaaf7ae6b960e06cc68264d38aa6eb4d14'
MODELS = {
    'wn9_hiesp': ['_MODEL_INFO_', 'wn9_index', 'wn9_data'],
    'wn9_heywillow_tts': ['_MODEL_INFO_', 'wn9_index', 'wn9_data'],
    'wn10_himuse': ['_MODEL_INFO_p1', '_MODEL_INFO_p2', 'wn10_data_p1', 'wn10_data_p2'],
}
FILES = ['idf_component.yml', 'Kconfig.projbuild', 'CMakeLists.txt', 'model/pack_model.py',
         'src/model_path.c', 'src/include/model_path.h',
         'include/esp32p4_less_v3/esp_wn_iface.h', 'include/esp32p4_less_v3/esp_wn_models.h',
         'lib/esp32p4_less_v3/libwakenet.a'] + [
             f'model/wakenet_model/{name}/{member}' for name, members in MODELS.items() for member in members]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unpack(data):
    assert struct.unpack_from('<I', data)[0] == 1
    name, count = struct.unpack_from('<32sI', data, 4)
    end = 40 + 40 * count
    members = {}
    for index in range(count):
        key, offset, size = struct.unpack_from('<32sII', data, 40 + 40 * index)
        assert offset == end and offset + size <= len(data)
        key = key.rstrip(b'\0').decode('ascii')
        assert key not in members
        members[key] = data[offset:offset + size]
        end += size
    assert end == len(data)
    return name.rstrip(b'\0').decode('ascii'), members


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', type=Path, default=Path('managed_components/espressif__esp-sr'))
    parser.add_argument('--lock', type=Path, default=Path('dependencies.lock'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and sys.byteorder == 'little'
    locked = yaml.safe_load(args.lock.read_bytes())['dependencies']['espressif/esp-sr']
    assert locked['version'] == '2.6.1' and locked['component_hash'] == COMPONENT_HASH
    assert (args.component / '.component_hash').read_text().strip() == COMPONENT_HASH
    installed = yaml.safe_load((args.component / 'idf_component.yml').read_bytes())
    assert installed['repository_info']['commit_sha'] == COMMIT
    args.output.mkdir(parents=True)

    def receive(path):
        url = f'https://raw.githubusercontent.com/espressif/esp-sr/{COMMIT}/{path}'
        raw = urllib.request.urlopen(url, timeout=30).read()
        local = (args.component / path).read_bytes()
        if path == 'idf_component.yml':
            normalized = {key: value for key, value in installed.items()
                          if key not in ('repository', 'repository_info')}
            assert yaml.safe_load(raw) == normalized
            match = 'YAML fields identical after excluding registry repository annotations'
        else:
            assert raw == local, path
            match = 'byte-identical'
        target = args.output / 'upstream' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return path, dict(url=url, bytes=len(raw), sha256=sha(raw),
                          installed_bytes=len(local), installed_sha256=sha(local), match=match)

    with ThreadPoolExecutor(max_workers=6) as pool:
        received = dict(pool.map(receive, FILES))
    products = {}
    packer = args.output / 'upstream/model/pack_model.py'
    for name, expected in MODELS.items():
        folder = args.output / 'packaged' / name
        model = folder / name
        model.mkdir(parents=True)
        for member in sorted(expected):
            shutil.copyfile(args.output / f'upstream/model/wakenet_model/{name}/{member}', model / member)
        subprocess.run([sys.executable, str(packer), '-m', str(folder)], check=True, capture_output=True)
        packed = (folder / 'srmodels.bin').read_bytes()
        actual_name, members = unpack(packed)
        assert actual_name == name and set(members) == set(expected)
        assert all(blob == (model / member).read_bytes() for member, blob in members.items())
        products[name] = dict(raw_member_bytes=sum(map(len, members.values())),
            packed_bytes=len(packed), packed_sha256=sha(packed),
            packed_member_order=list(members), packed_sha256_is_canonical=False,
            all_packed_member_bytes_authenticated=True,
            model_info={member: members[member].decode().strip() for member in members if member.startswith('_MODEL_INFO')})
    config = (args.component / 'Kconfig.projbuild').read_text()
    assert re.search(r'config SR_WN_WN9_HEYWILLOW_TTS\s+bool "Hey,Willow', config)
    assert re.search(r'config SR_WN_WN10_HIMUSE\s+bool "Hi/Hey Muse', config)
    assert 'IDF_TARGET_ESP32P4' in config
    report = dict(schema='official-hey-reference-artifact-inventory-v1',
        source_sha256=sha(Path(__file__).read_bytes()), esp_sr='2.6.1', upstream_commit=COMMIT,
        dependency_lock_sha256=sha(args.lock.read_bytes()), registry_component_hash=COMPONENT_HASH,
        authenticated_source_files=len(received), files=received, products=products,
        p4_configuration_advertises_both_models=True, model_inference=False,
        sdk_upgrade=False, candidate_scored=False, reserved_acceptance_accessed=False,
        device_operated=False, audio_io_performed=False, accepted=False,
        limits=[
            'The registry component hash is a local installation identity check; only the listed files were independently compared with immutable upstream bytes.',
            'The installed idf_component.yml is normalized/reannotated by the registry; all remaining YAML fields match upstream exactly.',
            'Hey Willow directory is wn9_heywillow_tts; its metadata contains wakenet9l. This label alone does not establish the actual interface, model topology or resource cost. Preserve the official directory name for runtime selection.',
            'Muse contains two metadata/data pairs. Raw metadata suffix numbers are not measured threshold getters or calibrated operating points.',
            'Packing/unpacking authenticates transport and names, not successful create/detect calls, actual P4 sampling cadence, runtime memory, recognition or latency.',
            'The original packer uses filesystem enumeration order; each packed hash identifies this concrete inventory output, not a unique canonical encoding. A sorted encoding must match names and all member bytes, not necessarily the entire packed hash.',
            'Only the P4 revision<3 IDF5 WakeNet library/headers are inventoried; other chip/library variants are not covered.',
            'No current training, original report, model weights, production configuration or audio pipeline was changed.'
        ])
    (args.output / 'inventory.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('esp_sr', 'upstream_commit', 'authenticated_source_files', 'products')}, indent=2))


if __name__ == '__main__':
    main()
