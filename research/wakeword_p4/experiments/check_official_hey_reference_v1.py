"""Check reference packing and engine identity without inference or device I/O.

Pass the output of inventory_official_hey_models_v1.py and a fresh --output.
Requires pyserial only to import the guarded runner; no port is opened.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import zlib


REPO = Path(__file__).resolve().parents[3]
LEGACY_COMMIT = '67e4896'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unpack(data):
    count, name, members = struct.unpack_from('<I32sI', data)
    assert count == 1
    end = 40 + 40 * members
    contents = {}
    for i in range(members):
        member, offset, size = struct.unpack_from('<32sII', data, 40 + 40 * i)
        assert offset == end and offset + size <= len(data)
        member = member.rstrip(b'\0').decode('ascii')
        assert member not in contents
        contents[member] = data[offset:offset + size]
        end += size
    assert end == len(data)
    return name.rstrip(b'\0').decode('ascii'), contents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = json.loads((args.inventory / 'inventory.json').read_text())
    packer = load('reference_packer', REPO / 'tools/embed_wake_model.py')
    runner = load('reference_runner', REPO / 'research/wakeword_baseline/run.py')
    comparison = load('historical_comparison', REPO / 'research/wakeword_baseline/compare.py')
    results = {}
    for model_name in packer.MODELS:
        source = args.inventory / 'upstream/model/wakenet_model' / model_name
        output = args.output / (model_name + '.bin')
        for member in source.iterdir():
            relative = str(member.relative_to(args.inventory / 'upstream'))
            assert hashlib.sha256(member.read_bytes()).hexdigest() == inventory['files'][relative]['sha256']
        packer.pack(source, output, model_name)
        compressed = output.read_bytes()
        expanded = zlib.decompress(compressed[4:])
        assert len(expanded) == struct.unpack_from('<I', compressed)[0]
        original = (args.inventory / 'packaged' / model_name / 'srmodels.bin').read_bytes()
        assert hashlib.sha256(original).hexdigest() == inventory['products'][model_name]['packed_sha256']
        assert unpack(expanded) == unpack(original)
        results[model_name] = dict(compressed_bytes=len(compressed),
            compressed_sha256=hashlib.sha256(compressed).hexdigest(),
            expanded_bytes=len(expanded), expanded_sha256=hashlib.sha256(expanded).hexdigest(),
            upstream_model_name_and_all_members_identical=True,
            full_packed_bytes_identical_to_upstream=expanded==original)
        runner.validate_ready(dict(engine=model_name, mode='DET_MODE_95', rate=16000), model_name)
    legacy = args.output / 'legacy-packer.py'
    legacy.write_bytes(subprocess.check_output(['git', 'show', f'{LEGACY_COMMIT}:tools/embed_wake_model.py'], cwd=REPO))
    legacy_bin = args.output / 'legacy-hiesp.bin'
    subprocess.run([sys.executable, str(legacy),
        str(args.inventory / 'upstream/model/wakenet_model/wn9_hiesp'), str(legacy_bin)], check=True)
    default_bin = args.output / 'default-hiesp.bin'
    packer.pack(args.inventory / 'upstream/model/wakenet_model/wn9_hiesp', default_bin)
    assert default_bin.read_bytes() == legacy_bin.read_bytes() == (args.output / 'wn9_hiesp.bin').read_bytes()
    rejections = []
    def rejected(label, action, error):
        try:
            action()
        except error:
            rejections.append(label)
        else:
            raise AssertionError('Not rejected: ' + label)
    comparison.validate_engine(dict(engine='wn9_hiesp'))
    rejected('Historical Hi ESP summary cannot mislabel Willow results',
        lambda: comparison.validate_engine(dict(engine='wn9_heywillow_tts')), ValueError)
    rejected('Historical summary requires engine identity', lambda: comparison.validate_engine({}), ValueError)
    rejected('Willow cannot be silently aliased as default Hi ESP',
        lambda: packer.pack(args.inventory / 'upstream/model/wakenet_model/wn9_heywillow_tts', args.output / 'bad.bin'), ValueError)
    rejected('Unsupported Muse layout cannot enter this WakeNet9 probe',
        lambda: packer.pack(args.inventory / 'upstream/model/wakenet_model/wn10_himuse', args.output / 'bad.bin', 'wn10_himuse'), ValueError)
    incomplete = args.output / 'incomplete/wn9_heywillow_tts'
    shutil.copytree(args.inventory / 'upstream/model/wakenet_model/wn9_heywillow_tts', incomplete)
    (incomplete / 'wn9_index').unlink()
    rejected('Missing model member', lambda: packer.pack(incomplete, args.output / 'bad.bin', incomplete.name), ValueError)
    base = dict(engine='wn9_heywillow_tts', mode='DET_MODE_95', rate=16000)
    for label, change in [('Wrong engine', dict(engine='wn9_hiesp')),
                          ('Wrong mode', dict(mode='DET_MODE_90')),
                          ('Wrong rate', dict(rate=8000)),
                          ('Missing engine', dict(engine=None))]:
        rejected(label, lambda change=change: runner.validate_ready({**base, **change}, base['engine']), RuntimeError)
    report = dict(schema='official-hey-reference-host-check-v1',
        legacy_commit=LEGACY_COMMIT, default_hiesp_compressed_bytes_unchanged=True,
        products=results, explicit_rejections=rejections,
        sources={str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in [Path(__file__), REPO / 'tools/embed_wake_model.py',
                              REPO / 'research/wakeword_baseline/run.py',
                              REPO / 'research/wakeword_baseline/compare.py']},
        inference_performed=False, device_operated=False, audio_io_performed=False,
        candidate_scored=False, reserved_acceptance_accessed=False, accepted=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
