"""Prepare Muse for P4 with the pinned official SDK helpers, then compress it."""
import argparse
import hashlib
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib

MODEL = 'wn10_himuse'
MEMBERS = ('_MODEL_INFO_p1', '_MODEL_INFO_p2', 'wn10_data_p1', 'wn10_data_p2')
PREPARED_TO_SOURCE = {'_MODEL_INFO_': '_MODEL_INFO_p2', 'wn10_data': 'wn10_data_p2'}
SDK_HELPERS = {
    'movemodel.py': 'bbad1ab1c026c5e0211a2c4c662d4ac6a5a043a0abd29ac47f609d785819c785',
    'pack_model.py': 'fe753aa8e64f6442459362bb71d29ec714cb271e4aa476b5b3807864ea318346',
}


def pack(source, output):
    if source.name != MODEL or {p.name for p in source.iterdir() if p.is_file()} != set(MEMBERS):
        raise ValueError('Expected the exact official Muse directory and four-member layout')
    sdk_model = source.resolve().parents[1]
    for name, digest in SDK_HELPERS.items():
        if hashlib.sha256((sdk_model / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Official preparation helper identity mismatch: ' + name)
    # An isolated interpreter prevents our identically named script from shadowing
    # the SDK pack_model imported by movemodel. No SDK/model files are modified.
    code = '''import sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
from movemodel import copy_wn10_model, pack_models
copy_wn10_model(sys.argv[2], sys.argv[3] + '/wn10_himuse', 'esp32p4')
pack_models(sys.argv[3])
'''
    with tempfile.TemporaryDirectory(prefix='muse-p4-model-') as temporary:
        subprocess.run([sys.executable, '-c', code, str(sdk_model),
                        str(source.resolve()), temporary], check=True)
        prepared = Path(temporary) / MODEL
        if {p.name for p in prepared.iterdir()} != set(PREPARED_TO_SOURCE):
            raise ValueError('Official P4 preparation did not produce exactly two runtime members')
        for name, original in PREPARED_TO_SOURCE.items():
            if (prepared / name).read_bytes() != (source / original).read_bytes():
                raise ValueError('Official P4 preparation changed model bytes: ' + name)
        expanded = (Path(temporary) / 'srmodels.bin').read_bytes()
    if not 1024 <= len(expanded) <= 1024 * 1024:
        raise ValueError('Muse model exceeds the explicit standalone expansion bound')
    output.write_bytes(struct.pack('<I', len(expanded)) + zlib.compress(expanded, 9))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    pack(args.source, args.output)
