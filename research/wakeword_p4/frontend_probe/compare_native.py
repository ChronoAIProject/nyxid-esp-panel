"""Bind rebuilt native frontends to the fixed30 saved training witnesses.

No model or audio device is used. Preserve all differences; never overwrite the
received pack or install/rebuild the Python extension in its environment.
"""
import argparse
import ctypes
import hashlib
from importlib.metadata import distribution, version
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import soundfile as sf

TRAINING = Path(__file__).resolve().parents[1]/'training'
sys.path.insert(0, str(TRAINING))
from audit_old_complete_sources import member, reference_stream
from features import features
from prepare_old_complete_sources import source_path

UNITS = ('kiss_fft_int16', 'fft', 'fft_util', 'filterbank', 'filterbank_util',
         'frontend', 'frontend_util', 'log_lut', 'log_scale', 'log_scale_util',
         'noise_reduction', 'noise_reduction_util', 'pcan_gain_control',
         'pcan_gain_control_util', 'window', 'window_util')
PINNED = {
    'sources.json': '0f2b008c2525ee9c093580008af1029618823f92d2718622acd513e682cbc981',
    'records.json': '757f1e6df724943c0e0bf68b9fe7de9b3402cbcebb9cf6c87a89e876ee9af090',
    'report.json': '2c2c49dd90a6c3547a78b190de420619ea25d79344d9efd9b8a69c7bc6dc0751',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compare(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype == np.uint16
    delta = actual.astype(np.int32)-expected.astype(np.int32)
    where = np.argwhere(delta)
    return {'exact': not len(where), 'different_elements': len(where),
            'max_abs_delta': int(np.abs(delta).max()),
            'actual_sha256': sha(actual.astype('<u2').tobytes()),
            'expected_sha256': sha(expected.astype('<u2').tobytes()),
            'differences': [{'frame': int(i), 'channel': int(j),
                             'actual': int(actual[i, j]), 'expected': int(expected[i, j])}
                            for i, j in where]}


class Native:
    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path))
        self.lib.nyx_frontend_create.argtypes = []
        self.lib.nyx_frontend_create.restype = ctypes.c_void_p
        self.lib.nyx_frontend_destroy.argtypes = [ctypes.c_void_p]
        self.lib.nyx_frontend_destroy.restype = None
        self.lib.nyx_frontend_process.argtypes = [ctypes.c_void_p,
            np.ctypeslib.ndpointer(np.int16, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(np.uint16, ndim=1, flags='C_CONTIGUOUS')]
        self.lib.nyx_frontend_process.restype = ctypes.c_int
        self.lib.nyx_frontend_weight.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.nyx_frontend_weight.restype = ctypes.c_int
        self.lib.nyx_frontend_set_weight_pair.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                        ctypes.c_int, ctypes.c_int]
        self.lib.nyx_frontend_set_weight_pair.restype = ctypes.c_int

    def process(self, pcm, weight_pairs=()):
        pcm = np.pad(pcm, (0, (-len(pcm)) % 160)).astype(np.int16)
        state = self.lib.nyx_frontend_create()
        if not state:
            raise RuntimeError('Native frontend allocation failed')
        rows = []
        try:
            for index, weight, unweight in weight_pairs:
                if self.lib.nyx_frontend_set_weight_pair(state, index, weight, unweight) != 1:
                    raise ValueError('Invalid diagnostic weight pair')
                assert [self.lib.nyx_frontend_weight(state, index, flag) for flag in (0, 1)] == [weight, unweight]
            coefficient = [self.lib.nyx_frontend_weight(state, 293, flag) for flag in (0, 1)]
            for at in range(0, len(pcm), 160):
                out = np.empty(40, dtype=np.uint16)
                n = self.lib.nyx_frontend_process(state, pcm[at:at+160], out)
                if n not in (0, 40):
                    raise RuntimeError('Native frontend did not consume a complete hop')
                if n:
                    rows.append(out)
        finally:
            self.lib.nyx_frontend_destroy(state)
        return np.asarray(rows, dtype=np.uint16).reshape(-1, 40), coefficient


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'study', 'source-root', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--compiler', default='c++')
    p.add_argument('--weight-pair', type=int, nargs=3, action='append', default=[],
                   metavar=('INDEX', 'WEIGHT', 'UNWEIGHT'),
                   help='Separate diagnostic arm: override only these initialized pairs')
    args = p.parse_args()
    assert version('pymicro-features') == '2.0.2'
    for name, digest in PINNED.items():
        assert sha((args.study/name).read_bytes()) == digest, name
    assert sha((TRAINING/'features.py').read_bytes()) == '63fb8e1691781817fd31027fdedfefbe15e0f8e1c104d3d1817c63725f9fa9e1'
    sources = json.loads((args.study/'sources.json').read_bytes())
    records = json.loads((args.study/'records.json').read_bytes())
    first = {}
    for index, group in enumerate(sources):
        row = group['aliases'][0]
        first.setdefault((row['engine'], row['label']), index)
    assert len(first) == 10
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve()
    lib = source/'tensorflow/lite/experimental/microfrontend/lib'
    own = Path(__file__).with_name('native_frontend.cpp').resolve()
    units = [own, *(lib/(name+'.cc') for name in UNITS),
             source/'kissfft/kiss_fft.cc', source/'kissfft/tools/kiss_fftr.cc']
    manifest = {str(path.relative_to(source)): sha(path.read_bytes())
                for path in sorted({*units[1:], *lib.glob('*.h'),
                                    *(source/'kissfft').rglob('*.h')})}
    libraries, builds = {}, {}
    for mode in ('default', 'off'):
        binary = args.output.resolve()/('frontend-'+mode+'.so')
        cmd = [args.compiler, '-std=c++17', '-O2', '-shared', '-fPIC',
               '-DFIXED_POINT=16', '-fno-fast-math']
        if mode != 'default':
            cmd.append('-ffp-contract='+mode)
        cmd += ['-I', str(source), '-I', str(source/'kissfft'), *map(str, units), '-o', str(binary)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        (args.output/('build-'+mode+'.log')).write_text(result.stdout+result.stderr)
        result.check_returncode()
        builds[mode] = {'command': cmd, 'binary_sha256': sha(binary.read_bytes())}
        libraries[mode] = Native(binary)
    details, arrays = [], {}
    for index in first.values():
        group = sources[index]
        row = group['aliases'][0]
        wav = source_path(args.source_root.resolve(), row['path'])
        assert sha(wav.read_bytes()) == group['source_sha256']
        pcm, rate = sf.read(wav, dtype='float64')
        assert rate == 16000 and pcm.shape == (row['samples'],)
        for variant in (0, 1, 2):
            pack_index = index*3+variant
            record = records[pack_index]
            rendered = reference_stream(pcm, row['path'], variant)
            assert sha(rendered.astype('<i2').tobytes()) == record['rendered_pcm_sha256']
            saved = np.load(member(args.study.resolve(), record['raw']), allow_pickle=False)
            assert sha(saved.astype('<u2').tobytes()) == record['raw_sha256']
            wheel = features(rendered)
            item = {'pack_index': pack_index, 'engine': row['engine'], 'label': row['label'],
                    'variant': variant, 'rendered_pcm_sha256': record['rendered_pcm_sha256'],
                    'shape': list(saved.shape), 'wheel_vs_saved': compare(wheel, saved)}
            arrays['wheel_'+str(pack_index)] = wheel
            for mode, native in libraries.items():
                actual, coefficient = native.process(rendered)
                arrays[mode+'_'+str(pack_index)] = actual
                item[mode+'_vs_saved'] = compare(actual, saved)
                item[mode+'_coefficient293'] = coefficient
            if args.weight_pair:
                actual, _ = libraries['default'].process(rendered, args.weight_pair)
                arrays['override_'+str(pack_index)] = actual
                item['override_vs_saved'] = compare(actual, saved)
            details.append(item)
    assert len(details) == 30
    np.savez_compressed(args.output/'arrays.npz', **arrays)
    package = distribution('pymicro-features')
    binary = next(path for path in package.files if str(path).endswith('.so'))
    report = {'scope': __doc__, 'pinned_inputs': PINNED, 'builds': builds,
              'compiler_version': subprocess.check_output([args.compiler, '--version'], text=True),
              'platform': platform.platform(), 'python': sys.version,
              'versions': {key: version(key) for key in ('numpy', 'scipy', 'soundfile', 'pymicro-features')},
              'upstream_source_sha256': manifest, 'wrapper_sha256': sha(own.read_bytes()),
              'runner_sha256': sha(Path(__file__).read_bytes()),
              'rendering_source_sha256': sha((TRAINING/'audit_old_complete_sources.py').read_bytes()),
              'wheel_binary_sha256': sha(package.locate_file(binary).read_bytes()),
              'witnesses': len(details), 'details': details, 'diagnostic_weight_pairs': args.weight_pair,
              'arrays_npz_sha256': sha((args.output/'arrays.npz').read_bytes()),
              'audio_io': False, 'board_access': False, 'trained_or_scored_model': False,
              'limits': 'Fixed training witnesses only; matching outputs do not prove all internal tables or other inputs equivalent. No Linux/P4/recognition parity claim.'}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    modes = ['wheel_vs_saved', 'default_vs_saved', 'off_vs_saved']
    if args.weight_pair:
        modes.append('override_vs_saved')
    print(json.dumps({'witnesses': len(details),
        'exact': {key: sum(row[key]['exact'] for row in details)
                  for key in modes}}))


if __name__ == '__main__':
    main()
