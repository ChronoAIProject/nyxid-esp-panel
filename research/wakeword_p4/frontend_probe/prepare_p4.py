"""Build a numeric-only P4 project from the fixed30 old training witnesses."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_native import PINNED, TRAINING
from audit_old_complete_sources import member, reference_stream
from prepare_old_complete_sources import source_path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'source-root', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    for name, digest in PINNED.items():
        assert sha((args.study/name).read_bytes()) == digest
    groups = json.loads((args.study/'sources.json').read_bytes())
    records = json.loads((args.study/'records.json').read_bytes())
    first = {}
    for index, group in enumerate(groups):
        row = group['aliases'][0]
        first.setdefault((row['engine'], row['label']), index)
    assert len(first) == 10
    template = Path(__file__).with_name('p4')
    args.output.mkdir(parents=True, exist_ok=False)
    project = args.output/'project'
    shutil.copytree(template, project)
    data, witnesses = bytearray(b'NYXFE001'+struct.pack('<I', 30)), []
    for index in first.values():
        group = groups[index]; row = group['aliases'][0]
        path = source_path(args.source_root.resolve(), row['path'])
        assert sha(path.read_bytes()) == group['source_sha256']
        pcm, rate = sf.read(path, dtype='float64')
        assert rate == 16000 and pcm.shape == (row['samples'],)
        for variant in (0, 1, 2):
            pack_index = index*3+variant; record = records[pack_index]
            rendered = reference_stream(pcm, row['path'], variant)
            assert sha(rendered.astype('<i2').tobytes()) == record['rendered_pcm_sha256']
            padded = np.pad(rendered, (0, (-len(rendered)) % 160)).astype('<i2')
            raw = np.load(member(args.study.resolve(), record['raw']), allow_pickle=False)
            assert raw.dtype == np.uint16 and raw.shape == (len(padded)//160-2, 40)
            assert sha(raw.astype('<u2').tobytes()) == record['raw_sha256']
            data += struct.pack('<III', pack_index, len(padded), len(raw))
            data += padded.tobytes()+raw.astype('<u2').tobytes()
            witnesses.append({'pack_index': pack_index, 'samples': len(padded), 'frames': len(raw),
                'rendered_pcm_sha256': record['rendered_pcm_sha256'],
                'padded_pcm_sha256': sha(padded.tobytes()), 'mac_raw_sha256': record['raw_sha256']})
    assert len(witnesses) == 30
    (project/'vectors.bin').write_bytes(data)
    report = {'scope': __doc__, 'witnesses': witnesses, 'pinned_inputs': PINNED,
              'vectors_bytes': len(data), 'vectors_sha256': sha(data),
              'preparer_sha256': sha(Path(__file__).read_bytes()),
              'renderer_sha256': sha((TRAINING/'audit_old_complete_sources.py').read_bytes()),
              'template_sha256': {str(x.relative_to(template)): sha(x.read_bytes())
                                  for x in sorted(template.rglob('*')) if x.is_file()},
              'audio_io': False, 'model_inference': False, 'board_access_at_preparation': False,
              'limits': 'Numeric fixture only. No detector/model or recognition result. Actual P4 coefficients/output must be recorded without overriding its tables.'}
    (args.output/'vectors-manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'witnesses': 30, 'vectors_bytes': len(data), 'vectors_sha256': report['vectors_sha256']}))


if __name__ == '__main__':
    main()
