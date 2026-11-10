"""Fetch pinned public measured responses for local training-only research.

The input audit is the GPU branch's byte-exact public source manifest. Raw
responses remain in the requested local directory; no archive is published.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.parse
import urllib.request

AUDIT_SHA256 = '4c3d8370240df9dda319d180c91f9b0c0ad0740731844e2ed8ff2b299fe69985'
REPO = 'davidscripka/MIT_environmental_impulse_responses'
REVISION = 'b824a1ef2821f112fda0b9cb26e4278c62b425bb'


def main():
    import numpy as np
    import soundfile as sf

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    raw = args.audit.read_bytes()
    if hashlib.sha256(raw).hexdigest() != AUDIT_SHA256 or not 1 <= args.workers <= 16:
        p.error('Exact pinned audit and one to16 workers required')
    audit = json.loads(raw)
    if audit['repo'] != REPO or audit['revision'] != REVISION or len(audit['records']) != 270:
        raise ValueError('Unexpected pinned response population')
    for row in audit['records']:
        path = PurePosixPath(row['path'])
        if path.is_absolute() or '..' in path.parts or path.parts[0] != '16khz' or path.suffix != '.wav':
            raise ValueError('Unexpected public response path')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'audit.json').write_bytes(raw)
    base = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/'

    def download(row):
        request = urllib.request.Request(base + urllib.parse.quote(row['path']),
                                         headers={'User-Agent': 'nyx-wake-research'})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Pinned response WAV differs')
        path = args.output / row['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        pcm, rate = sf.read(path, dtype='float32')
        if (rate != 16000 or pcm.ndim != 1 or len(pcm) != row['samples']
                or not np.isfinite(pcm).all() or not np.any(pcm)):
            raise ValueError('Invalid measured mono16kHz response')
        return len(data)

    total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for number, size in enumerate(pool.map(download, audit['records']), 1):
            total += size
            if number % 30 == 0 or number == len(audit['records']):
                print(json.dumps({'verified_rirs': number, 'total': len(audit['records'])}), flush=True)
    report = {'repo': REPO, 'revision': REVISION, 'audit_sha256': AUDIT_SHA256,
              'files': len(audit['records']), 'bytes': total, 'wav_hashes_headers_verified': True,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source': 'https://mcdermottlab.mit.edu/Reverb/IR_Survey.html',
              'attribution': 'MIT Computational Audition Lab environmental impulse responses;16kHz mirror.',
              'scope': 'Local training-only research download; no response redistribution or model accuracy claim.'}
    (args.output / 'download-complete.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
