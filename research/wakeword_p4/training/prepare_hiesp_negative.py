"""Expose only audited existing continuous validation WAVs to Hi ESP replay."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import soundfile as sf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'features', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh output directory')
    audit = json.loads((a.source/'audit.json').read_text())
    cache = json.loads((a.features/'cache.json').read_text())
    if (sha(a.source/'manifest.jsonl') != audit['manifest_sha256'] or
            sha(a.source/'audit.json') != cache['source_audit_sha256'] or
            sha(a.source/'manifest.jsonl') != cache['source_manifest_sha256'] or
            sha(a.features/'human.json') != cache['index_sha256']):
        raise ValueError('Source and feature audit hashes differ')
    # Check metadata only across all splits. Decode validation audio exclusively.
    source_rows = [json.loads(line) for line in (a.source/'manifest.jsonl').read_text().splitlines()]
    feature_rows = {r['speaker']:r for r in json.loads((a.features/'human.json').read_text()) if r['split']=='validation'}
    selected = [r for r in source_rows if r['split']=='validation']
    if not selected or len({r['speaker'] for r in selected}) != len(selected):
        raise ValueError('Require one continuous WAV per validation speaker/session')
    rows = []
    a.output.mkdir(parents=True)
    for r in selected:
        f = feature_rows[r['speaker']]
        path = (a.source/r['path']).resolve()
        if (not path.is_relative_to(a.source.resolve()) or sha(path) != r['sha256'] or
                f['files'] != 1 or abs(f['source_seconds']-r['samples']/16000)>1e-9 or
                sha(a.features/f['path']) != f['feature_sha256']):
            raise ValueError('Continuous source or feature provenance mismatch')
        pcm, rate = sf.read(path,dtype='int16')
        if (rate!=16000 or pcm.ndim!=1 or len(pcm)!=r['samples'] or
                hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()!=r['pcm_sha256']):
            raise ValueError('Decoded source differs from audit')
        target = a.output/path.name; shutil.copyfile(path,target)
        rows.append({**r,'path':target.name,'role':'development-negative-validation',
                     'original_seconds':len(pcm)/16000,'prior_feature_sha256':f['feature_sha256']})
    report = dict(role='Audited continuous development negatives for Hi ESP, not final',
                  source_audit_sha256=sha(a.source/'audit.json'),source_cache_sha256=sha(a.features/'cache.json'),
                  source_manifest_sha256=sha(a.source/'manifest.jsonl'),
                  prior_feature_index_sha256=cache['index_sha256'], source_sha256=sha(Path(__file__)),
                  continuous_recordings=True, records=rows,independent_final_audio_accessed=False,
                  limits='Only prior validation WAVs; no extra gaps, segmentation or normalization. Native original exposure. Not an independent acceptance set.')
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__ == '__main__':
    main()
