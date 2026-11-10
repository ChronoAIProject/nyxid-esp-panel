"""Reconstruct validation-only read-speech streams from an audited WAV manifest."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from prepare_hiesp_libri_validation import sha, worker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'features', 'output'):
        p.add_argument('--'+key,type=Path,required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use fresh output')
    cache = json.loads((a.features/'cache.json').read_text())
    audit = json.loads((a.source/'audit.json').read_text())
    if (sha(a.source/'manifest.jsonl') != cache['source_manifest_sha256']
            or audit['manifest_sha256'] != cache['source_manifest_sha256']
            or sha(a.source/'audit.json') != cache['source_audit_sha256']
            or sha(a.features/'human.json') != cache['index_sha256']):
        raise ValueError('Changed source audit/manifest/feature index')
    source_rows = [json.loads(x) for x in (a.source/'manifest.jsonl').read_text().splitlines()]
    groups = defaultdict(list)
    splits = {}
    for row in source_rows:
        speaker = row['speaker']
        if speaker in splits and splits[speaker] != row['split']:
            raise ValueError('Speaker crosses splits')
        splits[speaker] = row['split']
        if row['split'] == 'validation':
            groups[speaker].append(row)
    features = [r for r in json.loads((a.features/'human.json').read_text()) if r['split'] == 'validation']
    if not features or len({r['speaker'] for r in features}) != len(features) or set(groups) != {r['speaker'] for r in features}:
        raise ValueError('Incomplete validation coverage')
    jobs = []
    for row in features:
        sources = sorted(groups[row['speaker']],key=lambda r:r['path'])
        if len(sources) != row['files']:
            raise ValueError('Changed source file count')
        jobs.append((str(a.source),str(a.features),str(a.output),dict(row,sources=sources)))
    a.output.mkdir(parents=True)
    records = []
    with ProcessPoolExecutor(4) as pool:
        for row in pool.map(worker,jobs):
            records.append(row)
            print(json.dumps(dict(completed=len(records),planned=len(jobs))),flush=True)
    result = dict(role='Existing audited WAV read-speech validation PCM for Hi ESP, not final',
                  source_cache_sha256=sha(a.features/'cache.json'),source_audit_sha256=sha(a.source/'audit.json'),
                  source_manifest_sha256=sha(a.source/'manifest.jsonl'),prior_feature_index_sha256=cache['index_sha256'],
                  source_sha256=sha(Path(__file__)),reconstruction_worker_sha256=sha(Path(__file__).with_name('prepare_hiesp_libri_validation.py')),
                  continuous_recordings=False,utterance_concatenation=True,records=records,
                  original_source_hours=sum(r['source_seconds'] for r in records)/3600,
                  independent_final_audio_accessed=False,
                  limits='Previously reused development read speech. Same PCM as Nyx;200ms gaps plus alignment excluded from exposure. Reset between speakers, not utterances. No natural-conversation or independent acceptance claim.')
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__ == '__main__':
    main()
