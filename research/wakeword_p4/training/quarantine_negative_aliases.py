"""Conservatively quarantine training negatives transcribed as a target alias.

This is a data-quality screen, not proof of pronunciation or wake recognition.
Other ASR mismatches remain in the corpus; original files and labels stay intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'details', 'report', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh output directory')
    rows = [json.loads(l) for l in (a.source/'manifest.jsonl').read_text().splitlines()]
    details = [json.loads(l) for l in a.details.read_text().splitlines()]
    audit = json.loads(a.report.read_text())
    indexed = {r['path']: r for r in details}
    if (not audit['training_source'] or audit['manifest_sha256'] != sha(a.source/'manifest.jsonl')
            or len(indexed) != len(details) or set(indexed) != {r['path'] for r in rows}):
        raise ValueError('Audit does not cover this source')
    kept, excluded = [], []
    for row in rows:
        detail = indexed[row['path']]
        path = (a.source/row['path']).resolve()
        if row['split'] != 'training' or row['label'] != 0 or not path.is_relative_to(a.source.resolve()):
            raise ValueError('Requires training-only negatives within source')
        if sha(path) != row['sha256'] or detail['sha256'] != row['sha256'] or detail['expected'] != row['text']:
            raise ValueError('Changed audio or audit text')
        words = ' '.join(re.findall('[a-z]+', detail['transcript'].lower()))
        if re.search(r'\bhey (?:nicks|nix|nyx|knicks|nick s)\b', words):
            excluded.append(dict(path=row['path'], sha256=row['sha256'], transcript=detail['transcript'],
                                 reason='Target alias in independent ASR; uncertain negative label'))
        else:
            kept.append(row)
    a.output.mkdir(parents=True)
    for row in kept:
        target = a.output/row['path']; target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to((a.source/row['path']).resolve())
    (a.output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in kept))
    report = dict(role='Training-only negative label precaution; no labels changed', selected=len(kept),
                  excluded=excluded, parent_manifest_sha256=sha(a.source/'manifest.jsonl'),
                  parent_generation_sha256=sha(a.source/'generation.json'),
                  asr_report_sha256=sha(a.report), asr_details_sha256=sha(a.details),
                  source_sha256=sha(Path(__file__)), manifest_sha256=sha(a.output/'manifest.jsonl'),
                  evaluation_accessed=False, labels_changed=0,
                  limits='Target-alias transcription may be an ASR error; quarantine is precautionary. Other text mismatches retained to avoid replacing hard negatives with ASR-easy negatives. Original audio retained unchanged.')
    (a.output/'generation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
