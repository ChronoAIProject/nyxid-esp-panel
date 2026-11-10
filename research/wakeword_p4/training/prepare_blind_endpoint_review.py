"""Package unchanged development audio for human endpoint review without detector outputs."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import random

import soundfile as sf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alignment-inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh output required')
    parent = a.alignment_inputs/'manifest.json'
    source = json.loads(parent.read_text())
    if source['independent_final_audio_accessed'] or source['wake_scores_accessed']:
        raise ValueError('Only existing unscored development alignment inputs')
    records, mapping = [], []
    for row in source['records']:
        path = (a.alignment_inputs/'corpus'/row['path']).resolve()
        if not path.is_relative_to(a.alignment_inputs.resolve()) or sha(path) != row['sha256']:
            raise ValueError('Changed or escaped development waveform')
        pcm, rate = sf.read(path, dtype='int16')
        if (rate != 16000 or pcm.ndim != 1 or len(pcm) != row['samples']
                or hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest() != row['pcm_sha256']):
            raise ValueError('Changed development PCM')
        identity = hashlib.sha256(('blind-endpoint-v1:'+row['path']+row['sha256']).encode()).hexdigest()[:16]
        records.append(dict(id=identity, intended_text=row['transcript'], samples=len(pcm), rate=rate,
                            wav_sha256=row['sha256'], pcm_sha256=row['pcm_sha256'],
                            audio='data:audio/wav;base64,'+base64.b64encode(path.read_bytes()).decode()))
        mapping.append(dict(id=identity, source_path=row['path'], source_kind=row['source_kind'],
                            target=row['target'], wav_sha256=row['sha256']))
    if len(records) != 26 or len({r['id'] for r in records}) != len(records):
        raise ValueError('Need unchanged 24 development targets and two official controls')
    random.Random(726).shuffle(records)
    pack = dict(protocol='blind-human-endpoint-review-v1', parent_manifest_sha256=sha(parent),
                rate=16000, records=records)
    payload = json.dumps(pack, separators=(',', ':'))
    pack_id = hashlib.sha256(payload.encode()).hexdigest()
    pack['pack_id'] = pack_id
    template = Path(__file__).with_name('blind_endpoint_review.html')
    html = template.read_text().replace('/* REVIEW_DATA */null', json.dumps(pack).replace('<', '\\u003c'))
    a.output.mkdir(parents=True)
    (a.output/'index.html').write_text(html)
    audit = dict(role='Blind human development endpoint review inputs; no human labels yet',
                 source_sha256=sha(Path(__file__)), template_sha256=sha(template),
                 html_sha256=sha(a.output/'index.html'), pack_id=pack_id,
                 parent_manifest_sha256=sha(parent), clips=len(records), mapping=mapping,
                 detector_outputs_included=False, automatic_boundaries_included=False,
                 human_annotations_present=False, independent_final_audio_accessed=False,
                 limits='Human must independently review pronunciation and endpoint uncertainty. No automatic gold labels. Development review alone does not validate final-positive endpoints or prove inter-rater accuracy. Existing PCM and historical scoring windows unchanged.')
    (a.output/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='mapping'}), flush=True)


if __name__ == '__main__':
    main()
