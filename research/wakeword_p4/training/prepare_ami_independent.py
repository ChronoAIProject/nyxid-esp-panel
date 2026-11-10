"""Download prospectively selected AMI test dialogues without detector inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import soundfile as sf

from prepare_ami_training import METADATA, CHANNEL, REVISION, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('metadata', 'selection', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--training-audit', type=Path, action='append', required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh output required; preserve incomplete downloads')
    plan = json.loads(a.selection.read_text())
    if sha(Path(__file__)) != plan['preparer_sha256']:
        raise ValueError('Changed prospectively frozen preparation code')
    for name, (_, digest) in METADATA.items():
        if sha(a.metadata/name) != digest:
            raise ValueError('Changed official split/annotation metadata')
    archive = zipfile.ZipFile(a.metadata/'ami_public_manual_1.6.2.zip')
    meetings = {m.attrib['observation']: m for m in ET.fromstring(archive.read('corpusResources/meetings.xml'))}
    selected = sorted((a.metadata/'split_eval.orig').read_text().split())
    if selected != plan['sessions'] or plan['channel'] != CHANNEL:
        raise ValueError('Changed predeclared session/channel population')
    other = set((a.metadata/'split_train.orig').read_text().split()) | set((a.metadata/'split_dev.orig').read_text().split())
    other |= {n for n, m in meetings.items() if m.attrib.get('seen_type') in ('training', 'development')}
    people = lambda names: {s.attrib['global_name'] for n in names for s in meetings[n]}
    selected_people = people(selected)
    if set(selected) & other or selected_people & people(other):
        raise ValueError('Participant or session overlap with train/development')
    if sorted(sha(path) for path in a.training_audit) != sorted(plan['training_audit_sha256']):
        raise ValueError('Changed audited training source population')
    training_people = set()
    for path in a.training_audit:
        audit = json.loads(path.read_text())
        if (sha(path.parent/'manifest.jsonl') != audit['manifest_sha256']
                or any(r['split'] != 'training' for r in audit['records'])):
            raise ValueError('Invalid actual training audit')
        training_people |= {x.removeprefix('ami_') for r in audit['records'] for x in r['participants']}
    if training_people & selected_people:
        raise ValueError('Actual training participant overlap')
    transcripts = []
    for session in selected:
        for speaker in meetings[session]:
            name = f"words/{session}.{speaker.attrib['nxt_agent']}.words.xml"
            data = archive.read(name)
            words = [w.text for w in ET.fromstring(data) if w.tag == 'w' and w.text and w.attrib.get('punc') != 'true']
            if not words or re.search(r'\b(?:hey\s+(?:nyx|nix|nicks|knicks|nick\s+s)|hi\s+(?:esp|e\s+s\s+p))\b', ' '.join(words), re.I):
                raise ValueError('Empty or target-containing transcript; preserve selection and review before scoring: '+name)
            transcripts.append(dict(path=name, sha256=hashlib.sha256(data).hexdigest(), words=len(words)))
    a.output.mkdir(parents=True)

    def download(session):
        name = f'{session}.{CHANNEL}.wav'
        url = f'https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{session}/audio/{name}'
        target = a.output/name
        with urllib.request.urlopen(url, timeout=60) as response, target.open('xb') as stream:
            length = int(response.headers['Content-Length'])
            while block := response.read(1 << 20):
                stream.write(block)
        if target.stat().st_size != length:
            raise ValueError('Incomplete download: '+name)
        info = sf.info(target)
        pcm, rate = sf.read(target, dtype='int16')
        if rate != 16000 or pcm.ndim != 1 or info.subtype != 'PCM_16' or len(pcm) < 16000*60 or not np.any(pcm):
            raise ValueError('Invalid original PCM: '+name)
        row = dict(path=name, session=session, channel=CHANNEL,
                   participants=sorted('ami_'+s.attrib['global_name'] for s in meetings[session]),
                   role='independent-dialogue-final-source', split='testing',
                   category='farfield-dialogue', language='en', samples=len(pcm), seconds=len(pcm)/rate,
                   sha256=sha(target), pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),
                   url=url, download_content_length=length)
        print(json.dumps(dict(completed=session, source_seconds=row['seconds'])), flush=True)
        return row

    with ThreadPoolExecutor(2) as pool:
        rows = list(pool.map(download, selected))
    if len({r['pcm_sha256'] for r in rows}) != len(selected):
        raise ValueError('Duplicate audio bodies')
    manifest = a.output/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result = dict(role='Fresh independent AMI final negative source preparation; no detector inference',
                  attribution='AMI Meeting Corpus, https://groups.inf.ed.ac.uk/ami/corpus/',
                  license='CC BY 4.0', license_url='https://groups.inf.ed.ac.uk/ami/corpus/license.shtml',
                  source_sha256=sha(Path(__file__)), selection_sha256=sha(a.selection),
                  kaldi_revision=REVISION, metadata_sha256={n:d for n,(_,d) in METADATA.items()},
                  sessions=selected, channel=CHANNEL, unique_participants=len(selected_people),
                  source_hours=sum(r['seconds'] for r in rows)/3600,
                  training_and_development_participants_disjoint=True,
                  actual_training_participants_excluded=len(training_people),
                  transcript_screen=transcripts, records=rows, manifest_sha256=sha(manifest),
                  classifiers_scored=False, waveform_edited=False, training_data_changed=False,
                  limits='One distant microphone per meeting. Sixteen people recur across sixteen meetings; streams are not independent people. Literal public-transcript screening does not establish exhaustive phonetic absence. First acquisition hashes verify local integrity, not publisher checksums. Nominal Poisson bounds will not capture clustering or population shift. Downloading is not scoring or acceptance; a separate model/threshold/scorer preregistration must be pushed before inference. No Nyx/HiESP detector imported or invoked.')
    (a.output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','transcript_screen')}), flush=True)


if __name__ == '__main__':
    main()
