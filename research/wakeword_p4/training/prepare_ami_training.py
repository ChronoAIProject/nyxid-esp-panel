"""Prepare fixed AMI training meetings, with participant and transcript guards."""
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

REVISION = 'e02e35f0254bb033fab73d1df99fc34123e31d56'
SESSIONS = ('ES2002a', 'ES2005a', 'IS1000a', 'TS3005a')
CHANNEL = 'Array1-01'
METADATA = {
    'ami_public_manual_1.6.2.zip': (
        'https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip',
        'b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d'),
    'split_train.orig': ('', '49b80e3493c45785df2bb386c1e0d7cfd6da82a5923c7d7a96c87ccc79271c20'),
    'split_dev.orig': ('', 'd62e2073f31437c78029544e104305bae957583107a206070e183c1d7b9b52ff'),
    'split_eval.orig': ('', 'eb2f322acec2e2c86a1b322d134c868b96f49eb5ef47014e23e1dd493e275c67'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-audit', type=Path,
                   help='Optional previously published audit to pin audio download hashes')
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh output required; preserve partial failures')
    a.metadata.mkdir(parents=True, exist_ok=True)
    provenance = []
    for name, (url, digest) in METADATA.items():
        url = url or f'https://raw.githubusercontent.com/kaldi-asr/kaldi/{REVISION}/egs/ami/s5/local/{name}'
        path = a.metadata/name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        if sha(path) != digest:
            raise ValueError('Changed pinned metadata: ' + name)
        provenance.append(dict(file=name, url=url, sha256=digest, bytes=path.stat().st_size))
    archive = zipfile.ZipFile(a.metadata/'ami_public_manual_1.6.2.zip')
    meetings_bytes = archive.read('corpusResources/meetings.xml')
    meetings = {m.attrib['observation']: m for m in ET.fromstring(meetings_bytes)}
    train = set((a.metadata/'split_train.orig').read_text().split())
    reserved = set((a.metadata/'split_dev.orig').read_text().split()) | set((a.metadata/'split_eval.orig').read_text().split())
    # Original AMI and Kaldi splits differ: exclude either scheme's reserved people.
    reserved |= {n for n, m in meetings.items() if m.attrib.get('seen_type') in ('development', 'evaluation')}
    reserved_people = {s.attrib['global_name'] for n in reserved for s in meetings[n]}
    selected_people, jobs, transcripts = set(), [], []
    for session in SESSIONS:
        m = meetings[session]
        people = {s.attrib['global_name'] for s in m}
        if (session not in train or session in reserved or m.attrib.get('seen_type') != 'training'
                or people & reserved_people or people & selected_people or len(people) != 4):
            raise ValueError('Reserved or repeated meeting participants: ' + session)
        selected_people |= people
        for speaker in m:
            name = f"words/{session}.{speaker.attrib['nxt_agent']}.words.xml"
            data = archive.read(name)
            words = [w.text for w in ET.fromstring(data) if w.tag == 'w' and w.text and w.attrib.get('punc') != 'true']
            text = ' '.join(words)
            if not words or re.search(r'\bhey\s+(?:nyx|nix|nicks|knicks|nick\s+s)\b', text, re.I):
                raise ValueError('Empty or literal target transcript requires review: ' + name)
            transcripts.append(dict(path=name, sha256=hashlib.sha256(data).hexdigest(), words=len(words)))
        jobs.append((session, sorted(people)))
    expected = {}
    if a.expected_audit:
        old = json.loads(a.expected_audit.read_text())
        if tuple(old['sessions']) != SESSIONS or old['channel'] != CHANNEL:
            raise ValueError('Expected audit uses a different selection')
        expected = {r['session']: r for r in old['records']}
        if set(expected) != set(SESSIONS):
            raise ValueError('Incomplete expected audio hashes')
    a.output.mkdir(parents=True)

    def download(job):
        session, people = job
        name = f'{session}.{CHANNEL}.wav'
        url = f'https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{session}/audio/{name}'
        target = a.output/name
        with urllib.request.urlopen(url, timeout=60) as response, target.open('xb') as stream:
            length = int(response.headers['Content-Length'])
            while block := response.read(1 << 20):
                stream.write(block)
        if target.stat().st_size != length:
            raise ValueError('Incomplete WAV download')
        info = sf.info(target)
        pcm, rate = sf.read(target, dtype='int16')
        if rate != 16000 or pcm.ndim != 1 or info.subtype != 'PCM_16' or len(pcm) < 16000*60 or not np.any(pcm):
            raise ValueError('Unexpected AMI PCM format/duration')
        row = dict(path=name, speaker='ami_'+session, session=session, participants=['ami_'+s for s in people],
                   split='training', channel=CHANNEL, category='farfield-dialogue', language='en',
                   samples=len(pcm), seconds=len(pcm)/rate, sha256=sha(target),
                   pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),
                   independent_conversation=True, url=url, download_content_length=length)
        if expected and any(row[k] != expected[session][k] for k in ('sha256', 'pcm_sha256', 'samples')):
            raise ValueError('Audio differs from expected audit')
        print(json.dumps(dict(completed=name, seconds=row['seconds'])), flush=True)
        return row

    with ThreadPoolExecutor(2) as pool:
        rows = list(pool.map(download, jobs))
    if len({r['pcm_sha256'] for r in rows}) != len(rows):
        raise ValueError('Duplicate recording bodies')
    (a.output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report = dict(role='Training-only independent AMI far-field meetings; not recognition evidence',
                  attribution='AMI Meeting Corpus, https://groups.inf.ed.ac.uk/ami/corpus/',
                  license='CC BY 4.0', license_url='https://groups.inf.ed.ac.uk/ami/corpus/license.shtml',
                  source_sha256=sha(Path(__file__)), kaldi_revision=REVISION, metadata_sources=provenance,
                  meetings_xml_sha256=hashlib.sha256(meetings_bytes).hexdigest(), sessions=SESSIONS, channel=CHANNEL,
                  selected_before_wake_scoring=True, files=len(rows), unique_participants=len(selected_people),
                  independent_conversation_hours=sum(r['seconds'] for r in rows)/3600,
                  both_split_schemes_reserved_participants_excluded=True, training_transcripts=transcripts,
                  manifest_sha256=sha(a.output/'manifest.jsonl'), records=rows,
                  classifiers_scored=False, reserved_audio_accessed=False, reserved_transcripts_parsed=False,
                  independent_final_audio_accessed=False, existing_training_data_changed=False,
                  expected_audio_hashes_verified=bool(expected),
                  limits='Four distinct training meetings, one distant microphone each; selection is not population-representative. Metadata namespacing does not prove cross-corpus biometric identity disjointness. Literal transcript screen is not exhaustive phonetic truth. First download hashes are local integrity records, not publisher-provided checksums; use expected-audit on reproduction. Full annotation archive includes reserved text, but only selected training word files and shared meeting metadata are parsed. No wake detector or final audio accessed.')
    (a.output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records', 'training_transcripts', 'metadata_sources')}), flush=True)


if __name__ == '__main__':
    main()
