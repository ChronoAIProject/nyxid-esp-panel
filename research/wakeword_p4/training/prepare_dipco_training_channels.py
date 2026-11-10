"""Add fixed alternate microphone positions from existing training sessions only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import soundfile as sf
from download_dipco_development import REPO, REVISION, sha

CHANNELS = ('U02.CH1','U03.CH1')
SESSIONS = ('S02','S04')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    audit=json.loads((a.base_source/'audit.json').read_text())
    if sha(a.base_source/'manifest.jsonl')!=audit['manifest_sha256'] or audit['revision']!=REVISION or audit['repo']!=REPO:
        raise ValueError('Changed baseline source provenance')
    rows=[json.loads(x) for x in (a.base_source/'manifest.jsonl').read_text().splitlines()]
    by_session={r['session']:r for r in rows};split_by_participant={}
    for r in rows:
        for person in r['participants']:
            if person in split_by_participant and split_by_participant[person]!=r['split']:
                raise ValueError('Existing participant split contamination')
            split_by_participant[person]=r['split']
    entries={r['path']:r for path in sorted(a.base_source.glob('tree-*.json')) for r in json.loads(path.read_text())}
    jobs=[]
    for session in SESSIONS:
        r=by_session[session]
        transcript=a.base_source/(session+'.json')
        if r['split']!='training' or sha(transcript)!=r['transcript_sha256']:
            raise ValueError('Only original audited training sessions')
        utterances=json.loads(transcript.read_text())
        if sorted({u['speaker_id'] for u in utterances})!=r['participants'] or any(split_by_participant[x]!='training' for x in r['participants']):
            raise ValueError('Training participant mismatch')
        if any(re.search(r'\bhey\s+(?:nyx|nix|nicks|knicks|nick\s+s)\b',u['words'],re.I) for u in utterances):
            raise ValueError('Literal target candidate needs review')
        for channel in CHANNELS:
            remote=f'audio/dev/{session}_{channel}.wav'
            jobs.append((r,channel,remote,entries[remote]))
    a.output.mkdir(parents=True)
    def download(job):
        r,channel,remote,entry=job
        path=a.output/Path(remote).name
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{remote}'
        with urllib.request.urlopen(url,timeout=60) as response,path.open('xb') as target:
            while block:=response.read(1<<20):target.write(block)
        if path.stat().st_size!=entry['size'] or sha(path)!=entry['lfs']['oid']:
            raise ValueError('Pinned LFS size/hash mismatch')
        info=sf.info(path);pcm,rate=sf.read(path,dtype='int16')
        if rate!=16000 or pcm.ndim!=1 or info.subtype!='PCM_16' or len(pcm)!=r['samples']:
            raise ValueError('Unexpected format/session duration')
        result=dict(r,path=path.name,speaker=f"dipco_{r['session']}_{channel.replace('.','_')}",channel=channel,
                    sha256=sha(path),pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),source_path=remote,
                    same_conversation_as=r['speaker'],independent_conversation=False)
        print(json.dumps(dict(completed=path.name,source_seconds=len(pcm)/16000)),flush=True)
        return result
    with ThreadPoolExecutor(2) as pool:result=list(pool.map(download,jobs))
    if len({r['pcm_sha256'] for r in result})!=len(result) or set(r['pcm_sha256'] for r in result)&{r['pcm_sha256'] for r in rows}:
        raise ValueError('Exact duplicate PCM')
    (a.output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in result))
    report=dict(role='Training-only alternate DiPCo microphone channels, not new conversations or quality evidence',
                repo=REPO,revision=REVISION,license=audit['license'],source_sha256=sha(Path(__file__)),
                baseline_audit_sha256=sha(a.base_source/'audit.json'),manifest_sha256=sha(a.output/'manifest.jsonl'),
                sessions=SESSIONS,channels=CHANNELS,files=len(result),
                training_waveform_channel_hours=sum(r['seconds'] for r in result)/3600,
                new_independent_conversation_hours=0,existing_unique_conversation_hours=sum(by_session[s]['seconds'] for s in SESSIONS)/3600,
                participant_split_disjoint=True,validation_audio_accessed=False,independent_final_audio_accessed=False,
                classifiers_scored=False,limits='Alternate microphones on the same two existing training conversations; correlated channels add acoustic positions, no new independent speakers/conversation exposure. Selected device2/3 channel1 before scoring. Transcript literal alias screen only, not exhaustive phonetic truth. Never change current running v26 dataset.')
    (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__ == '__main__':main()
