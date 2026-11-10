"""Audit mined training windows and their full jitter footprint before reuse."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import soundfile as sf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('mining','features','corpus','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--jitter-frames',type=int,default=12)
    a=p.parse_args()
    if a.output.exists() or not 0<=a.jitter_frames<=30:p.error('Fresh output; jitter0..30')
    mining=json.loads(a.mining.read_text());cache=json.loads((a.features/'cache.json').read_text())
    if (sha(a.features/'human.json')!=mining['feature_index_sha256'] or
            sha(a.features/'cache.json')!=mining['feature_cache_sha256']):
        raise ValueError('Changed training cache')
    sources=json.loads((a.features/'human.json').read_text())
    if any(r['split']!='training' for r in sources):raise ValueError('Training-only cache required')
    by_speaker={r['speaker']:r for r in sources};transcripts={}
    for item in cache['transcripts']:
        path=a.corpus/item['path']
        if sha(path)!=item['sha256']:raise ValueError('Changed transcript')
        for line in path.read_text().splitlines():
            key,text=line.split(maxsplit=1);transcripts[key]=text
    accepted=[];quarantined=[];checked={};timelines={}
    for index,candidate in enumerate(mining['candidates']):
        row=by_speaker[candidate['speaker']]
        if candidate['split']!='training' or candidate['feature_sha256']!=row['feature_sha256']:
            raise ValueError('Candidate differs from training stream')
        if row['speaker'] not in timelines:
            at=0;segments=[]
            for source in row['sources']:
                end=at+source['samples'];segments.append((at,end,source))
                at=end+3200+(-source['samples']-3200)%160
            timelines[row['speaker']]=segments
        lo=max(0,candidate['frame_start']-a.jitter_frames)*160
        hi=(candidate['frame_end']+a.jitter_frames+2)*160
        overlap=[]
        for start,end,source in timelines[row['speaker']]:
            if start>=hi or end<=lo:continue
            relative=source['path'];path=(a.corpus/relative).resolve()
            if not path.is_relative_to(a.corpus.resolve()):raise ValueError('Escaped source')
            if relative not in checked:
                if sha(path)!=source['sha256']:raise ValueError('Changed FLAC')
                pcm,rate=sf.read(path,dtype='int16')
                if (rate!=16000 or pcm.ndim!=1 or len(pcm)!=source['samples'] or
                        hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()!=source['pcm_sha256']):
                    raise ValueError('Decoded training PCM mismatch')
                checked[relative]=source
            overlap.append(dict(path=relative,text=transcripts[path.stem],
                                sha256=source['sha256'],pcm_sha256=source['pcm_sha256']))
        if not overlap:raise ValueError('No speech in full jitter footprint')
        phrase=' '.join(s['text'] for s in overlap)
        suspect=bool(candidate['quarantined'] or re.search(r'\bhey\s+(?:nyx|nix|nicks|knicks|nick\s+s)\b',phrase,re.I))
        record={**candidate,'candidate_id':index,'jitter_frames':a.jitter_frames,'sources':overlap}
        (quarantined if suspect else accepted).append(record)
    if not accepted:raise ValueError('No audited negative candidates')
    report=dict(role='Audited training-only hard ambient windows',mining_report_sha256=sha(a.mining),
                feature_index_sha256=mining['feature_index_sha256'],source_cache_sha256=sha(a.features/'cache.json'),
                mining_model_sha256=mining['model_sha256'],source_sha256=sha(Path(__file__)),
                context_frames=150,jitter_frames=a.jitter_frames,candidates=accepted,quarantined=quarantined,
                checked_source_files=len(checked),validation_audio_accessed=False,independent_final_audio_accessed=False,
                label_policy='Original LibriSpeech human reference transcripts, screened over full jitter footprint including utterance joins; source FLAC/PCM hashes checked. Literal alias screen only, not independent phonetic truth. No ASR-driven relabeling.',
                limits='Selected on training data with fixed previous model; deliberately biased distribution, not quality evidence or added independent audio exposure.')
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['candidates','quarantined']}),flush=True)
    print(json.dumps(dict(accepted=len(accepted),quarantined=len(quarantined))),flush=True)


if __name__=='__main__':main()
