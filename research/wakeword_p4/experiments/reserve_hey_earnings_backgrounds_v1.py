"""Reserve fresh complete calls using metadata/LFS pointers only; no audio."""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

BASE=Path('.local/wakeword/hey-paired-earnings-source-review-v1')
OUT=Path('research/wakeword_p4/results/4090-hey-earnings-background-reservation-v1.json')
UPSTREAM='c05ab6fd8b4b627d123c922a22a39e993dd37635'
MAC='d0c589b8357b3b41127a724ed5f63df64a68d3b7'
OLD_PATH='research/wakeword_p4/experiments/earnings22_development_source_reservation_v1.json'


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    prior_raw=subprocess.check_output(['git','show',MAC+':'+OLD_PATH])
    assert sha(prior_raw)=='b10b875cbbcfb09bd9b5c2aa41ccb31450b48bc9f7f7c3ce62c4547595135d53'
    prior=json.loads(prior_raw)
    assert sha((BASE/'metadata.csv').read_bytes())==prior['upstream_metadata_sha256']
    rows=list(csv.DictReader((BASE/'metadata.csv').open()))
    assert len(rows)==125
    excluded={r['metadata']['Ticker Symbol'] for r in prior['records']}
    order=lambda s:sha(('hey-paired-earnings-v1:'+s).encode())
    tickers=sorted({r['Ticker Symbol'] for r in rows}-excluded,key=lambda t:order('ticker:'+t))
    chosen=[]
    for role,group in [('development',tickers[:12]),('acceptance',tickers[12:24])]:
        for ticker in group:
            row=min([r for r in rows if r['Ticker Symbol']==ticker],key=lambda r:order('file:'+r['File ID']))
            chosen.append(dict(role=role,source_id=row['File ID'],metadata=row))
    assert len(chosen)==24 and len({r['source_id'] for r in chosen})==24
    pointers=BASE/'pointers';pointers.mkdir(exist_ok=False)
    def fetch(record):
        identity=record['source_id']
        url=f'https://raw.githubusercontent.com/revdotcom/speech-datasets/{UPSTREAM}/earnings22/media/{identity}.mp3'
        with urllib.request.urlopen(url,timeout=60) as response:data=response.read()
        lines=data.decode().splitlines()
        assert lines[0]=='version https://git-lfs.github.com/spec/v1' and lines[1].startswith('oid sha256:')
        digest=lines[1].split(':',1)[1];size=int(lines[2].split()[1]);assert len(digest)==64 and size>0
        path=pointers/(identity+'.mp3.lfs');path.write_bytes(data)
        return dict(record,source_sha256=digest,source_bytes=size,
                    source_url=f'https://media.githubusercontent.com/media/revdotcom/speech-datasets/{UPSTREAM}/earnings22/media/{identity}.mp3',
                    pointer_url=url,pointer_sha256=sha(data),pointer_path=str(path),
                    transcript_url=f'https://raw.githubusercontent.com/revdotcom/speech-datasets/{UPSTREAM}/earnings22/transcripts/nlp_references/{identity}.nlp')
    with ThreadPoolExecutor(4) as pool:records=list(pool.map(fetch,chosen))
    hours={role:sum(int(r['metadata']['File Length (seconds)']) for r in records if r['role']==role)/3600 for role in ('development','acceptance')}
    assert all(h>=8 for h in hours.values())
    result=dict(schema='hey-paired-earnings-background-reservation-v1',upstream_commit=UPSTREAM,
                prior_mac_commit=MAC,prior_mac_reservation_path=OLD_PATH,prior_mac_reservation_sha256=sha(prior_raw),
                source_metadata={r['name']:r for r in json.loads((BASE/'receipt.json').read_text())},
                excluded_prior_source_ids=[r['source_id'] for r in prior['records']],excluded_prior_tickers=sorted(excluded),
                selection='Exclude all tickers in prior Mac12source reservation. Sort remaining tickers by SHA256(hey-paired-earnings-v1:ticker:TICKER); first12 development,next12 acceptance. Per ticker select minimum SHA256(hey-paired-earnings-v1:file:File ID). No audio/transcript/model score used.',
                records=records,metadata_hours=hours,script_sha256=sha(Path(__file__).read_bytes()),
                audio_downloaded=False,decoded=False,wake_models_scored=False,
                scope='Full uninterrupted calls. No clipping/segmentation/normalization/silence removal, retries or source replacement. All12 per role mandatory; decoded duration must be>=8h each. Every event conservatively counted; exact transcript alias hits stop readiness without replacement.',
                access='Only development download/decode may be authorized by separate pushed prep lock. Acceptance media/transcripts remain unopened until candidate/profile/acceptance rules frozen and development passes.',
                limitations='Ticker and recording disjointness is not physical speaker or pretraining independence; analysts may recur. Complete corpus cross-history encoded/PCM overlap audit still required. Company country is not speaker accent. Metadata hours are descriptive, not scored exposure.',
                publication='Original LICENSE explicitly covers transcripts/text CC BY-SA4.0, not the same grant for media; keep media/decoded waveform/frontend local. Publish only provenance/hash/source URL/code.',
                training=False,final_data_accessed=False)
    with OUT.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(metadata_hours=hours,sources=len(records),reservation_sha256=sha(OUT.read_bytes()),audio_downloaded=False)))


if __name__=='__main__':main()
