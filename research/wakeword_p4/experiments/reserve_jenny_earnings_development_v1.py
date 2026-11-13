"""Reserve12 new continuous development calls by metadata only, no audio."""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import urllib.request

RESULTS=Path('research/wakeword_p4/results')
OUT=Path('.local/wakeword/jenny-earnings-source-review-v1')
UPSTREAM='c05ab6fd8b4b627d123c922a22a39e993dd37635'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def order(s):return hashlib.sha256(('jenny-fresh-development-earnings-v1:'+s).encode()).hexdigest()


def main():
    prior_path=RESULTS/'4090-hey-earnings-background-reservation-v1.json';prior=json.loads(prior_path.read_text())
    metadata=Path('.local/wakeword/hey-paired-earnings-source-review-v1/metadata.csv')
    assert sha(metadata)==prior['source_metadata']['metadata.csv']['sha256']
    rows=list(csv.DictReader(metadata.open()));assert len(rows)==125
    excluded=set(prior['excluded_prior_tickers'])|{r['metadata']['Ticker Symbol'] for r in prior['records']}
    tickers=sorted({r['Ticker Symbol'] for r in rows}-excluded,key=lambda s:order('ticker:'+s))[:12]
    assert len(tickers)==12
    selected=[min([r for r in rows if r['Ticker Symbol']==t],key=lambda r:order('file:'+r['File ID'])) for t in tickers]
    OUT.mkdir(exist_ok=False)
    def pointer(row):
        identity=row['File ID'];url=f'https://raw.githubusercontent.com/revdotcom/speech-datasets/{UPSTREAM}/earnings22/media/{identity}.mp3'
        with urllib.request.urlopen(url,timeout=60) as response:data=response.read()
        lines=data.decode().splitlines();assert lines[0]=='version https://git-lfs.github.com/spec/v1'
        digest=lines[1].removeprefix('oid sha256:');size=int(lines[2].split()[1]);assert len(digest)==64 and size>0
        path=OUT/f'{identity}.mp3.lfs';path.write_bytes(data)
        return dict(role='development',source_id=identity,metadata=row,source_sha256=digest,source_bytes=size,
            source_url=f'https://media.githubusercontent.com/media/revdotcom/speech-datasets/{UPSTREAM}/earnings22/media/{identity}.mp3',
            pointer_url=url,pointer_path=str(path),pointer_sha256=sha(path),
            transcript_url=f'https://raw.githubusercontent.com/revdotcom/speech-datasets/{UPSTREAM}/earnings22/transcripts/nlp_references/{identity}.nlp')
    with ThreadPoolExecutor(4) as pool:records=list(pool.map(pointer,selected))
    assert len({r['source_id'] for r in records})==12
    assert not ({r['source_id'] for r in records}&({r['source_id'] for r in prior['records']}|set(prior['excluded_prior_source_ids'])))
    hours=sum(int(r['metadata']['File Length (seconds)']) for r in records)/3600;assert hours>=8
    d=dict(schema='jenny-fresh-earnings-development-reservation-v1',upstream_commit=UPSTREAM,
        files={str(p):sha(p) for p in [Path(__file__),prior_path,metadata]},excluded_all_prior_development_and_acceptance_tickers=sorted(excluded),
        selection='Exclude all prior Mac12 and paired24 tickers, including protected acceptance. Rank remaining tickers by SHA256(jenny-fresh-development-earnings-v1:ticker:TICKER), first12; minimum similarly salted file hash per ticker. No audio/transcript/score involved.',
        records=records,metadata_hours=hours,audio_downloaded=False,decoded=False,wake_models_scored=False,
        scope='All12 whole uninterrupted calls mandatory; no source replacement/trim/segmentation. Separate pushed prep lock required for download/decode. >=8decoded hours and transcript alias/history admission before wake scoring.',
        acceptance_accessed=False,training=False,
        publication=prior['publication'],limitations='Company/recording disjointness only, not speaker/pretraining/subclip independence. Metadata hours are not scored exposure. Full encoded/decoded history audit remains.')
    with (RESULTS/'4090-jenny-earnings-development-reservation-v1.json').open('x') as f:f.write(json.dumps(d,indent=2)+'\n')
    print(json.dumps(dict(sources=12,metadata_hours=hours,audio_downloaded=False)))


if __name__=='__main__':main()
