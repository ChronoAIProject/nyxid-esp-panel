"""Metadata-only canonical voice-family model-card review, no model/audio use."""
import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path

ROOT=Path('research/wakeword_p4');CATALOG=ROOT/'results/4090-hey-voice-coverage-catalog-v1.json'
OUT=Path('.local/wakeword/hey-coverage-source-cards-v1');RESULT=ROOT/'results/4090-hey-coverage-source-cards-v1.json'
def digest(data):return hashlib.sha256(data).hexdigest()
def main():
    c=json.loads(CATALOG.read_text());groups={}
    for row in c['english_entries']:groups.setdefault(row['key'].rsplit('-',1)[0],[]).append(row)
    # Canonical choice is fixed from metadata only, before reading card contents.
    selected=[sorted(rows,key=lambda r:({'medium':0,'high':1,'low':2}[r['quality']],r['key']))[0] for family,rows in sorted(groups.items())]
    cache={}
    for path in list((ROOT/'results').glob('*card*.md'))+list(Path('.local/wakeword/hey-paired-prospective-models-v1').glob('*/MODEL_CARD')):
        data=path.read_bytes();cache.setdefault(hashlib.md5(data).hexdigest(),(path,data))
    OUT.mkdir(exist_ok=False)
    def get(row):
        paths=[(k,v) for k,v in row['files'].items() if k.endswith('/MODEL_CARD')];assert len(paths)==1
        key,meta=paths[0];url=f"https://huggingface.co/rhasspy/piper-voices/resolve/{c['revision']}/{key}"
        try:
            if meta['md5_digest'] in cache:
                original,data=cache[meta['md5_digest']];reused=str(original)
            else:data=urllib.request.urlopen(url,timeout=30).read();reused=None
            assert len(data)==meta['size_bytes'] and hashlib.md5(data).hexdigest()==meta['md5_digest']
            path=OUT/(row['key']+'.md');path.write_bytes(data)
            return dict(voice=row['key'],family=row['key'].rsplit('-',1)[0],url=url,path=str(path),bytes=len(data),sha256=digest(data),reused_existing_path=reused,text=data.decode('utf-8'),source_admitted=False)
        except Exception as exc:return dict(voice=row['key'],url=url,error=f'{type(exc).__name__}: {exc}',source_admitted=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:cards=list(pool.map(get,selected))
    result=dict(schema='hey-coverage-source-card-review-v1',reviewer_sha256=digest(Path(__file__).read_bytes()),catalog_sha256=digest(CATALOG.read_bytes()),revision=c['revision'],canonical_rule='One model per family, medium then high then low, lexical tiebreak, fixed before card reads',cards=cards,reused=sum(bool(r.get('reused_existing_path')) for r in cards),failed=sum('error' in r for r in cards),models_downloaded=False,audio_generated=False,voices_reserved=False,limits='Metadata/source-card collection only. Catalog MD5/length checked and actual SHA256 recorded; shared ancestry and physical-speaker identity require interpretation. No absence-of-role or pretraining-independence claim and no training/generation authorization.')
    with RESULT.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(cards=len(cards),reused=result['reused'],failed=result['failed'],models_downloaded=False)))
if __name__=='__main__':main()
