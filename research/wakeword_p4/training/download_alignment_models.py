"""Download exact alignment model assets from a previously published hash manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    rows=json.loads(a.sources.read_text())
    if sorted(r['kind'] for r in rows)!=['acoustic','dictionary']:raise ValueError('Need pinned model and dictionary')
    a.output.mkdir(parents=True)
    for row in rows:
        name='english_us_arpa.zip' if row['kind']=='acoustic' else 'english_us_arpa.dict'
        path=a.output/name;digest=hashlib.sha256()
        with urllib.request.urlopen(row['url'],timeout=60) as response,path.open('xb') as stream:
            while block:=response.read(1<<20):stream.write(block);digest.update(block)
        if path.stat().st_size!=row['bytes'] or digest.hexdigest()!=row['sha256']:raise ValueError('Model asset size/hash mismatch')
    (a.output/'download.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
