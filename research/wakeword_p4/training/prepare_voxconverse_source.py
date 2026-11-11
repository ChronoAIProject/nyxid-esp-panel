"""Pin VoxConverse mirror metadata; inspect only first development shard."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPO='diarizers-community/voxconverse'
REV='3acfa1b45ca4b7419aee999d67d94c617f9c9d47'
PRIMARY='24bf60be297701cd7e4ef18550c6d390c1b87365'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fetch(url,path):
    with urllib.request.urlopen(url,timeout=60) as response,path.open('xb') as out:
        while chunk:=response.read(4*1024*1024): out.write(chunk)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    tree_url=f'https://huggingface.co/api/datasets/{REPO}/tree/{REV}/data'
    with urllib.request.urlopen(tree_url,timeout=30) as r: tree=json.load(r)
    wanted=[f'data/dev-{i:05d}-of-00005.parquet' for i in range(2)]+[f'data/test-{i:05d}-of-00011.parquet' for i in range(4)]
    selected=[x for x in tree if x['path'] in wanted]
    assert len(selected)==6
    metadata={}
    for name,url in [('primary-README.md',f'https://raw.githubusercontent.com/joonson/voxconverse/{PRIMARY}/README.md'),
                     ('mirror-README.md',f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/README.md')]:
        fetch(url,a.output/name);metadata[name]=dict(url=url,sha256=sha(a.output/name))
    source=dict(repository=REPO,revision=REV,primary_repository='joonson/voxconverse',primary_revision=PRIMARY,
        primary_archive_probe='Official test ZIP HTTP401; mirror explicitly identified, not original-byte authenticated',
        metadata=metadata,selected_shards=selected,development_shards=wanted[:2],reserved_final_shards=wanted[2:],
        selection='First2 dev and first4 test shards by index fixed before audio/model scores; current task downloads dev0 only',
        license='CC BY 4.0 as stated in primary README and mirror card',
        script_sha256=sha(Path(__file__)),wake_models_scored=False,final_audio_downloaded=False,
        limits='Mirror declares diarizers preprocessing; exact original archive PCM equivalence not established. Local speaker labels need not be global identities; no cross-corpus person-disjointness claim. No transcripts supplied, so target phrase absence not certified. Exposure and continuity must be audited before acceptance use.')
    (a.output/'source.json').write_text(json.dumps(source,indent=2)+'\n')
    row=next(x for x in selected if x['path']==wanted[0]);name=Path(row['path']).name
    print(json.dumps(dict(downloading=name,expected_size=row['size'])),flush=True)
    dest=a.output/name;fetch(f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{row['path']}",dest)
    assert dest.stat().st_size==row['size'] and sha(dest)==row['lfs']['oid']
    report=dict(downloaded=name,size_bytes=dest.stat().st_size,sha256=sha(dest),
        source_sha256=sha(a.output/'source.json'),wake_models_scored=False,final_audio_downloaded=False)
    (a.output/'download.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__': main()
