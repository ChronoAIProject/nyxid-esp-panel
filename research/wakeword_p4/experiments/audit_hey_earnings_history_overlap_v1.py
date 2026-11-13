"""Compare completed new development hashes to immutable history/local ledgers.

Reads metadata only, never historical/final audio or a wake detector. Exact
hash disjointness cannot certify no excerpt, speaker or re-encoding overlap.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import soundfile as sf

BASE=Path('.local/wakeword')
PREP=BASE/'hey-paired-earnings-development-v1'
OUT=BASE/'hey-paired-earnings-history-audit-v1'
RESULT=Path('research/wakeword_p4/results/4090-hey-earnings-history-overlap-v1.json')
COMMITS=['064423c','d0c589b8357b3b41127a724ed5f63df64a68d3b7']
HEX=re.compile(rb'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])')
EXCLUDED=[PREP,OUT,BASE/'hey-paired-earnings-source-review-v1']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    manifest_path=PREP/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    assert manifest['all_complete'] and manifest['integrity_preparation_passed'] and len(manifest['records'])==12
    rows=manifest['records']
    plan_path=Path('research/wakeword_p4/results/4090-hey-earnings-development-preparation-plan-v1.json')
    reservation_path=Path('research/wakeword_p4/results/4090-hey-earnings-background-reservation-v1.json')
    assert manifest['plan_sha256']==sha(plan_path)
    plan=json.loads(plan_path.read_text());assert plan['reservation_sha256']==sha(reservation_path)
    expected=[r for r in json.loads(reservation_path.read_text())['records'] if r['role']=='development']
    assert [r['source_id'] for r in rows]==[r['source_id'] for r in expected]
    for row,source in zip(rows,expected,strict=True):
        original=PREP/'original'/(row['source_id']+'.mp3')
        assert sha(original)==row['source_sha256']==source['source_sha256']
        assert original.stat().st_size==source['source_bytes']
        pcm_hash=hashlib.sha256();samples=0
        for block in sf.blocks(row['wav_path'],dtype='int16',blocksize=160000):
            pcm_hash.update(block.astype('<i2').tobytes());samples+=len(block)
        assert pcm_hash.hexdigest()==row['pcm_sha256'] and samples==row['samples']
        assert sha(Path(row['wav_path']))==row['wav_sha256']
        assert sha(Path(row['frontend_path']))==row['frontend_sha256']
    wanted={r[k] for r in rows for k in ('source_sha256','wav_sha256','pcm_sha256')}
    assert len(wanted)==36,'Duplicate candidate source objects'
    OUT.mkdir(exist_ok=False)
    inventory=[];matches=[]
    for revision in COMMITS:
        commit=subprocess.check_output(['git','rev-parse',revision],text=True).strip()
        entries=subprocess.check_output(['git','ls-tree','-rl',commit,'research/wakeword_p4'],text=True).splitlines()
        with subprocess.Popen(['git','cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE) as proc:
            for entry in entries:
                head,path=entry.split('\t',1)
                if not path.endswith(('.json','.jsonl')):continue
                _,kind,oid,size=head.split();assert kind=='blob'
                proc.stdin.write((oid+'\n').encode());proc.stdin.flush()
                header=proc.stdout.readline().split();assert header[1]==b'blob'
                data=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n'
                hit=sorted(wanted & {s.decode() for s in HEX.findall(data)})
                inventory.append(dict(scope='git',commit=commit,path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
                if hit:matches.append(dict(scope='git',commit=commit,path=path,hashes=hit))
            proc.stdin.close();assert proc.wait()==0
    count=0
    for path in sorted(BASE.rglob('*')):
        if not path.is_file() or path.suffix not in ('.json','.jsonl') or any(p in path.parents for p in EXCLUDED):continue
        h=hashlib.sha256();tail=b'';hits=set();size=0
        with path.open('rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):
                h.update(block);size+=len(block)
                hits.update(wanted & {s.decode() for s in HEX.findall(tail+block)})
                tail=block[-65:]
        inventory.append(dict(scope='local',path=str(path),bytes_read=size,sha256=h.hexdigest()))
        if hits:matches.append(dict(scope='local',path=str(path),hashes=sorted(hits)))
        count+=1
    ledger=OUT/'inventory.json';ledger.write_text(json.dumps(inventory,indent=2)+'\n')
    result=dict(schema='hey-earnings-history-overlap-audit-v1',manifest_sha256=sha(manifest_path),
                script_sha256=sha(Path(__file__)),candidate_hashes=len(wanted),
                git_commits=COMMITS,git_metadata_files=sum(r['scope']=='git' for r in inventory),
                local_metadata_files=count,excluded_current_source_metadata=[str(p) for p in EXCLUDED],
                inventory_path=str(ledger),inventory_sha256=sha(ledger),matches=matches,
                no_exact_encoded_or_pcm_hash_matches=not matches,
                overlap_admitted=False,scoring_authorized=False,acceptance_audio_opened=False,
                limits='Bounded exact hash audit of two published histories and current accessible JSON/JSONL ledgers. Does not decode old audio or prove no subclips/re-encodings, deleted/unrecorded files or physical-speaker overlap. Local mutable metadata is snapshot evidence, not an execution pin. No negative-score filtering.')
    with RESULT.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
