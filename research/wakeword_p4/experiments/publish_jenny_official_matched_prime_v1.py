"""Lossless original official runtime evidence, only after terminal complete scoring."""
import gzip,hashlib,io,json,tarfile
from pathlib import Path
R=Path('research/wakeword_p4/results');PLAN=R/'4090-jenny-official-matched-prime-plan-v1.json'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=json.loads(PLAN.read_text());base=Path(p['output']);c=json.loads((base/'complete.json').read_text());assert c['streams']==408 and c['plan_sha256']==sha(PLAN)
    scored=R/'4090-jenny-official-matched-prime-behavior-v1.json';s=json.loads(scored.read_text());assert s['plan_sha256']==sha(PLAN)
    files=[base/'launch.json',base/'complete.json']
    for i in range(408):files += [base/f'{i:03}-{name}' for name in ['report.json','serial.log','stage-monitor.txt','raw-returns.bin']]
    out=R/'4090-jenny-official-matched-prime-original-evidence-v1.tar.gz'
    with out.open('xb') as target,gzip.GzipFile(fileobj=target,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
        for path in files:
            data=path.read_bytes();info=tarfile.TarInfo(str(path.relative_to(base)));info.size=len(data);info.mode=0o644;info.mtime=0;tar.addfile(info,io.BytesIO(data))
    # Verify the durable lossless archive before publishing its receipt.
    with tarfile.open(out) as tar:
        assert tar.getnames()==[str(f.relative_to(base)) for f in files]
        for member,path in zip(tar.getmembers(),files,strict=True):assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==sha(path)
    receipt=dict(schema='jenny-official-matched-prime-evidence-v1',path=str(out),bytes=out.stat().st_size,sha256=sha(out),members=len(files),every_member_verified=True,summary_sha256=sha(scored),plan_sha256=sha(PLAN),original_session=1876,full_records=408,acceptance_accessed=False,limits=p['policy'])
    with (R/'4090-jenny-official-matched-prime-evidence-manifest-v1.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
if __name__=='__main__':main()
