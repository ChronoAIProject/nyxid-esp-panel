"""Verify and archive completed paired coverage originals, no model invocation."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
R=Path('research/wakeword_p4/results')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main(cohort):
    assert cohort in ['primed','jenny','joe']
    out=Path(f'.local/wakeword/context-pair-{cohort}-coverage-v1');result=json.loads((out/'result.json').read_text());assert result['all_passed']
    state=json.load(open('.local/wakeword/context-pair-coverage-execution-v1/state.json'))
    assert all(next(a for a in state if a['cohort']==cohort and a['action']==action)['exit_code']==0 for action in ['run','score'])
    files=[out/'result.json',out/'launch.json']
    for model in result['models']:
        report=Path(model['path']);assert sha(report)==model['sha256'];p=json.loads(report.read_text());assert p['all_passed'];files.extend([report,report.parent/'records.jsonl'])
        for row in p['records']:
            f=Path(row['trace_path']);assert sha(f)==row['trace_sha256'];files.append(f)
    manifest={str(f.relative_to(out)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in files}
    archive=R/f'4090-context-pair-{cohort}-coverage-traces-v1.tar.gz'
    with tarfile.open(archive,'x:gz') as t:
        for f in files:t.add(f,arcname=str(f.relative_to(out)))
    with tarfile.open(archive) as t:
        for name,info in manifest.items():
            data=t.extractfile(name).read();assert len(data)==info['bytes'] and hashlib.sha256(data).hexdigest()==info['sha256']
    (R/f'4090-context-pair-{cohort}-coverage-original-v1.json').write_bytes((out/'result.json').read_bytes())
    comparison=R/f'4090-context-pair-{cohort}-coverage-comparison-v1.json';assert comparison.exists()
    (R/f'4090-context-pair-{cohort}-coverage-archive-v1.json').write_text(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),comparison_sha256=sha(comparison),members=manifest,all_members_verified=True,replay_exit_code=0,score_exit_code=0),indent=2)+'\n')
    print(json.dumps(dict(cohort=cohort,bytes=archive.stat().st_size,members=len(manifest))))
if __name__=='__main__':main(sys.argv[1])
