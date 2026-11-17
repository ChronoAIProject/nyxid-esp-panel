"""Path-only archive correction; do not repeat inference or independent arbitration."""
import hashlib,json,tarfile
from pathlib import Path
R=Path('research/wakeword_p4/results');ROOT=Path.cwd().resolve()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=[R/'4090-parallel-history-expert-scope-plan-v1.json',R/'4090-parallel-history-expert-scope-original-v1.json',R/'4090-parallel-history-expert-scope-review-v1.json',R/'4090-parallel-history-expert-scope-archive-failure-v1.json',Path('research/wakeword_p4/experiments/replay_parallel_history_expert_scope_v1.py'),Path('research/wakeword_p4/experiments/review_parallel_history_expert_scope_v1.py'),Path(__file__)]
assert json.loads(files[2].read_text())['all_passed']
ledger=[dict(path=str(f.resolve().relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in files];archive=R/'4090-parallel-history-expert-scope-evidence-v2.tar.gz'
with tarfile.open(archive,'x:gz') as tf:
 for f,s in zip(files,ledger,strict=True):tf.add(f,arcname=s['path'],recursive=False)
with tarfile.open(archive,'r:gz') as tf:
 assert len(tf.getmembers())==len(ledger)
 for m,s in zip(tf.getmembers(),ledger,strict=True):
  b=tf.extractfile(m).read();assert m.name==s['path'] and len(b)==s['bytes'] and hashlib.sha256(b).hexdigest()==s['sha256']
r=dict(schema='parallel-history-expert-scope-archive-v2',archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=ledger),all_member_hashes_verified=True,scope='Path-only packing recovery; original report/independent event review bytes unchanged. No inference, decoding or aggregation rerun. Failed v1 preserved. Existing source raw archives retained by reference, no PCM.')
with (R/'4090-parallel-history-expert-scope-archive-v2.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(dict(archive=str(archive),bytes=r['archive']['bytes'],sha256=r['archive']['sha256'],members=len(ledger))))
