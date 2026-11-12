"""Package original training-only pilot and pinned synthesis sources, no regeneration."""
import gzip,hashlib,json,tarfile
from pathlib import Path
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=Path('.local/wakeword');audio=base/'us-training-coverage-v1';source=base/'piper-us-training-source-v1'
out=base/'gpu-us-training-coverage-v1.tar.gz';manifest=base/'us-training-coverage-handoff-manifest-v1.json'
assert not out.exists() and not manifest.exists()
files={}
for folder,root in [('audio',audio),('source',source)]:
 for p in sorted(root.rglob('*')):
  if p.is_file():files[folder+'/'+p.relative_to(root).as_posix()]=p
assert len([n for n in files if n.endswith('.wav')])==108 and len([n for n in files if n.endswith('.npy')])==216
for p in [Path('research/wakeword_p4/experiments/audit_us_training_coverage_v1.py'),Path('research/wakeword_p4/results/4090-us-training-coverage-audit-v1.json')]:files['audit/'+p.name]=p
m=dict(scope='Original108 training-only synthesis outputs and pinned original sources, for Mac mechanical audit only',source_commit='f013dcd',source_manifest_sha256=sha(source/'source.json'),files={name:dict(size_bytes=p.stat().st_size,sha256=sha(p),source_commit='f013dcd' if not name.startswith('audit/') else 'audit revision recorded by contained script SHA') for name,p in files.items()},limits='No wake-model scoring, training, playback, new development/final material or credentials. Original artifacts retained byte-for-byte. ONNX models included for provenance, no regeneration required.')
manifest.write_text(json.dumps(m,indent=2)+'\n');files['manifest.json']=manifest
with out.open('xb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as compressed:
  with tarfile.open(fileobj=compressed,mode='w') as tar:
   for name,p in sorted(files.items()):
    info=tar.gettarinfo(str(p),arcname=name);info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
    with p.open('rb') as content:tar.addfile(info,content)
print(json.dumps(dict(path=str(out),size_bytes=out.stat().st_size,sha256=sha(out),members=len(files),manifest_sha256=sha(manifest))))
