"""Publish original saved SAM context NPZ bytes, no inference or scoring."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
R=Path('research/wakeword_p4/results');source=R/'4090-binary40-sam-sam-context-original-v1.json'
def sha(b):return hashlib.sha256(b).hexdigest()
d=json.loads(source.read_text());members={};payloads={}
for row in d['streams']:
 p=Path(row['trace']['path']);b=p.read_bytes();assert len(b)==row['trace']['bytes'] and sha(b)==row['trace']['sha256']
 name=row['name']+'.npz';payloads[name]=b;members[name]=dict(bytes=len(b),sha256=sha(b),original_path=str(p),source_commit='97b1b13d')
manifest=dict(schema='sam-context-raw-traces-manifest-v1',source=dict(path=str(source),sha256=sha(source.read_bytes()),commit='97b1b13d'),members=members,limits='Original uint8/native-clock bytes, complete consumed contexts. No new inference or P4 qualification; retain C161.')
mp=R/'4090-sam-context-raw-traces-manifest-v1.json';mb=(json.dumps(manifest,indent=2)+'\n').encode()
with mp.open('xb') as f:f.write(mb)
payloads['manifest.json']=mb
archive=R/'4090-sam-context-raw-traces-v1.tar.gz'
with tarfile.open(archive,'x:gz') as t:
 for name,b in payloads.items():
  info=tarfile.TarInfo(name);info.size=len(b);info.mode=0o644;info.mtime=0;t.addfile(info,io.BytesIO(b))
with tarfile.open(archive,'r:gz') as t:
 assert len(t.getnames())==len(set(t.getnames()))==4 and set(t.getnames())==set(payloads)
 for m in t:assert t.extractfile(m).read()==payloads[m.name]
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()))))
