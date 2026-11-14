"""Package immutable rejected numerical screen arrays and existing receipts only."""
import hashlib,json,tarfile,io,gzip,subprocess
from pathlib import Path
R=Path('research/wakeword_p4/results');BASE='4090-hey-broad-oneway-consistency-'
def digest(b):return hashlib.sha256(b).hexdigest()
def main():
 p=R/(BASE+'fixed-grid-screen-result-v1.json');d=json.loads(p.read_text());assert len(d['cases'])==540
 files={c['arrays_path']:c['arrays_sha256'] for c in d['cases']};assert len(files)==540
 receipts=['formal-execution-v1.json','formal-result-v1.json','formal-independent-audit-v1.json','fixed-grid-compile-plan-v1.json','fixed-grid-compile-result-v1.json','fixed-grid-constant-audit-v1.json','fixed-grid-screen-plan-v1.json','fixed-grid-screen-execution-v1.json','fixed-grid-screen-result-v1.json','fixed-grid-screen-independent-audit-v2.json','screen-audit-correction-v2.json','screen-original-audit-failure-v1.log','continuation-original-failure-v1.json']
 for n in receipts:
  p=R/(BASE+n);files[str(p)]=digest(p.read_bytes())
 out=Path('.local/wakeword/oneway-screen-mac-handoff-v1');out.mkdir(exist_ok=False)
 archive=out/'gpu-oneway-full540-saved-arrays-mac-v1.tar.gz';entries=[];commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 with archive.open('xb') as f,gzip.GzipFile(fileobj=f,mode='wb',mtime=0,filename='') as gz,tarfile.open(fileobj=gz,mode='w') as tar:
  for name,h in sorted(files.items()):
   b=Path(name).read_bytes();assert digest(b)==h,name
   entries.append(dict(path=name,size=len(b),sha256=h));info=tarfile.TarInfo(name);info.size=len(b);tar.addfile(info,io.BytesIO(b))
  manifest=dict(schema='oneway-full540-saved-arrays-mac-v1',source_commit=commit,arrays=540,receipts=len(receipts),files=entries,limits='Previously consumed rejected numerical witnesses. No models/audio/new scoring/admission. Original continuation stopped; original errors retained.')
  b=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('manifest.json');info.size=len(b);tar.addfile(info,io.BytesIO(b))
 (R/'gpu-oneway-full540-saved-arrays-mac-v1-manifest.json').write_bytes(b)
 print(json.dumps(dict(path=str(archive),size=archive.stat().st_size,sha256=digest(archive.read_bytes()),members=len(entries)+1)))
if __name__=='__main__':main()
