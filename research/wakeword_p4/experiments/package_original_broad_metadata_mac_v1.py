"""Package only the four original target metadata files for Mac census."""
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

def main():
    census = json.loads((ROOT/'research/wakeword_p4/results/4090-hey-broad-metadata-sampling-mass-v1.json').read_text())
    assert census  # Original tool completed; independently bind actual original bytes below.
    source = subprocess.check_output(['git','rev-parse','e1f341a9'], cwd=ROOT, text=True).strip()
    payloads = {}
    for target in ('hey_willow','hey_nyx'):
        for name in ('features.json','target-provenance.json'):
            p = ROOT/'.local/wakeword/hey-broad-training-features-v1'/target/name
            payloads[target+'/'+name] = p.read_bytes()
    manifest = dict(schema='gpu-original-broad-training-metadata-mac-v1', source_commit=source,
                    tool_commit='28eb5a87ac0c8762def8bb3d672c7071055812c7',
                    purpose='Original metadata only for independent expected sampling mass census; no features.bin, audio, model or fresh data.',
                    payloads=[dict(path=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),
                                   source_path='.local/wakeword/hey-broad-training-features-v1/'+n, source_commit=source)
                              for n,b in payloads.items()])
    manifest_bytes=(json.dumps(manifest,indent=2)+'\n').encode()
    out=ROOT/'.local/wakeword/gpu-original-broad-training-metadata-mac-v1.tar.gz'
    with out.open('xb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode='w') as tar:
                for name,data in sorted({**payloads,'manifest.json':manifest_bytes}.items()):
                    info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644
                    tar.addfile(info,io.BytesIO(data))
    result=dict(archive=out.name,bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),manifest=manifest,
                url='https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/'+out.name)
    (ROOT/'research/wakeword_p4/results/4090-original-broad-metadata-mac-package-v1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='manifest'}))

if __name__=='__main__': main()
