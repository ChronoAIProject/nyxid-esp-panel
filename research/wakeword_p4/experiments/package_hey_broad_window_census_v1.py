"""Publish saved coordinate evidence only, no model/audio/features."""
import gzip,hashlib,io,json,subprocess,tarfile
from pathlib import Path
R=Path('research/wakeword_p4/results')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    result=R/'4090-hey-broad-surviving-window-census-v2.json';audit=R/'4090-hey-broad-surviving-window-independent-audit-v1.json'
    a=json.loads(audit.read_text());assert a['all_passed'] and a['original_result_sha256']==sha(result)
    c=json.loads(result.read_text());payloads={}
    for t in c['targets']:
        p=Path(t['trace_path']);assert sha(p)==t['trace_sha256'];payloads['traces/'+p.name]=p
    for n in ['4090-hey-broad-surviving-window-census-execution-v1.json','4090-hey-broad-surviving-window-census-execution-v2.json','4090-hey-broad-surviving-window-census-preflight-failure-v1.json',result.name,audit.name,'4090-hey-broad-metadata-sampling-mass-v1.json']:
        payloads['results/'+n]=R/n
    for n in ['census_hey_broad_surviving_windows_v1.py','census_hey_broad_surviving_windows_v2.py','audit_hey_broad_surviving_window_trace_v1.py',Path(__file__).name]:payloads['source/'+n]=Path(__file__).with_name(n)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    manifest=dict(source_commit=commit,payloads=[dict(path=n,bytes=p.stat().st_size,sha256=sha(p),source_path=str(p)) for n,p in sorted(payloads.items())],
                  limits='CPU original sampler replay coordinates with original batch hash identity; not historical live provenance or recognition. First10000 per word. Original metadata dependency: gpu-original-broad-training-metadata-mac-v1.tar.gz SHA256 fe15ab839bb6df3f17b264f6b84bd48f5d6199f7665bbac4aec75ddef24d0e9b. No audio/model/features.bin.')
    out=Path('.local/wakeword/gpu-original-broad-surviving-window-census-v2.tar.gz')
    with out.open('xb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
        for n,p in sorted(payloads.items()):
            info=tarfile.TarInfo(n);info.size=p.stat().st_size;info.mode=0o644
            with p.open('rb') as f:tar.addfile(info,f)
        b=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('manifest.json');info.size=len(b);info.mode=0o644;tar.addfile(info,io.BytesIO(b))
    receipt=dict(archive=out.name,bytes=out.stat().st_size,sha256=sha(out),manifest=manifest,url='https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/'+out.name)
    with (R/'4090-hey-broad-window-census-package-v1.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='manifest'}))
if __name__=='__main__':main()
