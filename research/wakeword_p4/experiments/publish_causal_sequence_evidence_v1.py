"""Lossless compact publication of actual complete-input reports and UINT8 traces."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path('.local/wakeword/causal-sequence-formal-development-v1')
DEST=Path('research/wakeword_p4/results')
PREFIX='4090-causal-sequence-formal'
def digest(data):return hashlib.sha256(data).hexdigest()
def main():
    summary=ROOT/'diagnostic/summary.json';s=json.loads(summary.read_text());assert len(s['points'])==277
    result=json.loads((ROOT/'result.json').read_text());assert result['all_passed'] and len(result['models'])==2
    stored=[];index=[];arrays={};paths=[ROOT/'result.json',summary]
    for model in result['models']:
        name=model['model'];report=ROOT/name/'report.json';assert digest(report.read_bytes())==model['report_sha256'];r=json.loads(report.read_text());paths.append(report);paths.append(ROOT/'diagnostic'/f'{name}.jsonl');chunks=[];offset=0
        for rec in r['records']:
            trace=Path(rec['trace_path']);assert digest(trace.read_bytes())==rec['trace_sha256']
            with np.load(trace) as a:
                raw=a['uint8'].copy();assert raw.dtype==np.uint8 and np.array_equal(raw,a['uint8_repeat'])
            chunks.append(raw);index.append(dict(model=name,phase=rec['phase'],index=rec['index'],offset=offset,calls=len(raw),uint8_sha256=digest(raw.tobytes()),original_trace_sha256=rec['trace_sha256']));offset+=len(raw)
        arrays[name]=np.concatenate(chunks)
    for p in paths:
        label=('result' if p.name=='result.json' else 'summary' if p.name=='summary.json' else p.parent.name+'-original' if p.name=='report.json' else p.stem+'-points')
        out=DEST/f'{PREFIX}-{label}-v1{p.suffix}.gz';original=p.read_bytes();data=gzip.compress(original,mtime=0)
        with out.open('xb') as f:f.write(data)
        stored.append(dict(path=str(out),bytes=len(data),sha256=digest(data),original_path=str(p),original_bytes=len(original),original_sha256=digest(original)))
    p=DEST/f'{PREFIX}-uint8-v1.npz'
    assert not p.exists();np.savez_compressed(p,**arrays);data=p.read_bytes();stored.append(dict(path=str(p),bytes=len(data),sha256=digest(data)))
    p=DEST/f'{PREFIX}-uint8-index-v1.json.gz';original=(json.dumps(index,indent=2)+'\n').encode();data=gzip.compress(original,mtime=0)
    with p.open('xb') as f:f.write(data)
    stored.append(dict(path=str(p),bytes=len(data),sha256=digest(data),original_bytes=len(original),original_sha256=digest(original)))
    # Verify published bytes and re-extracted original JSON before recording
    # receipts. Do not duplicate original PCM/features or full batch tensors.
    for row in stored:
        data=Path(row['path']).read_bytes();assert len(data)==row['bytes'] and digest(data)==row['sha256']
        if row['path'].endswith('.gz'):
            original=gzip.decompress(data);assert len(original)==row['original_bytes'] and digest(original)==row['original_sha256']
            if 'original_path' in row:assert original==Path(row['original_path']).read_bytes()
    with np.load(DEST/f'{PREFIX}-uint8-v1.npz') as packed:
        for row in index:assert digest(packed[row['model']][row['offset']:row['offset']+row['calls']].tobytes())==row['uint8_sha256']
    receipt=dict(schema='causal-sequence-complete-evidence-v1',files=stored,rawcode_traces=len(index),points=277,total_published_bytes=sum(x['bytes'] for x in stored),all_stored_original_bytes_and_reconstructed_codes_verified=True,note='All actual complete rawcodes including startup/tail; original report/point JSON bytes preserved losslessly. No PCM/features/batch duplication. Model v30 remains original frozen bytes.')
    with (DEST/f'{PREFIX}-evidence-manifest-v1.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(published_bytes=receipt['total_published_bytes'],rawcode_traces=len(index),points=277)))
if __name__=='__main__':main()
