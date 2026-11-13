"""Independent saved-array duration/sample arithmetic audit, no model runtime."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('research/wakeword_p4/results')
OUT=Path('.local/wakeword/hey-kokoro-duration-probe-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    report=ROOT/'4090-hey-kokoro-duration-probe-v1.json';d=json.loads(report.read_text())
    records=OUT/'records.jsonl';assert sha(records)==d['records_sha256']
    rows=[json.loads(x) for x in records.read_text().splitlines()];assert rows==d['records'] and len(rows)==24
    receipts=[]
    for r in rows:
        arrays={}
        for name,f in r['files'].items():
            p=OUT/f['path'];assert sha(p)==f['sha256'];a=np.load(p,allow_pickle=False)
            assert list(a.shape)==f['shape'] and str(a.dtype)==f['dtype'];arrays[name]=a
        dur=arrays['duration'].reshape(-1);tokens=arrays['tokens'].reshape(-1);wav=arrays['waveform'].reshape(-1)
        assert len(dur)==len(tokens) and np.isfinite(dur).all() and np.all(dur>=0)
        assert np.array_equal(dur,np.rint(dur))
        samples=sum(int(x) for x in dur)*600
        assert samples==len(wav) and np.isfinite(wav).all() and np.max(np.abs(wav))<1
        assert tokens[0]==tokens[-1]==0 and r['prefix_tokens_exact']
        n=r['prefix_token_count'];assert 0<n<len(tokens)-1
        end=sum(int(x) for x in dur[:1+n])*600
        assert 0<end<=samples
        assert arrays['speed'].dtype==np.float32 and arrays['speed'][0]==np.float32(r['request']['speed'])
        receipts.append(dict(order=r['order'],native_samples=samples,generated_prefix_end_native_sample=end,
            generated_prefix_end_16k_sample=(end*2+2)//3,
            units='600 native samples per duration unit; original model 24kHz',
            includes_initial_padding_allocation=True))
    result=dict(schema='hey-kokoro-duration-array-audit-v1',report_sha256=sha(report),script_sha256=sha(Path(__file__)),
        cases=24,all_file_hashes_and_array_shapes=True,integer_duration_sample_identity=True,receipts=receipts,
        limits='Actual saved24 cases only; native internal generated timing, not acoustic truth or universal graph/runtime proof. No resynthesis, model scoring or heldout access.')
    with (ROOT/'4090-hey-kokoro-duration-array-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print('24/24 original saved waveform/duration/token/style/speed arrays verified')
if __name__=='__main__':main()
