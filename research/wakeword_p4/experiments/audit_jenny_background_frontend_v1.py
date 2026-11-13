"""All12 full continuous PCM, alternate IO boundaries; no wake inference."""
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from pymicro_features import MicroFrontend


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    manifest=Path('.local/wakeword/jenny-earnings-development-v1/manifest.json');m=json.loads(manifest.read_text())
    history=Path('research/wakeword_p4/results/4090-jenny-earnings-history-overlap-v1.json');h=json.loads(history.read_text())
    assert h['manifest_sha256']==sha(manifest) and h['no_exact_encoded_or_pcm_hash_matches'] and not h['matches']
    assert sha(h['inventory_path'])==h['inventory_sha256']
    receipts=[]
    for row in m['records']:
        assert sha(row['wav_path'])==row['wav_sha256'] and sha(row['frontend_path'])==row['frontend_sha256']
        cached=np.load(row['frontend_path'],mmap_mode='r',allow_pickle=False)
        assert cached.dtype==np.uint16 and cached.shape==(row['frames'],40)
        frontend=MicroFrontend();carry=np.empty(0,np.int16);index=0;samples=0;pcm_hash=hashlib.sha256()
        def hop(block):
            nonlocal index
            result=frontend.process_samples(block.astype('<i2',copy=False).tobytes())
            assert result.samples_read==160
            if result.features:
                expected=np.rint(np.asarray(result.features)*25.6).astype('<u2')
                assert index<len(cached) and np.array_equal(expected,cached[index]);index+=1
        with sf.SoundFile(row['wav_path']) as source:
            assert source.samplerate==16000 and source.channels==1 and source.subtype=='PCM_16'
            while True:
                chunk=source.read(65536,dtype='int16')
                if len(chunk)==0:break
                samples+=len(chunk);pcm_hash.update(chunk.astype('<i2').tobytes())
                values=np.concatenate((carry,chunk));end=len(values)//160*160
                for at in range(0,end,160):hop(values[at:at+160])
                carry=values[end:].copy()
        padding=(-samples)%160
        if len(carry):hop(np.pad(carry,(0,160-len(carry))))
        assert pcm_hash.hexdigest()==row['pcm_sha256'] and samples==row['samples']
        assert index==len(cached)==(samples+159)//160-2 and padding==row['final_padding_samples']
        receipts.append(dict(source_id=row['source_id'],samples=samples,frames=index,final_padding_samples=padding,all_cached_frames_exact=True))
        print(json.dumps(dict(audited=len(receipts),sources=12)),flush=True)
    result=dict(schema='jenny-background-complete-frontend-audit-v1',script_sha256=sha(__file__),
        manifest_sha256=sha(manifest),history_report_sha256=sha(history),cases=receipts,
        io_samples=65536,all_complete_arrays_exact=True,source_resets=12,
        limits='Independent65536-sample IO/carry traversal versus original160000sample preparation, same pinned underlying MicroFrontend library and formula. No original media redownload, model inference, FAR or universal frontend/pretraining proof.')
    with Path('research/wakeword_p4/results/4090-jenny-background-complete-frontend-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
