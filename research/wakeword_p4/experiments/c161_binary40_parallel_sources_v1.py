"""Original complete source reconstruction for fixed C161/Binary40 parallel replay."""
import hashlib,json
from pathlib import Path
import numpy as np
from replay_binary40_v1 import output as introot,lock as intlock
from replay_c161_binary40_parallel_events_v2 import sha,read
def wave(source,mac=False):
    import soundfile as sf
    path=Path('.local/wakeword/mac-consumed-speech-grid-witness-v1/fixtures')/source['path'] if mac else Path(source.get('wav_path',source.get('path','')))
    expected=source.get('wav_sha256',source.get('sha256'));assert sha(path)==expected
    pcm,rate=sf.read(path,dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==source['samples']
    assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==source['pcm_sha256']
    return pcm

def inputs(c):
    root=introot(c);receipt=read(root/'result.json');p=read(intlock(c))
    if c in ['primed','jenny','joe','melo']:
        rows=[json.loads(s) for s in (root/'binary40/records.jsonl').read_text().splitlines()] if c=='melo' else read(receipt['models'][0]['path'])['records']
        for row in rows:
            s=row['record'];yield row['index'],s,wave(s),Path(row['trace_path']),row['trace_sha256'],row['quantization'],row,None
    elif c=='mac':
        for row in receipt['records']:yield row['index'],row['source'],wave(row['source'],True),root/f'{row["index"]:03d}.npz',row['trace_sha256'],row['quantization'],None,None
    elif c in ['builtin','daily']:
        import soundfile as sf
        for row,s in zip(receipt['records'],p['rows'],strict=True):
            assert row['index']==s['index'];item,=[x for x in s['files'] if x['path'].endswith('.wav')];assert sha(item['path'])==item['sha256']
            pcm,rate=sf.read(item['path'],dtype='int16');assert rate==16000 and pcm.ndim==1
            v=row['models']['binary40'];yield row['index'],s['request'],pcm,root/f'{row["index"]:03d}-binary40.npz',v['trace_sha256'],v['quantization'],None,None
    else:
        import soundfile as sf
        for row in receipt['results']:
            pieces=[];gap=row['gap_samples']
            for i in p['order']:
                item,=[x for x in p['rows'][i]['files'] if x['path'].endswith('.wav')];assert sha(item['path'])==item['sha256']
                pcm,rate=sf.read(item['path'],dtype='int16');assert rate==16000 and pcm.ndim==1;pieces.extend([pcm,np.zeros(gap,np.int16)])
            pieces.append(np.zeros(p['tail_samples'],np.int16));pcm=np.concatenate(pieces)
            combined=np.concatenate((np.zeros(81600,np.int16),pcm));assert hashlib.sha256(combined.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
            v=row['models']['binary40'];yield gap,dict(gap_samples=gap),pcm,Path(v['trace_path']),v['trace_sha256'],v['quantization'],None,row['timeline']
