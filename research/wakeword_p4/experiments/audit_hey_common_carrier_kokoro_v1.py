"""Independent saved native/duration/PCM audit, never synthesizes or scores."""
import hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import onnx
import soundfile as sf
from scipy.signal import resample_poly

R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    result=R/'4090-hey-common-carrier-kokoro-generation-result-v1.json'
    if not result.exists():print(json.dumps(dict(status='not_ready',missing=str(result))));return 2
    report=json.loads(result.read_text());planpath=R/'4090-hey-common-carrier-kokoro-generation-execution-v1.json';plan=json.loads(planpath.read_text())
    assert report['execution_sha256']==sha(planpath)
    for p,h in plan['files'].items():assert sha(p)==h
    original=onnx.load(BASE/'kokoro-models/kokoro-v1.0.onnx');derived=onnx.load(BASE/'hey-common-carrier-kokoro-seeded-v1.onnx')
    assert len(original.graph.node)==len(derived.graph.node)
    changed=[]
    for old,new in zip(original.graph.node,derived.graph.node,strict=True):
        if 'Random' in old.op_type:
            seeds=[a.f for a in new.attribute if a.name=='seed'];assert seeds==[float(plan['derived_random_seed'][len(changed)])]
            changed.append(old.name);new.CopyFrom(old)
    assert len(changed)==2 and derived.SerializeToString()==original.SerializeToString()
    path=BASE/'hey-common-carrier-kokoro-native-v1/records.jsonl';assert sha(path)==report['records_sha256']
    rows=[json.loads(l) for l in path.read_text().splitlines()];assert len(rows)==len(plan['requests'])==1620
    counts=Counter();bad=[];max_error=0.
    for item,row in zip(plan['requests'],rows,strict=True):
        req=item['request'];assert row['request']==req and not row['trimmed'] and not row['normalized']
        assert req['label']==0 and req['split']=='training' and req['target_id']=='common_negative'
        counts[req['conditioning_id'],req['speed'],req['carrier'],req['anchor_text']]+=1
        if 'error' in row:bad.append(dict(order=req['order'],reason=row['error']));continue
        assert row['metadata']==item['metadata']
        for f in row['files'].values():assert Path(f['path']).stat().st_size==f['bytes'] and sha(f['path'])==f['sha256']
        for f in item['anchor_files'].values():assert sha(f['path'])==f['sha256']
        arrays={n:np.load(f['path'],allow_pickle=False) for n,f in row['files'].items() if n!='wav'}
        native=arrays['native'];resampled=arrays['resampled'];dur=arrays['duration'].ravel();tokens=arrays['tokens']
        assert native.dtype==np.float32 and np.array_equal(native,arrays['engine_tensor'].reshape(-1))
        assert tokens.dtype==np.int64 and tokens.tolist()==[item['metadata']['token_ids']]
        assert hashlib.sha256(arrays['style'].tobytes()).hexdigest()==item['metadata']['style_sha256']
        assert np.array_equal(arrays['speed'],np.asarray([req['speed']],dtype=np.float32))
        assert np.array_equal(resample_poly(native,2,3),resampled)
        pcm,rate=sf.read(row['files']['wav']['path'],dtype='int16');info=sf.info(row['files']['wav']['path'])
        assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
        rms=float(np.sqrt(np.mean(resampled.astype(np.float64)**2)));assert rms==row['rms']
        error=float(np.max(np.abs(pcm.astype(np.float64)/32768-resampled.astype(np.float64)))) if len(pcm)==len(resampled) and len(pcm) else float('inf');max_error=max(max_error,error)
        qualified=(all(np.isfinite(a).all() for a in arrays.values()) and pcm.ndim==1 and rate==16000 and info.subtype=='PCM_16' and np.any(pcm) and
                   3200<=len(pcm)<=160000 and len(pcm)==len(resampled)==(len(native)*2+2)//3 and rms>=.001 and
                   np.all(np.abs(native)<1) and np.all(np.abs(resampled)<1) and not np.any((pcm==-32768)|(pcm==32767)) and
                   dur.size==tokens.size and np.all(dur>=0) and np.array_equal(dur,np.rint(dur)) and np.sum(dur,dtype=np.float64)*600==len(native) and error<=1/32768+1e-12)
        assert bool(qualified)==row['quality_pass']
        if not qualified:bad.append(dict(order=req['order'],reason='independent_quality_failure'))
    assert len(counts)==1620 and all(n==1 for n in counts.values())
    assert (not bad)==report['whole_pack_qualified'] and report['quality_pass']==1620-len(bad)
    out=dict(original_result_sha256=sha(result),source_sha256=sha(__file__),cases=1620,independently_qualified=not bad,failures=bad,
             native_resample_pcm_hashes_and_quality_recomputed=True,learned_graph_unchanged_except_two_declared_random_seeds=True,maximum_pcm_serialization_error=max_error,
             synthesis_invocations=0,wake_model_invocations=0,limits='Synthetic pipeline only. No listening/phonetic truth, recognition, freshdevelopment/acceptance or training admission.')
    with (R/'4090-hey-common-carrier-kokoro-independent-audit-v1.json').open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out));return 0
if __name__=='__main__':raise SystemExit(main())
