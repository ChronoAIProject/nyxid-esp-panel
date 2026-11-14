"""Independent complete saved render/feature traversal; no TTS/model calls."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import soundfile as sf

R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def render():
    reportpath=R/'4090-hey-common-carrier-render-result-v1.json'
    if not reportpath.exists():print(json.dumps(dict(status='not_ready',missing=str(reportpath))));return 2
    report=json.loads(reportpath.read_text());planpath=R/'4090-hey-common-carrier-render-execution-v1.json';p=json.loads(planpath.read_text());assert report['execution_sha256']==sha(planpath)
    records=BASE/'hey-common-carrier-rendered-v1/records.jsonl';assert sha(records)==report['records_sha256'];bad=[];count=0;maximum=0.
    for row in map(json.loads,records.read_text().splitlines()):
        index=count//3;variant=count%3;source=p['requests'][index];native=source['native'];req=native['request'];assert row['order']==count and row['native_order']==index and row['variant']==variant and row['request']==req
        assert row['seed']==source['render_seeds'][variant] and row['anchor_id']==source['anchor']['id'] and row['generated_prefix_end_sample'] is None
        f=native['files']['wav'];assert sha(f['path'])==f['sha256'];pcm,sr=sf.read(f['path'],dtype='int16');assert sr==16000 and len(pcm)==native['samples']
        dry=pcm.astype(np.float64)/32768
        if variant==2:
            speech=np.zeros(len(dry)+960)
            for delay,amp in zip(p['rir_delays'],p['rir_amplitudes'],strict=True):speech[delay:delay+len(dry)]+=amp*dry
        else:speech=dry
        values=np.pad(speech,(48000,16000))
        if variant:
            noise=np.random.default_rng(source['render_seeds'][variant]).standard_normal(len(values));noise*=np.linalg.norm(dry)/np.sqrt(len(dry))*10**(-(25 if variant==1 else 15)/20)/(np.linalg.norm(noise)/np.sqrt(len(noise)));values+=noise
        gain=min(1.,.95/max(float(np.max(np.abs(values))),1e-8)) if variant else 1.;values*=gain
        wav=Path(row['path']);assert sha(wav)==row['sha256'] and wav.stat().st_size==row['bytes'];actual,rate=sf.read(wav,dtype='int16')
        assert hashlib.sha256(actual.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
        error=float(np.max(np.abs(actual.astype(np.float64)/32768-values))) if len(actual)==len(values) else float('inf');maximum=max(maximum,error)
        good=(actual.ndim==1 and rate==16000 and sf.info(wav).subtype=='PCM_16' and len(actual)==row['samples']==len(pcm)+64000+(960 if variant==2 else 0) and
              np.isfinite(values).all() and np.all(np.abs(values)<1) and not np.any((actual==-32768)|(actual==32767)) and error<=1/32768+1e-7 and abs(gain-row['gain_after_mix'])<=1e-12 and
              row['lead_samples']==48000 and row['tail_samples']==16000 and row['rir_tail_samples']==(960 if variant==2 else 0) and row['generated_container_end_sample']==48000+len(speech))
        if variant==0:good=good and np.array_equal(actual[48000:48000+len(pcm)],pcm) and not np.any(actual[:48000]) and not np.any(actual[-16000:])
        assert bool(good)==row['quality_pass']
        if not good:bad.append(count)
        count+=1
    assert count==4860==report['cases'] and report['quality_pass']==count-len(bad) and report['whole_pack_qualified']==(not bad)
    result=dict(original_result_sha256=sha(reportpath),source_sha256=sha(__file__),cases=count,independently_qualified=not bad,failures=bad,maximum_pcm_error=maximum,
                limits='Full sparse4tap render reconstruction vs originalFFT; common NumPy noiseRNG, no TTS/wake inference or phonetic truth.')
    with (R/'4090-hey-common-carrier-render-independent-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    return 0

def features():
    reportpath=R/'4090-hey-common-carrier-feature-result-v1.json'
    if not reportpath.exists():print(json.dumps(dict(status='not_ready',missing=str(reportpath))));return 2
    from pymicro_features import MicroFrontend
    report=json.loads(reportpath.read_text());planpath=R/'4090-hey-common-carrier-feature-execution-v1.json';assert sha(planpath)==report['execution_sha256']
    root=BASE/'hey-common-carrier-features-v1';assert sha(root/'features.bin')==report['binary_sha256'] and sha(root/'features.json')==report['metadata_sha256']
    raw=np.memmap(root/'features.bin',mode='r',dtype='<u2').reshape(-1,40);assert raw.nbytes==report['binary_bytes'] and len(raw)==report['frames']
    rows=json.loads((root/'features.json').read_text());records=[json.loads(l) for l in (BASE/'hey-common-carrier-rendered-v1/records.jsonl').read_text().splitlines()];offset=0;seen=set()
    for i,(row,source) in enumerate(zip(rows,records,strict=True)):
        assert row['order']==source['order']==i and row['offset']==offset and row['label']==0 and row['split']=='training' and row['target_id']=='common_negative' and row['engine']=='kokoro'
        assert row['anchor_id']==source['anchor_id'] and row['carrier']==source['request']['carrier'] and row['variant']==source['variant'] and row['text']==source['request']['text']
        assert row['conditioning_id']==source['request']['conditioning_id'] and row['speed']==source['request']['speed'] and row['generated_prefix_end_sample'] is None and row['keyword_frame'] is None
        key=row['anchor_id'],row['carrier'],row['variant'];assert key not in seen;seen.add(key)
        assert sha(row['path'])==row['sha256']==source['sha256'];pcm,rate=sf.read(row['path'],dtype='int16');assert rate==16000 and len(pcm)==row['samples']==source['samples']
        assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
        frontend=MicroFrontend();padded=np.pad(pcm,(0,(-len(pcm))%160));values=[]
        for start in range(0,len(padded),160):
            frame=frontend.process_samples(padded[start:start+160].astype('<i2').tobytes());assert frame.samples_read==160
            if frame.features:values.append(np.rint(np.asarray(frame.features)*25.6).astype('<u2'))
        expected=np.asarray(values,dtype='<u2').reshape(-1,40)
        assert len(expected)==row['frames']==(len(pcm)+159)//160-2 and len(expected)>=225 and row['lead_frame']==298
        assert np.array_equal(raw[offset:offset+len(expected)],expected) and hashlib.sha256(expected.tobytes()).hexdigest()==row['raw_sha256'];offset+=len(expected)
        if (i+1)%100==0:print(json.dumps(dict(frontend_audited=i+1,total=4860)),flush=True)
    assert len(rows)==4860 and offset==len(raw)
    result=dict(original_result_sha256=sha(reportpath),source_sha256=sha(__file__),rows=4860,all_rows_independently_recomputed=True,all_offsets_labels_cells_exact=True,all_passed=True,
                limits='Complete separate Python frontend traversal with same pinned C++ frontend library; not independent frontend implementation, acoustic truth or recognition. No model training/fresh split.')
    with (R/'4090-hey-common-carrier-feature-independent-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    return 0
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['render','features']);a=ap.parse_args();raise SystemExit(render() if a.stage=='render' else features())
