"""Independent saved-array native-quality audit, no model execution."""
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    results=Path('research/wakeword_p4/results')
    report_path=results/'4090-jenny-development-native-generation-v1.json';r=json.loads(report_path.read_text())
    records=Path(r['records_path']);assert sha(records)==r['records_sha256']
    plan_path=results/'4090-jenny-development-generation-plan-v1.json';assert sha(plan_path)==r['plan_sha256']
    material_path=results/'4090-jenny-development-material-plan-v1.json';material=json.loads(material_path.read_text())
    rows=[json.loads(x) for x in records.read_text().splitlines()];assert len(rows)==132
    maximum=0.;files=0
    for i,row in enumerate(rows):
        request=material['requests'][i];assert row['request']==request and row['metadata']==material['texts'][request['text']]
        for f in row['files'].values():
            path=Path(f['path']);assert sha(path)==f['sha256'] and path.stat().st_size==f['bytes'];files+=1
        a={k:np.load(row['files'][k]['path'],allow_pickle=False) for k in ('native','resampled','durations','tokens','scales')}
        assert np.array_equal(a['tokens'],np.asarray([row['metadata']['token_ids']],np.int64))
        assert np.array_equal(a['scales'],np.asarray([request['noise_scale'],request['length_scale'],request['noise_scale_w']],np.float32))
        assert a['native'].ndim==a['resampled'].ndim==a['durations'].ndim==1
        assert all(np.isfinite(v).all() for v in a.values())
        assert np.array_equal(resample_poly(a['native'],320,441),a['resampled'])
        assert len(a['resampled'])==(len(a['native'])*320+440)//441
        assert len(a['durations'])==a['tokens'].shape[1]
        assert np.array_equal(a['durations'],np.rint(a['durations'])) and np.all(a['durations']>=0)
        assert float(a['durations'].sum(dtype=np.float64))*256==len(a['native'])
        prefix=int(a['durations'][:row['metadata']['prefix_token_count']].sum(dtype=np.float64))*256
        assert prefix==row['native_prefix_end_sample'] and (prefix*320+440)//441==row['generated_prefix_end_sample']
        assert 0<prefix<=len(a['native'])
        pcm,rate=sf.read(row['files']['wav']['path'],dtype='int16')
        assert rate==16000 and pcm.ndim==1 and len(pcm)==len(a['resampled'])==row['samples']
        assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
        assert sf.info(row['files']['wav']['path']).subtype=='PCM_16'
        assert .2<=len(pcm)/16000<=10 and np.any(pcm)
        assert np.sqrt(np.mean(a['resampled'].astype(np.float64)**2))>=.001
        assert np.all(np.abs(a['native'])<1) and np.all(np.abs(a['resampled'])<1)
        assert not np.any((pcm==-32768)|(pcm==32767))
        error=float(np.max(np.abs(pcm.astype(np.float64)/32768-a['resampled'].astype(np.float64))))
        assert error<=1/32768+1e-7;maximum=max(maximum,error)
        assert row['quality_pass'] and all(row['checks'].values())
    result=dict(schema='jenny-development-independent-native-audit-v1',script_sha256=sha(Path(__file__)),
        report_sha256=sha(report_path),records_sha256=sha(records),rows=132,files_verified=files,
        full_resampled_arrays_exact=True,actual_token_duration_identities_exact=True,maximum_pcm_serialization_error=maximum,
        all_native_process_quality_pass=True,
        limits='Saved132native outputs only, no rerun/model inference, no original-vs-observer numeric equivalence or random-seed certification. Not acoustic/human phoneme truth, source ancestry independence, wake recognition or threshold admission.')
    with (results/'4090-jenny-development-independent-native-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
