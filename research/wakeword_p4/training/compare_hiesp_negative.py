"""Audit same-source PCM negative events across Hi ESP and prior Nyx reports."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.stats import chi2
from evaluate import event_indices


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','features','hiesp-report','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--nyx-report',type=Path,action='append',required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Use fresh output')
    manifest=json.loads(a.manifest.read_text()); hiesp=json.loads(a.hiesp_report.read_text())
    if (not hiesp['development_negative_validation'] or hiesp['manifest_sha256']!=sha(a.manifest)
            or hiesp['firmware_stage']!=104 or not hiesp['firmware_crc32_matched']):
        raise ValueError('Hi ESP completion/PCM evidence missing')
    if len(manifest['records'])!=1:
        raise ValueError('This comparison requires one continuous validation recording')
    row=manifest['records'][0]
    cache=json.loads((a.features/'cache.json').read_text())
    index=sha(a.features/'human.json')
    if sha(a.features/'cache.json')!=manifest['source_cache_sha256'] or index!=cache['index_sha256']:
        raise ValueError('Feature cache differs from audited replay source')
    f=[r for r in json.loads((a.features/'human.json').read_text()) if r['speaker']==row['speaker']]
    if len(f)!=1 or f[0]['split']!='validation' or row['split']!='validation':
        raise ValueError('Only exact existing validation session allowed')
    f=f[0];pcm,rate=sf.read(a.manifest.parent/row['path'],dtype='int16')
    if rate!=16000 or pcm.ndim!=1 or len(pcm)!=row['samples']:
        raise ValueError('Invalid source PCM')
    raw=pcm.astype('<i2').tobytes()
    if hashlib.sha256(raw).hexdigest()!=hiesp['pcm_sha256'] or hiesp['record']!=row:
        raise ValueError('Hi ESP consumed different original PCM')
    padded=np.pad(pcm,(0,(-len(pcm))%160)).astype('<i2').tobytes()
    if hashlib.sha256(padded).hexdigest()!=f['stream_pcm_sha256']:
        raise ValueError('Existing Nyx frontend source differs from original PCM plus final alignment')
    if sha(a.features/f['path'])!=f['feature_sha256']:
        raise ValueError('Changed Nyx feature stream')
    hours=len(pcm)/16000/3600
    if abs(hours-f['source_seconds']/3600)>1e-12 or abs(hours-hiesp['original_seconds']/3600)>1e-12:
        raise ValueError('Different original exposure')
    def metrics(count):
        return dict(events=count,source_hours=hours,false_per_hour=count/hours,
                    false_per_hour_upper_95=float(.5*chi2.ppf(.95,2*(count+1))/hours))
    models=[]
    for path in a.nyx_report:
        r=json.loads(path.read_text())
        if (r['feature_index_sha256']!=index or r.get('split','validation')!='validation' or 'validation' not in r['role']
                or r.get('independent_final_scored') or r.get('development_regression_scored')
                or r.get('testing_or_independent_final_scored')):
            raise ValueError('Nyx report is not matching original validation')
        if len(r['speakers'])!=1 or r['speakers'][0]['features_sha256']!=f['feature_sha256']:
            raise ValueError('Nyx feature provenance differs')
        trace=path.parent/(r['speakers'][0]['speaker']+'-scores.npy')
        if sha(trace)!=r['speakers'][0]['score_sha256']:
            raise ValueError('Changed score trace')
        scores=np.load(trace,allow_pickle=False)
        if len(event_indices(scores,r['threshold']))!=r['false_events']:
            raise ValueError('Stored event count not reproduced')
        count=r['false_events']
        if count!=r['speakers'][0]['false_events'] or abs(hours-r['source_hours'])>1e-12:
            raise ValueError('Nyx counts/exposure mismatch')
        models.append(dict(model_sha256=r['model_sha256'],threshold=r['threshold'],
                           report_sha256=sha(path),**metrics(count)))
    result=dict(role='Same-original-PCM continuous development negative comparison, not acceptance',
                source_sha256=sha(Path(__file__)),manifest_sha256=sha(a.manifest),
                hiesp_report_sha256=sha(a.hiesp_report),original_pcm_sha256=hiesp['pcm_sha256'],
                nyx_frontend_pcm_sha256=f['stream_pcm_sha256'],same_original_pcm_verified=True,
                nyx_final_alignment_samples=(-len(pcm))%160,
                hiesp_final_padding_samples=hiesp['final_chunk_zero_pad_samples'],
                hiesp=dict(threshold_requested=hiesp['threshold_requested'],
                           threshold_getter=hiesp['threshold_getter_after_set'],**metrics(len(hiesp['events']))),
                nyx=models,independent_final_audio_accessed=False,
                limits='One reused45-minute development session; not enough exposure for0.5/h upper bound, nor a positive recall comparison. Native engine trigger policies retained. Thresholds not selected to match general false-wake rate. Zero observed events do not prove equal quality.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
