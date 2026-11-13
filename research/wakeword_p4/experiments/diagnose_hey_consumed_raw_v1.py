"""Bounded diagnostics of saved consumed streams only; no inference or selection."""
import hashlib,json
from pathlib import Path
import numpy as np

BASE=Path('.local/wakeword')
RESULT=Path('research/wakeword_p4/results/4090-hey-consumed-raw-diagnostic-v1.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def peaks(raw,samples=None):
    assert raw.dtype==np.uint8 and raw.ndim==1 and len(raw)>=5
    sums=np.convolve(raw.astype(np.int32),np.ones(5,dtype=np.int32),'valid')
    raw_step=int(np.argmax(raw))+1;smooth_step=int(np.argmax(sums))+5
    def at(step):
        value=(3*step+2)*160
        return min(value,samples) if samples else value
    return dict(raw_peak=int(raw.max()),raw_peak_first_step=raw_step,raw_peak_first_support_sample=at(raw_step),five_sum_peak=int(sums.max()),five_mean_peak=float(sums.max()/1280),five_peak_first_step=smooth_step,five_peak_first_support_sample=at(smooth_step))
def main():
    records=[];training=[]
    freeze=Path('research/wakeword_p4/results/4090-hey-development-custom-execution-v1.json');f=json.loads(freeze.read_text())
    for model in ('hey_willow','hey_nyx','v30'):
        root=BASE/'hey-development-custom-streams-v1'/model;p=root/'report.json';r=json.loads(p.read_text())
        assert r['execution_sha256']==sha(freeze) and r['model_sha256']==f['models'][model]['sha256']
        for item in r['phases']['synthetic']['records']:
            path=Path(item['trace_path']);assert sha(path)==item['trace_sha256'] and item['reset_repeat_exact']
            row=item['record'];raw=np.load(path,allow_pickle=False);assert len(raw)==((row['samples']+159)//160-2)//3
            records.append(dict(model=model,model_sha256=r['model_sha256'],source_id=row['source_id'],target_id=row['target_id'],literal=row['literal'],variant=row['variant'],carrier=row['carrier'],length_scale=row['length_scale'],trace_sha256=item['trace_sha256'],generated_prefix_end_sample=row['generated_prefix_end_sample'],**peaks(raw,row['samples'])))
    for model,folder in [('hey_willow','hey-willow-formal-stream-check-v1'),('hey_nyx','hey-nyx-formal-stream-check-v1')]:
        root=BASE/folder;r=json.loads((root/'report.json').read_text());archive=root/'smoke-training-waveform-streams.npz'
        assert sha(archive)==r['raw_archive_sha256'] and r['model_sha256']==f['models'][model]['sha256']
        with np.load(archive,allow_pickle=False) as z:
            for i,case in enumerate(r['cases']):
                raw=z[f'case{i}_first'];np.testing.assert_array_equal(raw,z[f'case{i}_repeat'])
                assert hashlib.sha256(raw.tobytes()).hexdigest()==case['raw_uint8_sha256']
                training.append(dict(model=model,model_sha256=r['model_sha256'],case=case,**peaks(raw,case['samples'])))
    summary=[]
    for model in ('hey_willow','hey_nyx','v30'):
        target='hey_willow' if model=='hey_willow' else 'hey_nyx';rows=[r for r in records if r['model']==model and r['target_id']==target];assert len(rows)==36
        summary.append(dict(model=model,target=target,development_positive_clips=36,maximum_raw_peak=max(r['raw_peak'] for r in rows),maximum_five_mean_peak=max(r['five_mean_peak'] for r in rows),median_five_mean_peak=float(np.median([r['five_mean_peak'] for r in rows])),training_positive_saved_cases=[r for r in training if r['model']==model and r['case']['label']==1]))
    result=dict(schema='hey-consumed-raw-diagnostic-v1',source_sha256=sha(__file__),execution_sha256=sha(freeze),records=records,training_cases=training,summary=summary,inference_run=False,threshold_selected=False,acceptance_accessed=False,limits='Saved consumed outputs only. Peaks and support times are diagnostics, not alternative detection/calibration or phonetic truth. Training comparison only2previouslycertified training PCM per model, not whole training pack. No cause isolated from voice/material differences.')
    with RESULT.open('x') as out:out.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary))
if __name__=='__main__':main()
