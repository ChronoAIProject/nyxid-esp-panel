"""Paired gain-only development diagnostic, fixed thresholds and all phrases."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from evaluate import Detector,smooth,event_indices
from features import features


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','v30-model','v35-model','v30-report','v35-report','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    manifest=a.dataset/'manifest.jsonl'
    assert sha(manifest)=='dcf1bbd2ee183bd1bdb893f4662e94d74cc6554b2cc15651834bdc5e59f030da'
    source=[json.loads(x) for x in manifest.read_text().splitlines()]
    baselines={name:json.loads(getattr(a,name+'_report').read_text()) for name in ('v30','v35')}
    thresholds={name:r.get('threshold',r.get('nyx_threshold')) for name,r in baselines.items()}
    expected={name:{r['record']['path']:r['nyx']['event_seconds'] for r in b['records']} for name,b in baselines.items()}
    for r in source:assert sha(a.dataset/r['path'])==r['sha256']
    assert max(r['peak'] for r in source)*1.5<1
    detectors={name:Detector(getattr(a,name+'_model')) for name in baselines}
    for name,b in baselines.items():
        assert sha(getattr(a,name+'_model'))==b.get('model_sha256',b.get('nyx_model_sha256'))
    results=[]
    for i,r in enumerate(source):
        pcm,rate=sf.read(a.dataset/r['path'],dtype='int16');assert rate==16000 and pcm.ndim==1
        for gain in (.5,1.,1.5):
            changed=np.rint(pcm.astype(np.float64)*gain)
            assert np.max(changed)<32767 and np.min(changed)>-32768
            changed=changed.astype(np.int16)
            f=features(np.pad(changed,(28800,16000)))
            for name,detector in detectors.items():
                detector.reset();scores=smooth(detector.predict(f),5)
                times=[float((3*(j+4)+5)*.01) for j in event_indices(scores,thresholds[name])]
                if gain==1.:assert times==expected[name][r['path']]
                path=a.output/f'{name}-{i}-g{gain}.npy';np.save(path,scores)
                results.append(dict(model=name,source=r['path'],target=r['target'],voice=r['voice'],gain=gain,
                    source_sha256=r['sha256'],scaled_pcm_sha256=hashlib.sha256(changed.astype('<i2').tobytes()).hexdigest(),
                    clipped_samples=0,event_seconds=times,max_score=float(scores.max()),score_path=path.name,score_sha256=sha(path)))
        if (i+1)%50==0:print(json.dumps({'sources_completed':i+1,'planned':300}),flush=True)
    summaries={}
    for name in baselines:
        summaries[name]={}
        for gain in (.5,1.,1.5):
            group=[r for r in results if r['model']==name and r['gain']==gain]
            pos=[r for r in group if r['target']=='hey_nyx'];neg=[r for r in group if r['target'] in ('hey_nick','hey_next','hey_mix')]
            summaries[name][str(gain)]=dict(positive_clips=len(pos),positive_any_event_clips=sum(bool(r['event_seconds']) for r in pos),
                common_negative_clips=len(neg),common_false_clips=sum(bool(r['event_seconds']) for r in neg),
                cross_target_false_clips=sum(bool(r['event_seconds']) for r in group if r['target']=='hi_esp'))
    report=dict(role='Fixed gain-only VCTK development diagnostic',manifest_sha256=sha(manifest),thresholds=thresholds,
        model_sha256={name:sha(getattr(a,name+'_model')) for name in baselines},source_sha256=sha(Path(__file__)),
        baseline_report_sha256={name:sha(getattr(a,name+'_report')) for name in baselines},
        gain1_exact_baseline_event_reproduction=True,summaries=summaries,records=results,
        final_data_used=False,threshold_selected=False,limits='All300 development sources, no phrase filtering, fixed gain levels, no clipping; quantized PCM rounding retained. Any-event intention counts, not phonetic recall. No background FAR at changed gain, no preprocessing adoption or model promotion.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summaries),flush=True)


if __name__=='__main__':main()
