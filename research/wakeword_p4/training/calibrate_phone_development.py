"""Select candidate threshold on two synthetic dev protocols and new background."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from evaluate import event_indices,THRESHOLDS


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def trace(path,expected):
    if sha(path)!=expected: raise ValueError('Changed score trace')
    a=np.load(path,allow_pickle=False)
    if a.ndim!=1 or not np.isfinite(a).all() or np.any((a<0)|(a>1)):
        raise ValueError('Invalid smoothed scores')
    return a


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic',type=Path,nargs=2,required=True)
    p.add_argument('--background',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): p.error('Fresh output required')
    expected={'2ff49bdd98b7b8ccaa2d01bef9f8764db331ad2a08c98ace522a05df48763dc8':80,
              '779077bae3a01ae80392962d572b1d6a42a5b58ecd1bb583d23599ea931ba5be':300}
    reports=[json.loads((root/'report.json').read_text()) for root in a.synthetic]
    background=json.loads((a.background/'report.json').read_text())
    models={r['model_sha256'] for r in reports}|{background['model_sha256']}
    if len(models)!=1 or {r['manifest_sha256'] for r in reports}!=set(expected):
        raise ValueError('Need one model and both exact synthetic development protocols')
    clips=[]
    for root,r in zip(a.synthetic,reports):
        if r['final_data_used'] or len(r['records'])!=expected[r['manifest_sha256']]:
            raise ValueError('Wrong synthetic scope/count')
        for i,item in enumerate(r['records']):
            row=item['record'];scores=trace(root/f'{i}-scores.npy',item['nyx']['score_sha256'])
            if row['split']!='validation' or row['role']!='development-paired-validation':
                raise ValueError('Nondevelopment clip')
            clips.append((r['manifest_sha256'],row,scores))
    if (background['feature_index_sha256']!='11cdc3e0680cf920e01809e5bab55e5590a7308380e36c762267e7632ac123de'
            or background['independent_final_scored'] or background['sliding_window']!=5
            or background['split']!='validation' or len(background['speakers'])!=87):
        raise ValueError('Wrong background scope/index')
    ambient=[trace(a.background/(r['speaker']+'-scores.npy'),r['score_sha256']) for r in background['speakers']]
    hours=background['source_hours'];assert abs(hours-8.173672725694447)<1e-10
    def metrics(threshold):
        events=sum(len(event_indices(scores,threshold)) for scores in ambient)
        upper=float(.5*chi2.ppf(.95,2*(events+1))/hours)
        groups={key:dict(positives=0,triggered=0,negative_clips=0,false_activation_clips=0,pre_generation_positive_events=0) for key in expected}
        for key,row,scores in clips:
            indices=event_indices(scores,threshold)
            times=np.asarray([(3*(i+4)+5)*.01 for i in indices])
            g=groups[key]
            if row['target']=='hey_nyx':
                g['positives']+=1
                # Declared synthetic presentation interval, not phonetic validity.
                lead=row['lead_samples']/16000;end=row['keyword_end']/16000
                g['triggered']+=int(np.any((times>=lead)&(times<=end+.8)))
                g['pre_generation_positive_events']+=int(np.count_nonzero(times<lead))
            else:
                g['negative_clips']+=1;g['false_activation_clips']+=int(bool(indices))
        macro=float(np.mean([g['triggered']/g['positives'] for g in groups.values()]))
        eligible=(upper<=.5 and macro>=.48125 and all(g['false_activation_clips']==0 and g['pre_generation_positive_events']==0 for g in groups.values()))
        return dict(threshold=threshold,synthetic_macro_clip_trigger=macro,groups=groups,
            background_events=events,background_hours=hours,background_upper_95=upper,eligible=eligible)
    grid=sorted(set(THRESHOLDS+[.5+i/256 for i in range(128)]))
    values=[metrics(t) for t in grid];eligible=[r for r in values if r['eligible']]
    selected=max(eligible,key=lambda r:(r['synthetic_macro_clip_trigger'],-r['background_upper_95'],r['threshold'])) if eligible else None
    report=dict(role='New synthetic/background development calibration only, not final acceptance',
        model_sha256=next(iter(models)),sliding_window=5,threshold=selected['threshold'] if selected else None,
        selected=selected,viable_operating_point=bool(selected),grid=values,validation_only=True,
        source_sha256=sha(Path(__file__)),detector_source_sha256=sha(Path(__file__).with_name('evaluate.py')),
        synthetic_report_sha256=[sha(root/'report.json') for root in a.synthetic],background_report_sha256=sha(a.background/'report.json'),
        final_data_used=False,
        selection='Fixed legacy+1/256 grid; background upper95<=.5/h; zero synthetic negative clips and pre-generation events; macro clip trigger>=frozen-v30 .48125. Max macro, then lower background upper, then higher threshold. No fallback if none.',
        limits='Presentation interval lead..generation-end+800ms, not a verified complete-phoneme detection window. Change from narrow generation-end diagnostic is declared before candidate scores, motivated by known tail bias. Voices/renders correlated; no independent final acceptance or matched population FAR claim.')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'selected':selected,'viable_operating_point':bool(selected)}),flush=True)


if __name__=='__main__': main()
