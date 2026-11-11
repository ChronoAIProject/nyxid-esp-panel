"""Frozen v30 training-source diagnostic; no threshold selection or acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from evaluate import Detector, smooth, event_indices
from features import features


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert sha(a.model)=='04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c'
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=a.dataset/'manifest.jsonl'
    rows=[json.loads(x) for x in manifest.read_text().splitlines()]
    assert len(rows)==48 and all(r['role']=='training-only-mechanical-pilot' and r['quality_pass'] for r in rows)
    for r in rows:
        assert sha(a.dataset/r['path'])==r['sha256']
    detector=Detector(a.model);details=[]
    for r in rows:
        pcm,rate=sf.read(a.dataset/r['path'],dtype='int16')
        assert rate==16000 and pcm.ndim==1 and len(pcm)==r['samples']
        # Preserve the full source waveform, add only fixed leading/trailing silence.
        wave=np.pad(pcm,(28800,16000))
        f=features(wave);detector.reset();raw=detector.predict(f)
        detector.reset();np.testing.assert_array_equal(raw,detector.predict(f))
        scores=smooth(raw,5);times=(3*(np.arange(len(scores))+4)+5)*.01
        events=event_indices(scores,.91796875)
        end=(28800+len(pcm))/16000
        details.append(dict(path=r['path'],voice=r['voice'],speed=r['speed'],target=r['target'],
            max_score=float(scores.max()),event_times_seconds=times[events].tolist(),
            generation_end_seconds=end,event_offsets_from_generation_end_ms=((times[events]-end)*1000).tolist(),
            scores=scores.tolist(),times_seconds=times.tolist()))
    groups={}
    for target in sorted({r['target'] for r in details}):
        group=[r for r in details if r['target']==target]
        groups[target]=dict(clips=len(group),clips_with_any_event=sum(bool(r['event_times_seconds']) for r in group),
            total_events=sum(len(r['event_times_seconds']) for r in group),
            max_score_min=min(r['max_score'] for r in group),max_score_max=max(r['max_score'] for r in group))
    detail_path=a.output/'details.json';detail_path.write_text(json.dumps(details)+'\n')
    report=dict(model_sha256=sha(a.model),manifest_sha256=sha(manifest),script_sha256=sha(Path(__file__)),
        details_sha256=sha(detail_path),threshold=.91796875,smoothing_steps=5,
        groups=groups,reset_repeat_exact_clips=len(rows),threshold_selected=False,
        independent_test=False,final_data_used=False,
        limits='Training voices; any-event counts are diagnostics, not verified full-phrase recall. Whole untrimmed generation end is not phonetic end. No FAR bound or Hi ESP comparison; silence padding not exposure evidence.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
