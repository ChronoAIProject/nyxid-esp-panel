"""Verify explicit-phone training output and model-effective whole-source support."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--partitions',type=Path,required=True)
    p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): p.error('Fresh output required')
    manifest=a.dataset/'manifest.jsonl'
    rows=[json.loads(x) for x in manifest.read_text().splitlines()]
    partitions=json.loads(a.partitions.read_text())['voice_partitions']
    assert len(rows)==400 and {r['voice'] for r in rows}==set(partitions['training'])
    assert not {r['voice'] for r in rows}&(set(partitions['validation'])|set(partitions['testing']))
    previous={r['sha256'] for r in map(json.loads,a.pilot.read_text().splitlines())}
    counts=Counter();details=[]
    for r in rows:
        assert r['split']=='training' and r['role']=='training-only-explicit-phone-expansion'
        pcm,rate=sf.read(a.dataset/r['path'],dtype='int16');info=sf.info(a.dataset/r['path'])
        raw=np.load(a.dataset/r['raw_path'],allow_pickle=False)
        assert sha(a.dataset/r['path'])==r['sha256'] and sha(a.dataset/r['raw_path'])==r['raw_sha256']
        assert rate==16000 and pcm.ndim==1 and len(pcm)==r['samples'] and info.subtype=='PCM_16'
        assert raw.ndim==1 and np.isfinite(raw).all() and len(raw)==r['native_samples']
        assert len(raw)==sum(r['native_durations'])*600 and len(r['token_ids'])==len(r['native_durations'])
        assert r['trimmed'] is False
        serialized_ok=not np.any((pcm==-32768)|(pcm==32767))
        counts['quality_pass']+=int(r['quality_pass'] and serialized_ok)
        counts['pilot_duplicate_wavs']+=int(r['sha256'] in previous)
        # teacher60/context150 has 149 used feature frames, covering151 hops.
        # Choose exclusive feature crop end e so last USED feature ends beyond PCM.
        lead=32000
        end=(lead+len(pcm)+159)//160-1
        begin=(end-150)*160;stop=(end+1)*160
        fits=begin<=lead and stop>=lead+len(pcm)
        counts['entire_source_fits']+=int(fits)
        counts[r['target']+'/sources']+=1
        counts[r['target']+'/entire_source_fits']+=int(fits)
        details.append(dict(path=r['path'],samples=len(pcm),target=r['target'],voice=r['voice'],
            speed=r['speed'],entire_source_fits=fits,feature_crop_end=end if fits else None,
            crop_start_sample=begin,crop_stop_sample=stop))
    report=dict(clips=len(rows),counts=dict(counts),unique_wav_hashes=len({r['sha256'] for r in rows}),
        samples_min=min(r['samples'] for r in rows),samples_max=max(r['samples'] for r in rows),
        model_effective_support_samples=24160,context_frames=150,assumed_lead_samples=32000,
        manifest_sha256=sha(manifest),partitions_sha256=sha(a.partitions),pilot_manifest_sha256=sha(a.pilot),
        script_sha256=sha(Path(__file__)),records=details,wake_models_scored=False,final_data_used=False,
        limits='Whole untrimmed PCM includes generated silence. Not fitting does not prove lost wake phonemes, but cannot be called complete-source crop. No trimming or label changes; all failed/non-fitting samples retained. Training sources only.')
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__=='__main__': main()
