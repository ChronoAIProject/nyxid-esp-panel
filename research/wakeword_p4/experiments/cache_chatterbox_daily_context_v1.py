"""Training daily negatives with old admitted speech prefixes and paired zeros."""
import importlib.metadata,json
from pathlib import Path
import sys
import cache_chatterbox_context_pair_v1 as parent
R=parent.R
ADMISSION=R/'4090-chatterbox-daily-training-admission-v1.json'
PLAN=R/'4090-chatterbox-daily-context-cache-plan-v1.json'
OUT=Path('.local/wakeword/chatterbox-daily-context-cache-v1')
def lock():
    rows=json.loads(ADMISSION.read_text())['admitted'];assert len(rows)==171
    original=parent.ADMISSION;old=json.loads(original.read_text())['admitted']
    paths=[Path(__file__),Path(parent.__file__),ADMISSION,original,Path(__file__).with_name('causal_sequence_supervision_v1.py'),Path('research/wakeword_p4/training/features.py')]
    schedule=[]
    for row in rows:
        assert row['request']['role']=='training-only' and row['request']['requested_label']==0
        wav=Path(row['wav']['path']);assert parent.sha(wav)==row['wav']['sha256'];paths.append(wav)
        pool=sorted([r for r in old if r['request']['voice']==row['request']['voice'] and r['request']['requested_label']==0],key=lambda r:r['index']);assert len(pool)==12
        prefix=pool[row['index']%len(pool)];assert parent.sha(prefix['wav']['path'])==prefix['wav']['sha256'];paths.append(Path(prefix['wav']['path']))
        for gap in [8000,32000]:schedule.append(dict(target=row,prefix=prefix,gap_samples=gap))
    p=dict(files={str(f):parent.sha(f) for f in paths},schedule=schedule,versions={n:importlib.metadata.version(n) for n in ['numpy','soundfile','pymicro-features']},prime=81600,tail=8000,retained_lead_frames=360,max_frames=846,
        policy='All171ASR-admitted provisional daily negatives x2fixed gaps. Same existing48admitted training-negative prefix pool, voice-matched deterministic index selection; speech vs equal-length zeros. Entire PCM frontend then same multiple3crop retaining>=360history frames, whole target+8000tail; original masks/all retained calls negative. New lexical targets differ in duration/phase/prefix-selection distribution from old negatives, not pure lexical-only causal isolation. No crop truncation, voice/dev/acceptance admission, scoring or fit;21quarantined excluded unchanged. Separate sampler audit must preserve all positive and120parent slots and exact four negative presentations, mixed2silence+2speech condition budget.')
    with PLAN.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
def run():
    parent.PLAN=PLAN;parent.OUT=OUT;parent.run()
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
