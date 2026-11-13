"""Register Qwen standalone wake supervision without fabricated command alignment."""
import hashlib
import json
from pathlib import Path

RESULTS=Path('research/wakeword_p4/results')
OUT=RESULTS/'4090-hey-broad-qwen-standalone-material-plan-v1.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not OUT.exists()
    auxpath=RESULTS/'4090-hey-broad-aux-material-plan-v1.json';aux=json.loads(auxpath.read_text())
    negativepath=RESULTS/'4090-hey-broad-common-negative-plan-v1.json';neg=json.loads(negativepath.read_text())
    review=RESULTS/'4090-hey-broad-qwen-timing-api-review-v1.json'
    requests=[]
    for identity,description in sorted(aux['qwen']['training_conditions'].items()):
        for draw in range(3):
            for target,text in [('hey_willow','Hey Willow.'),('hey_nyx','Hey Nicks.')]:
                requests.append(dict(order=len(requests),split='training',engine='qwen-cuda',role='own-target-positive',
                    conditioning_id=identity,description=description,target=target,text=text,draw=draw,seed=597726+len(requests),
                    generated_reference='Entire original standalone waveform end, no command or phonetic-boundary claim.'))
    path=Path(neg['schedule']['path']);assert sha(path)==neg['schedule']['sha256']
    for r in map(json.loads,path.read_text().splitlines()):
        if r['engine']=='qwen-cuda':
            requests.append(dict(order=len(requests),split='training',engine='qwen-cuda',role='common-negative',
                conditioning_id=r['conditioning_id'],description=r['description'],text=r['text'],seed=r['seed'],source_negative_order=r['order']))
    assert len(requests)==360
    d=dict(schema='hey-broad-qwen-standalone-material-plan-v1',source_sha256=sha(Path(__file__)),
        input_hashes={str(p):sha(p) for p in [auxpath,negativepath,review]},requests=requests,
        positive_per_arm=72,common_negatives=216,
        amendment='Before any Qwen inference, supersedes unexecuted auxiliary Qwen192 command matrix for this experiment: use24 original training conditions x3 standalone draws x2targets. Original matrix preserved unmodified; command timing unsupported by inspected API, so do not fabricate endpoints. Piper/Kokoro retain full4carrier coverage.',
        causal_scope='Mixed-engine broader recipe, not isolated voice-count intervention. Qwen contributes standalone voice/condition diversity; Piper/Kokoro provide command contexts.',
        original_qwen=aux['qwen'],
        generation_requirements=['Pin actual generator/source/runtime/available IDs before inference.',
            'Retain actual text/instruction IDs, codec codes, full native/resampled floats and PCM; phoneme/text-aligned durations explicitly unavailable.',
            'One attempt per request, no retries/trim/normalization or ASR selection; persist all failures.',
            'Reject codec cap saturation or incomplete output under predefined mechanical checks; do not silently resample/reseed.',
            'No command-derived prefix-negative crops from Qwen; prefix-negative handling must be explicit in training lock.'],
        heldout_used=False,training_started=False,
        limits='Standalone full synthesis endpoint is generated reference, not acoustic end-phone. Conditions/seeds are not independent people. No candidate score informed amendment.')
    OUT.write_text(json.dumps(d,indent=2)+'\n');print('360 exact Qwen requests registered, no inference')
if __name__=='__main__':main()
