"""Pin old training-only auxiliary identities and exact paired requests, no inference."""
import hashlib
import json
from collections import Counter
from pathlib import Path

BASE=Path('.local/wakeword')
OUT=BASE/'hey-broad-aux-material-plan-v1'
REPORT=Path('research/wakeword_p4/results/4090-hey-broad-aux-material-plan-v1.json')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    assert not OUT.exists() and not REPORT.exists()
    kp=BASE/'hey-nyx/kokoro-speed-v2/generation.json'
    kd=json.loads(kp.read_text())
    voices=sorted(kd['voice_partitions']['training']);assert len(voices)==20
    model=BASE/'kokoro-models/kokoro-v1.0.onnx';bank=BASE/'kokoro-models/voices-v1.0.bin'
    assert sha(model)==kd['model_sha256'] and sha(bank)==kd['voices_sha256']
    qp=BASE/'hey-nyx/qwen-cuda-v1/provenance.json';qd=json.loads(qp.read_text())
    assert qd['source']==dict(repo='Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign',revision='5ecdb67327fd37bb2e042aab12ff7391903235d3')
    for name,expected in qd['model_files'].items():assert sha(BASE/'qwen3-design'/name)==expected
    oldplan=BASE/'hey-nyx/qwen-cuda-v1/plan.json';assert sha(oldplan)==qd['plan_sha256']
    conditions={}
    for r in json.loads(oldplan.read_text()):
        assert r['split']=='training' and len(r['speakers'])==1
        sid=r['speakers'][0]
        if sid in conditions:assert conditions[sid]==r['description']
        conditions[sid]=r['description']
    assert len(conditions)==24
    rows=[]
    for engine,ids,scales in [('kokoro',voices,[.9,1.,1.1]),('qwen-cuda',sorted(conditions),[1.])]:
        for sid in ids:
            for scale in scales:
                for carrier,suffix in [('bare','.'),('light',', turn on the light.'),('time',', what time is it?'),('timer',', set a timer.')]:
                    for target,prefix in [('hey_willow','Hey Willow'),('hey_nyx','Hey Nicks')]:
                        row=dict(order=len(rows),split='training',engine=engine,conditioning_id=sid,
                            target=target,text=prefix+suffix,prefix_text=prefix,carrier=carrier,
                            seed=397726+len(rows),action='generate',
                            reason='Original selected WAV-only sources do not supply retained native float/token provenance for this paired matrix; preserve old data unchanged.')
                        if engine=='kokoro':row.update(speed=scale,lang='en-gb' if sid.startswith('b') else 'en-us',seed_effect='Record runtime RNG support; no claim seed controls ONNX stochastic nodes.')
                        else:row.update(description=conditions[sid],native_speed_control=None)
                        rows.append(row)
    assert len(rows)==672
    OUT.mkdir();p=OUT/'schedule.jsonl';p.write_text(''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in rows))
    result=dict(schema='hey-broad-aux-material-plan-v1',scope='Material subplan only; no synthesis/training',
        planner_sha256=sha(Path(__file__)),schedule=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,
            cells=len(rows),engines=dict(Counter(r['engine'] for r in rows))),
        kokoro=dict(original_generation_sha256=sha(kp),model_sha256=sha(model),bank_sha256=sha(bank),
            training_voices=voices,protected_roles=kd['voice_partitions'],requested_speeds=[.9,1.,1.1],
            runtime='Original kokoro-onnx0.4.9 typed float speed adapter must be pinned and tested; preserve generated phones/tokens if available.'),
        qwen=dict(original_provenance_sha256=sha(qp),original_plan_sha256=sha(oldplan),
            source=qd['source'],model_files=qd['model_files'],generation=qd['generation'],training_conditions=conditions),
        timing='Command wake-prefix timing unavailable until engine-native allocation is certified. Do not infer phone truth from RMS or substitute whole command end as wake-prefix end.',
        required_before_inference=['Pinned runtime/source and generator quality checks for each engine.',
            'Exact common negative material/recipe lock and command-prefix timing admission or separately registered unavailable-timing handling.'],
        limits=['Training conditions/voice presets do not prove independent people or disjoint pretraining.',
            'No old WAV repair/normalization or ASR-filter outcome automatically inherited.',
            'No model inference, media decoding or heldout exposure.'])
    REPORT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['schedule']))

if __name__=='__main__':main()
