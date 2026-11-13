"""Transport all retained Joe tensors, without TTS, gain, selection or resynthesis."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from synthetic_output_quality_v1 import audit_output

SOURCE = Path('.local/wakeword/joe-development-native-v1')
OUT = Path('.local/wakeword/joe-development-native-transport-v1')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    plan_path = SOURCE/'plan.json'
    plan = json.loads(plan_path.read_text())
    expected = [r['id'] for r in plan['requests'] if r['voice']=='joe']
    source_rows = [json.loads(line) for line in (SOURCE/'records.jsonl').read_text().splitlines()]
    assert [r['id'] for r in source_rows] == expected
    assert sha(SOURCE/'records.jsonl') == '91b310721d53642bea56fe838d57040785681ced5dce5b0b2f17b6c45a1a7e03'
    assert sha(plan_path) == 'd69dbc7486cb974641ae9849e33c965e98998a79ce55c95b8a334d4d4f71c03a'
    OUT.mkdir(exist_ok=False)
    rows=[]
    with (OUT/'records.jsonl').open('x') as log:
        for source in source_rows:
            row=dict(source)
            row['original_generator_error']=row.pop('error',None)
            row['source_engine_tensor_path']=str(SOURCE/source['engine_tensor_path'])
            try:
                tensor_path=SOURCE/source['engine_tensor_path']
                assert sha(tensor_path)==source['engine_tensor_sha256']
                tensor=np.load(tensor_path,allow_pickle=False)
                assert tensor.dtype==np.float32 and tensor.ndim==4 and tensor.shape[:3]==(1,1,1)
                native=tensor[0,0,0,:]
                assert native.tobytes()==tensor.tobytes()
                raw=OUT/(row['id']+'.npy');np.save(raw,native,allow_pickle=False)
                row.update(raw_path=raw.name,raw_sha256=sha(raw),native_samples=len(native),
                           singleton_removal_sample_bytes_exact=True)
                divisor=math.gcd(row['native_rate'],16000)
                assert np.isfinite(native).all()
                pcm=resample_poly(native,16000//divisor,row['native_rate']//divisor)
                wav=OUT/(row['id']+'.wav');sf.write(wav,pcm,16000,subtype='PCM_16')
                row.update(path=wav.name,sha256=sha(wav),samples=len(pcm),generated_reference_end_sample=len(pcm))
                row['audit']=audit_output(OUT,row)
            except Exception as exc:
                row['error']=f'{type(exc).__name__}: {exc}'
            rows.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
    report=dict(source_plan_sha256=sha(plan_path),source_records_sha256=sha(SOURCE/'records.jsonl'),
                source_summary_sha256=sha(SOURCE/'summary.json'),adapter_sha256=sha(__file__),
                requested=len(expected),recorded=len(rows),
                passed=sum(r.get('audit',{}).get('passed',False) for r in rows),
                mechanical_pack_qualified=all(r.get('audit',{}).get('passed',False) for r in rows) and len(rows)==120,
                records_sha256=sha(OUT/'records.jsonl'),
                new_tts_calls=0,mike_inference=False,wake_model_scoring=False,
                original_generation_failed=True,
                limits='Retained full tensors only; no phonetic truth, family independence or rendering qualification.')
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
