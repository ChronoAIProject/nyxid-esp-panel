"""Check a completed locked final export using complete training-only PCM.

No event/threshold scoring and no development or acceptance media access.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from smoke_hey_target_adapter_v2 import training_stream_checks


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',choices=('hey_willow','hey_nyx'),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    lock_path=Path('research/wakeword_p4/results/4090-hey-v2-formal-training-lock-v1.json')
    lock=json.loads(lock_path.read_text());arm=lock['arms'][a.target]
    root=Path(arm['output']);report_path=root/'training.json'
    r=json.loads(report_path.read_text());model=root/(a.target+'.tflite')
    assert r['target_id']==a.target and r['steps']==r['selected_step']==20000
    assert r['training_lock_sha256']==sha(lock_path)
    assert r['model_sha256']==sha(model) and model.stat().st_size==54440
    assert r['batch_schedule']['all_actual_batch_tensors_match_precomputed']
    assert r['positive_crop']['all_positive_draws_match_precomputed']
    assert r['batchnorm_initial']==r['batchnorm_selected']
    features=Path(arm['features'])
    for name in ('features.bin','features.json'):
        path=features/name
        assert sha(path)==lock['files'][str(path)]
    a.output.mkdir(parents=True,exist_ok=False)
    result=training_stream_checks(model,features,a.output)
    result.update(schema='hey-formal-training-stream-check-v1',target=a.target,
                  model_sha256=sha(model),model_bytes=model.stat().st_size,
                  original_training_report_sha256=sha(report_path),
                  training_lock_sha256=sha(lock_path),checker_sha256=sha(Path(__file__)),
                  helper_sha256=sha(Path(__file__).with_name('smoke_hey_target_adapter_v2.py')),
                  limits='Two complete training-only PCM cases; INT8 I/O and reset repeatability. Not recognition/FAR, whole-pack accuracy or held-out acceptance.')
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
