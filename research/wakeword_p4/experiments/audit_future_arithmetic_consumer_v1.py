"""Analytic contract fixture against isolated Mac consumer; no candidate scoring."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(b):return hashlib.sha256(b).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--consumer-root',type=Path,required=True)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    source=a.consumer_root/'research/panel_wake_runtime/prepare_runtime.py'
    spec=importlib.util.spec_from_file_location('isolated_consumer',source)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    training=a.training.read_bytes()
    assert sha(training)=='41d3a232920143d6bfa39b74fc1068f2675615f2df23ed599155b58a085312f9'
    profile=json.loads((source.parent/'plans/gpu-v30-runtime-v1.json').read_text())
    profile.update(calibration_schema='joint-development-v1',event_arithmetic='uint8-sum-binary64-v1')
    fixture=dict(model_sha256=profile['model_sha256'],threshold=profile['threshold'],
        selected=dict(eligible=True,threshold=profile['threshold']),final_data_used=False,
        sliding_window=5,validation_only=True,viable_operating_point=True,
        event_arithmetic='uint8-sum-binary64-v1')
    cases=[('matched',{}, {},True),('profile missing',{'event_arithmetic':None},{},False),
        ('calibration missing',{}, {'event_arithmetic':None},False),
        ('unknown',{'event_arithmetic':'unknown'},{'event_arithmetic':'unknown'},False),
        ('wrong window',{'smoothing_steps':3},{},False),
        ('old schema',{'calibration_schema':'expanded-english-v1'},{},False),
        ('null selected',{}, {'selected':None},False),
        ('final used',{}, {'final_data_used':True},False)]
    checks=[]
    for name,pm,cm,expected in cases:
        prof={**profile,**pm};cal={**fixture,**cm}
        if name=='profile missing':prof.pop('event_arithmetic')
        if name=='calibration missing':cal.pop('event_arithmetic')
        blob=json.dumps(cal).encode();prof.update(training_report_sha256=sha(training),calibration_sha256=sha(blob))
        try:m.validate_gpu_reports(prof,training,blob);accepted=True
        except (ValueError,KeyError):accepted=False
        assert accepted==expected,name
        checks.append(dict(case=name,accepted=accepted))
    result=dict(schema='gpu-future-arithmetic-consumer-audit-v1',consumer_sha256=sha(source.read_bytes()),
        consumer_commit='3111a576d10a35d9ec3402225c796711d8455071',checks=checks,
        actual_candidate_scored=False,actual_candidate_packaged=False,
        limits='Analytic selected metadata fixture with authenticated original v30 training; not candidate eligibility, generated header or P4 verification.')
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
