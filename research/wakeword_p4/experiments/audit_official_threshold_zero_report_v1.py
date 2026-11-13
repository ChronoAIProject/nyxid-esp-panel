"""Independently check saved Mac native API report; no P4/speech execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib

REQUESTS=[None,.4,.5,.6,.644,.7,.8,.9,.95,.98,.99,.999,.9999,None]


def bits(value):return struct.pack('>f',value).hex()


def audit(path):
    raw=path.read_bytes();report=json.loads(raw)
    assert report['schema']=='official-hey-threshold-native-zero-v1'
    assert report['protocol']['requests']==REQUESTS
    assert list(report['arms'])==['wn9_heywillow_tts','wn9_hiesp']
    crc=zlib.crc32(bytes(16001*2));prime=0;calls=0;setters=0;maps={}
    for engine,arm in report['arms'].items():
        assert len(arm['points'])==14
        points=[]
        for index,(point,request) in enumerate(zip(arm['points'],REQUESTS,strict=True)):
            assert point['point']==index and point['requested_decimal']==request
            init=point['initialization'];ctrl=init['control'];result=point['result']
            assert init['chunk']==512
            assert len(ctrl['prime_returns'])==63 and all(v==0 for v in ctrl['prime_returns'])
            prime+=len(ctrl['prime_returns'])
            for decimal,hexkey in [('default_getter','default_getter_f32_hex'),('getter_after_set','getter_after_set_f32_hex')]:
                value=ctrl[decimal]
                assert math.isfinite(value) and 0<value<1 and bits(value)==ctrl[hexkey]
            assert bits(init['threshold'])==ctrl['getter_after_prime_f32_hex']==ctrl['getter_after_set_f32_hex']
            if request is None:
                assert not ctrl['setter_called'] and ctrl['setter_result'] is None
                assert ctrl['requested_f32_hex'] is None and point['requested_f32_hex'] is None
            else:
                assert .4<=request<=.9999
                assert ctrl['setter_called'] and ctrl['setter_result']==1
                assert ctrl['requested_f32_hex']==point['requested_f32_hex']==bits(request)
                setters+=1
            assert result['complete'] and result['samples']==16001 and result['crc32']==crc
            assert result['calls']==32 and result['no_detect_calls']==32 and result['channel_verified_calls']==0
            assert result['events']==result['events_emitted']==0 and result['offsets_complete']
            assert not result['offsets'] and not point['raw_events']
            calls+=result['calls']
            points.append(dict(configuration='DEFAULT' if request is None else 'SET',request=request,
                               request_f32_hex=point['requested_f32_hex'],actual_getter=init['threshold'],
                               actual_getter_f32_hex=ctrl['getter_after_prime_f32_hex']))
        assert points[0]['actual_getter_f32_hex']==points[-1]['actual_getter_f32_hex']
        assert len({p['actual_getter_f32_hex'] for p in points[:13]})==13
        assert points[0]['actual_getter_f32_hex']!=points[4]['actual_getter_f32_hex'],'DEFAULT != SET .644'
        maps[engine]=points
    assert setters==24 and calls==896 and prime==1764
    assert report['zero_streams_completed']==28 and not report['speech_inference']
    assert not report['thresholds_calibrated'] and not report['candidate_promoted']
    assert not report['reserved_acceptance_accessed']
    return dict(schema='gpu-official-native-zero-saved-report-audit-v1',
                source_commit='aa4f47c917cad3a1a5bad7ad8bf51f37de2e5702',
                source_path='research/wakeword_p4/results/official-hey-threshold-native-zero-v1.json',
                source_sha256=hashlib.sha256(raw).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                streams=28,setter_successes=setters,input_calls=calls,prime_returns=prime,
                expected_zero_pcm_crc32=crc,working_points=maps,
                saved_report_numeric_and_accounting_consistent=True,
                p4_execution_rerun=False,speech_scored=False,threshold_selected=False,
                limits='Saved JSON report checks only; not independent UART/P4 rerun, >64-event coverage, speech/FAR, whole-device or cross-runtime certification. Getter behavior applies only to reported engine/mode/version.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.source)
    with a.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='working_points'}))
