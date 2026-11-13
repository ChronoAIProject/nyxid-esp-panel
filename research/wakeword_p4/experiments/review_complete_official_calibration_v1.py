"""Constructed native official events/UART/PCM tests; no inference or TTS."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import ModuleType
from unittest.mock import patch
import wave
import zlib

import numpy as np

from review_complete_development_metrics_v1 import fixture


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.modules[name]=module
    return module


def reject(error,call):
    try:
        call()
    except error:
        return
    raise AssertionError('Expected rejection: '+str(error))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu-ref',default='916fed0')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    commit=subprocess.check_output(['git','rev-parse',args.gpu_ref],text=True).strip()
    here=Path(__file__).resolve().parent
    sources={'experiments/calibrate_complete_official_v1.py':(here/'calibrate_complete_official_v1.py').read_bytes()}
    for name in ('experiments/hey_development_metrics_v1.py','experiments/uint8_event_policy_v1.py',
                 'qemu_hey_reference_v2/calibrate_development.py','qemu_hey_reference_v2/run_development_grid.py'):
        sources[name]=subprocess.check_output(['git','show',commit+':research/wakeword_p4/'+name])
    checks=[]
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);exp=root/'experiments';old=root/'qemu_hey_reference_v2';exp.mkdir();old.mkdir()
        for name,data in sources.items():
            (root/name).write_bytes(data)
        sys.path[:0]=[str(exp),str(old)]
        module=load('new_official',exp/'calibrate_complete_official_v1.py')
        # The pinned module imports soundfile for its old main, but neither
        # metrics nor audit_stream uses it. Forbid audio-library IO explicitly;
        # the numerical PCM decoder test below executes real stdlib wave.
        unused_audio=ModuleType('soundfile')
        def forbidden_audio(*args,**kwargs):
            raise AssertionError('Constructed review must not call soundfile')
        unused_audio.SoundFile=forbidden_audio
        with patch.dict(sys.modules,{'soundfile':unused_audio}):
            legacy=load('old_official',old/'calibrate_development.py')
        rows,_=fixture()
        for model,target,common,cross in [('willow','hey_willow',324,36),('hiesp','hi_esp',288,72)]:
            clips=[(r,[dict(call=75,sample_end=38400,result=1)] if r['target_id']==target else []) for r in rows]
            result=module.metrics(clips,model,rows,['bare'])
            assert result['denominators']==dict(positive=36,standalone=9,command=27,
                conditions={'0':12,'1':12,'2':12},common=common,cross_target=cross)
            assert result['macro']==result['standalone']==result['command']==1
            assert module.eligible(result,.3)
            index=next(i for i,(r,_) in enumerate(clips) if r['target_id']=='hey_muse')
            clips[index][1].append(dict(call=75,sample_end=38400,result=1))
            assert not module.eligible(module.metrics(clips,model,rows,['bare']),.3)
            clips[index][1].clear()
            for r,events in clips:
                if r['target_id']==target:
                    events[:]=[dict(call=98,sample_end=50000,result=1)]
            tail=module.metrics(clips,model,rows,['bare'])
            assert tail['macro']==1
            event=next(r['events'][0] for r in tail['records'] if r['target_id']==target)
            assert event['unclamped_support_sample']==50176 and event['generated_prefix_end_sample_signed_ms']==1125
            oldclips=[(dict(r,carrier='' if r['carrier']=='bare' else r['carrier']),e) for r,e in clips if r['target_id']!='hey_muse']
            previous=legacy.metrics(oldclips,model,True)
            current=module.metrics(oldclips,model,[r for r,_ in oldclips],[''])
            for r in current['records']:
                for e in r['events']:
                    e.pop('unclamped_support_sample')
            assert all(current[k]==v for k,v in previous.items())
        checks.append('both396 native targets and exact36/9/27/negative denominators, bare/Muse/tail support, unchanged360 shared legacy fields')
        baseline=[(r,[]) for r in rows]
        reject(ValueError,lambda:module.metrics(baseline[:-1],'willow',rows,['bare']))
        reject(ValueError,lambda:module.metrics(baseline+[baseline[0]],'willow',rows+[rows[0]],['bare']))
        reject(ValueError,lambda:module.metrics(baseline,'willow',rows,['bare','typo']))
        for events in ([dict(call=99,sample_end=50000,result=1)],
                       [dict(call=2,sample_end=1000,result=1)],
                       [dict(call=3,sample_end=1536,result=1),dict(call=2,sample_end=1024,result=1)]):
            bad=copy.deepcopy(baseline);bad[0]=(bad[0][0],events)
            reject(ValueError,lambda:module.metrics(bad,'willow',rows,['bare']))
        checks.append('missing/duplicate source, unknown carrier, out-of-range call, wrong512 clock and reordered native events rejected')
        def point(config,actual,request,recall=1,upper=.3,ok=True):
            return dict(configuration=config,actual_threshold=actual,request=request,
                        synthetic={'macro':recall},background_upper95=upper,eligible=ok)
        points=[point(0,.9,'DEFAULT'),point(1,.8,'0.99'),point(2,.95,'0.4'),point(3,.99,'0.999',ok=False)]
        assert module.choose(points)==points[2]
        tied=[point(1,.9,'0.9'),point(0,.9,'DEFAULT')]
        assert module.choose(tied)==tied[1]
        assert module.choose([point(0,.5,'DEFAULT',upper=.2),point(1,.9,'0.9')])['configuration']==0
        assert module.choose([]) is None and module.choose([points[-1]]) is None
        checks.append('selection uses recall/FAR/actual getter/original order, retains duplicate requests, never requested numeric order; null when none')

        # A tiny real numerical WAV exercises independent stdlib full-PCM CRC
        # admission and the original auditor's complete native return ledger.
        pcm=np.arange(1025,dtype='<i2').tobytes();wav=root/'constructed.wav'
        with wave.open(str(wav),'wb') as out:
            out.setnchannels(1);out.setsampwidth(2);out.setframerate(16000);out.writeframes(pcm)
        row=dict(path=str(wav),sha256=module.sha(wav),pcm_sha256=hashlib.sha256(pcm).hexdigest(),samples=1025)
        assert module.pcm_crc(row)==zlib.crc32(pcm)
        reject(ValueError,lambda:module.pcm_crc(dict(row,samples=1024)))
        reject(ValueError,lambda:module.pcm_crc(dict(row,pcm_sha256='0'*64)))
        checks.append('complete numerical PCM WAV encoded/decoded/sample identity and CRC; truncation/hash mismatch rejected')
        plan=root/'constructed-plan.json';plan.write_text('{}')
        original_plan=legacy.PLAN;legacy.PLAN=plan
        request='0.7';bits=f'{struct.unpack("<I",struct.pack("<f",.7))[0]:08x}'
        actual=struct.unpack('<f',struct.pack('<f',.7))[0]
        prime=[(i,0) for i in range(1,64)]
        returns=[(1,512,0),(2,1024,-1),(3,1025,1)]
        text=(f'HEY_REFERENCE abi=2 mode=SET model=wn9_heywillow_tts request_bits={bits} actual_bits={bits} actual={actual}\n'
            +'NYX_PCM_PARAMETERS chunk=512 rate=16000 channels=1 set_result=1 actual='+str(actual)+'\n'
            +''.join(f'HEY_REFERENCE_PRIME call={c} result={v}\n' for c,v in prime)
            +f'HEY_REFERENCE_AFTER_PRIME actual_bits={bits} actual={actual}\n'
            +''.join(f'HEY_REFERENCE_RETURN call={c} sample_end={s} result={v}\n' for c,s,v in returns)
            +f'NYX_PCM_DONE samples=1025 calls=3 events=1 verified=1 crc32={zlib.crc32(pcm):08x}\n')
        stem='constructed';uart=root/(stem+'-serial.log');uart.write_text(text)
        trace=root/(stem+'-raw-returns.bin');trace.write_bytes(bytes([1,0,2]))
        (root/(stem+'-stage-monitor.txt')).write_text('0x3ffb0000: 0x00000068\n')
        runtime={key:str(i)*64 for i,key in enumerate(('qemu','elf','flash'),1)}
        report=dict(record=row,model='wn9_heywillow_tts',execution_sha256=module.sha(plan),
            serial_sha256=module.sha(uart),pcm_sha256=row['pcm_sha256'],input_samples=1025,input_calls=3,stage=104,
            request_float32_bits=bits,mode='SET',actual_float32_bits=bits,post_prime_getter_bits=bits,
            prime_returns=prime,raw_returns_sha256=module.sha(trace),events=[dict(call=3,sample_end=1025,result=1)],
            input_crc32=f'{zlib.crc32(pcm):08x}',**{key+'_sha256':value for key,value in runtime.items()})
        rp=root/(stem+'-report.json');rp.write_text(json.dumps(report))
        def audit():return legacy.audit_stream(root,stem,row,'willow',request,runtime,zlib.crc32(pcm))
        assert audit()[1]==report['events']
        bad=dict(report,actual_float32_bits='3f000000');rp.write_text(json.dumps(bad))
        reject(AssertionError,audit);rp.write_text(json.dumps(report))
        trace.write_bytes(bytes([1,1,2]));report['raw_returns_sha256']=module.sha(trace);rp.write_text(json.dumps(report))
        reject(AssertionError,audit)
        assert original_plan!=legacy.PLAN and original_plan.name=='4090-hey-official-development-execution-v1.json'
        checks.append('pinned original auditor with isolated new PLAN admits complete prime/getter/UART/raw/stage/CRC and rejects getter or negative-return corruption')
    result=dict(schema='complete-official-calibration-constructed-review-v1',gpu_source_commit=commit,
        source_sha256={k:hashlib.sha256(v).hexdigest() for k,v in sources.items()},
        audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks=checks,all_checks_pass=True,
        actual_wake_model_inference=False,actual_development_or_acceptance_accessed=False,
        unused_soundfile_import_stubbed=True,
        limits='Constructed396/360 native-event rows, points and UART/report/raw bytes; temporary numerical1025sample PCM WAV. Original pinned auditor and actual metric/CRC/selector run, but no QEMU/model/frontend/P4/TTS or actual background. Production full-grid main remains dependent on future real execution/results.')
    with args.output.open('x') as output:
        output.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(checks=len(checks),all_checks_pass=True)))


if __name__=='__main__':main()
