"""Audit official full input returns and report all registered development points.

Uses original official events, never custom smoothing/refractory/thresholds.
"""
import json
import math
from pathlib import Path
import re
import struct
import sys
import zlib
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from hey_development_metrics_v1 import far_upper
from run_development_grid import PLAN,sha


def metrics(clips,model,details=False):
    target='hey_willow' if model=='willow' else 'hi_esp'
    cross={'hey_nyx'} if model=='willow' else {'hey_nyx','hey_willow'}
    positives=[];early=0;neg={k:dict(clips=0,activated_clips=0,events=0) for k in ('common','cross_target')};records=[]
    for row,events in clips:
        times=[e['sample_end'] for e in events]
        if row['target_id']==target:
            positives.append((row,any(t>=row['generated_prefix_end_sample'] for t in times),any(t>=row['full_rir_prefix_end_sample'] for t in times),any(t>=row['lead_samples'] for t in times)))
            early+=sum(t<row['lead_samples'] for t in times)
        else:
            n=neg['cross_target' if row['target_id'] in cross else 'common'];n['clips']+=1;n['activated_clips']+=bool(times);n['events']+=len(times)
        if details:
            records.append(dict(source_id=row['source_id'],target_id=row['target_id'],variant=row['variant'],carrier=row['carrier'],length_scale=row['length_scale'],events=[dict(e,**{key+'_signed_ms':(e['sample_end']-row[key])/16 for key in ('generated_prefix_end_sample','full_rir_prefix_end_sample','generated_container_end_sample','full_rir_container_end_sample')}) for e in events]))
    assert len(positives)==36
    assert (neg['common']['clips'],neg['cross_target']['clips'])==((288,36) if model=='willow' else (252,72))
    def rate(group,col=1):
        assert group
        return sum(r[col] for r in group)/len(group)
    conditions={str(v):rate([r for r in positives if r[0]['variant']==v]) for v in range(3)}
    result=dict(positive_clips=36,complete_phrase_activated=sum(r[1] for r in positives),macro=sum(conditions.values())/3,conditions=conditions,standalone=rate([r for r in positives if not r[0]['carrier']]),command=rate([r for r in positives if r[0]['carrier']]),full_rir_prefix_recall=rate(positives,2),any_after_source_recall=rate(positives,3),pre_source_positive_events=early,negatives=neg)
    if details:result.update(records=records,speeds={str(s):rate([r for r in positives if r[0]['length_scale']==s]) for s in (.9,1.,1.1)})
    return result


def audit_stream(root,stem,row,model,request,runtime,crc):
    rp=root/(stem+'-report.json');r=json.loads(rp.read_text());uart=root/(stem+'-serial.log');text=uart.read_text()
    assert r['record']==row and r['model']==('wn9_heywillow_tts' if model=='willow' else 'wn9_hiesp')
    assert r['execution_sha256']==sha(PLAN) and r['serial_sha256']==sha(uart)
    for key in ('qemu','elf','flash'):assert r[key+'_sha256']==runtime[key]
    assert r['pcm_sha256']==row['pcm_sha256'] and r['input_samples']==row['samples']
    assert r['input_calls']==(row['samples']+511)//512 and r['stage']==104
    assert re.search(r':\s+0x00000068\b',(root/(stem+'-stage-monitor.txt')).read_text())
    expected=struct.unpack('<I',struct.pack('<f',0 if request=='DEFAULT' else float(request)))[0]
    assert r['request_float32_bits']==f'{expected:08x}' and r['mode']==('DEFAULT' if request=='DEFAULT' else 'SET')
    identity=re.search(r'HEY_REFERENCE abi=2 mode=(DEFAULT|SET) model=(\S+) request_bits=([0-9a-f]+) actual_bits=([0-9a-f]+) actual=([0-9.eE+-]+)',text)
    assert identity and identity[1]==r['mode'] and identity[2]==r['model'] and identity[3]==r['request_float32_bits'] and identity[4]==r['actual_float32_bits']
    assert math.isfinite(float(identity[5])) and 0<float(identity[5])<1
    assert struct.pack('<f',float(identity[5]))==struct.pack('<I',int(identity[4],16))
    after=re.search(r'HEY_REFERENCE_AFTER_PRIME actual_bits=([0-9a-f]+) actual=([0-9.eE+-]+)',text)
    assert after and after[1]==r['actual_float32_bits']==r['post_prime_getter_bits']
    assert struct.pack('<f',float(after[2]))==struct.pack('<I',int(after[1],16))
    assert re.search(r'NYX_PCM_PARAMETERS chunk=512 rate=16000 channels=1 set_result='+('0' if request=='DEFAULT' else '1')+r' actual=',text)
    primes=[(int(c),int(v)) for c,v in re.findall(r'HEY_REFERENCE_PRIME call=(\d+) result=(-?\d+)',text)]
    assert [c for c,v in primes]==list(range(1,64)) and primes==[tuple(v) for v in r['prime_returns']] and all(v in (-1,0,1) for c,v in primes)
    returns=[(int(c),int(end),int(v)) for c,end,v in re.findall(r'HEY_REFERENCE_RETURN call=(\d+) sample_end=(\d+) result=(-?\d+)',text)]
    assert len(returns)==r['input_calls'] and all(c==i and end==min(i*512,row['samples']) and v in (-1,0,1) for i,(c,end,v) in enumerate(returns,1))
    trace=root/(stem+'-raw-returns.bin');assert sha(trace)==r['raw_returns_sha256'] and trace.read_bytes()==bytes(v+1 for c,end,v in returns)
    events=[dict(call=c,sample_end=end,result=v) for c,end,v in returns if v==1]
    assert events==[{k:e[k] for k in ('call','sample_end','result')} for e in r['events']]
    done=re.search(r'NYX_PCM_DONE samples=(\d+) calls=(\d+) events=(\d+) verified=(\d+) crc32=([0-9a-f]+)',text)
    assert done and int(done[1])==row['samples'] and int(done[2])==r['input_calls'] and int(done[3])==len(events) and int(done[4])==sum(v==-1 for c,end,v in returns)
    assert int(done[5],16)==int(r['input_crc32'],16)==crc
    return r,events,dict(report_sha256=sha(rp),uart_sha256=sha(uart),raw_returns_sha256=sha(trace),prime_positive_events=sum(v==1 for c,v in primes))


def main():
    p=json.loads(PLAN.read_text())
    for name,h in p['files'].items():assert sha(name)==h
    root=Path(p['output']);done=json.loads((root/'complete.json').read_text());assert done['streams']==p['streams']==9672 and done['execution_sha256']==sha(PLAN) and done['retries']==0
    crcs=[]
    for row in p['records']:
        crc=0
        with sf.SoundFile(row['path']) as f:
            while True:
                values=f.read(1048576,dtype='int16')
                if not len(values):break
                crc=zlib.crc32(values.astype('<i2').tobytes(),crc)
        crcs.append(crc)
    runtime_hashes={key:sha(value) for key,value in p['runtime'].items()}
    curves={}
    for model in p['models']:
        points=[]
        for config,request in enumerate(p['threshold_requests']):
            clips=[];events=0;hours=0;getters=set();audits=[]
            for index,row in enumerate(p['records']):
                r,ev,a=audit_stream(root,f'{model}-{config:02}-{index:03}',row,model,request,runtime_hashes,crcs[index]);getters.add(r['actual_float32_bits']);audits.append(a)
                if row['phase']=='synthetic':clips.append((row,ev))
                else:events+=len(ev);hours+=row['samples']/16000/3600
            assert len(getters)==1 and abs(hours-p['background_hours'])<1e-10
            m=metrics(clips,model,True);upper=far_upper(events,hours)
            ok=upper<=.5 and m['macro']>=.8 and m['standalone']>=.75 and m['command']>=.75 and m['pre_source_positive_events']==0 and all(n['events']==0 for n in m['negatives'].values())
            points.append(dict(request=request,actual_float32_bits=next(iter(getters)),synthetic=m,background_events=events,background_hours=hours,background_upper95=upper,eligible_diagnostic=ok,stream_audits=audits))
            print(json.dumps(dict(model=model,configuration=config,audited=372)),flush=True)
        curves[model]=points
    result=dict(schema='hey-official-development-curves-v1',execution_sha256=sha(PLAN),curves=curves,threshold_selected=False,acceptance_authorized=False,candidate_promoted=False,limits='Complete official diagnostic development curves, actual getter duplicates retained. Classic QEMU and generated timing only, not universal P4/human acoustic equivalence or independent acceptance.')
    with (root/'calibration.json').open('x') as out:out.write(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
