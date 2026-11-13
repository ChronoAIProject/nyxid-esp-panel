"""Audit one frozen >hour consumed stream and uncapped raw event transport."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zlib

import soundfile as sf

PLAN=Path('research/wakeword_p4/results/4090-hey-qemu-repeated-consumed-plan-v1.json')
ROOT=Path('research/wakeword_p4/results')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    plan=json.loads(PLAN.read_text())
    for p,h in plan['files'].items():assert sha(p)==h,p
    dest=Path(plan['output']);report=json.loads((dest/'report.json').read_text())
    text=(dest/'serial.log').read_text()
    pcm,rate=sf.read('.local/wakeword/mac-consumed-speech-grid-witness-v1/fixtures/clip-04.wav',dtype='int16')
    assert rate==16000
    raw=pcm.astype('<i2').tobytes();crc=0;h=hashlib.sha256()
    for _ in range(plan['repeat_count']):h.update(raw);crc=zlib.crc32(raw,crc)
    assert h.hexdigest()==plan['input_pcm_sha256']==report['pcm_sha256']
    samples=len(pcm)*plan['repeat_count'];assert samples==plan['input_samples']==report['input_samples']
    assert samples>16000*3600 and samples>=68499840
    done=re.search(r'NYX_PCM_DONE samples=(\d+) calls=(\d+) events=(\d+) verified=(\d+) crc32=([0-9a-f]+)',text)
    assert done and int(done[1])==samples and int(done[2])==report['input_calls']==(samples+511)//512
    assert int(done[5],16)==crc==int(report['input_crc32'],16)
    events=[dict(call=int(c),sample_end=int(s),result=int(r)) for c,s,r in re.findall(r'NYX_PCM_EVENT call=(\d+) sample_end=(\d+) result=(-?\d+)',text)]
    assert events==[{k:e[k] for k in ('call','sample_end','result')} for e in report['events']]
    assert int(done[3])==len(events)>64
    calls=[e['call'] for e in events];assert calls==sorted(set(calls))
    for e in events:assert e['result']==1 and 1<=e['call']<=report['input_calls'] and e['sample_end']==min(e['call']*512,samples)
    primes=[(int(c),int(r)) for c,r in re.findall(r'HEY_REFERENCE_PRIME call=(\d+) result=(-?\d+)',text)]
    assert [c for c,r in primes]==list(range(1,64)) and primes==[tuple(v) for v in report['prime_returns']]
    assert not any(r==1 for c,r in primes)
    assert re.search(r':\s+0x00000068\b',(dest/'stage-monitor.txt').read_text()) and report['stage']==104
    assert report['model']=='wn9_heywillow_tts' and report['mode']=='DEFAULT'
    assert report['actual_float32_bits']=='3f24dd2f'
    for name,ext in [('report.json','json'),('serial.log','uart.log'),('stage-monitor.txt','stage.txt')]:
        shutil.copyfile(dest/name,ROOT/f'4090-hey-qemu-repeated-consumed-original-v1.{ext}')
    result=dict(schema='hey-qemu-long-event-transport-audit-v1',plan_sha256=sha(PLAN),auditor_sha256=sha(__file__),
                samples=samples,seconds=samples/16000,calls=report['input_calls'],prime_returns=len(primes),
                positive_events=len(events),crc32=f'{crc:08x}',stage=104,whole_input_crc_matched=True,
                exceeds_one_hour=True,exceeds_64_events=True,source_resets=0,
                original_report_sha256=sha(dest/'report.json'),uart_sha256=sha(dest/'serial.log'),
                limits='Repeated consumed PCM proves this runtime whole-source and uncapped-event transport only. Not1235 independent speakers/trials or FAR, recall, P4 parity, or acceptance; no fresh held-out audio accessed.')
    with (ROOT/'4090-hey-qemu-long-event-transport-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
