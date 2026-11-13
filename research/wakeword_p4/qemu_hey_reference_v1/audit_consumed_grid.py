"""Independent saved-UART accounting and full-event native/QEMU comparison."""
import hashlib
import json
from pathlib import Path
import re

PLAN=Path('research/wakeword_p4/results/4090-hey-qemu-consumed-grid-plan-v1.json')
OUTPUT=Path('research/wakeword_p4/results/4090-hey-qemu-consumed-grid-comparison-v1.json')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit():
    plan=json.loads(PLAN.read_text())
    for p,h in plan['files'].items():assert sha(p)==h,p
    dest=Path(plan['output']);complete=json.loads((dest/'complete.json').read_text())
    assert complete['streams']==156 and complete['plan_sha256']==sha(PLAN) and complete['retries']==0
    native_path=Path('.local/wakeword/mac-consumed-speech-grid-witness-v1/reports/native-report.json')
    native=json.loads(native_path.read_text())['clips'];assert len(native)==156
    rows=[]
    for i,n in enumerate(native):
        report=dest/f'{i:03}-report.json';uart=dest/f'{i:03}-serial.log'
        q=json.loads(report.read_text());text=uart.read_text()
        assert q['record_index']==i and q['pcm_sha256']==n['pcm_sha256']
        prime=[int(v) for c,v in re.findall(r'HEY_REFERENCE_PRIME call=(\d+) result=(-?\d+)',text)]
        assert len(prime)==63 and prime==[v for c,v in q['prime_returns']]
        events=[dict(call=int(c),sample_end=int(s),result=int(r)) for c,s,r in
                re.findall(r'NYX_PCM_EVENT call=(\d+) sample_end=(\d+) result=(-?\d+)',text)]
        assert events==[{k:e[k] for k in ('call','sample_end','result')} for e in q['events']]
        for e in events:assert e['sample_end']==min(e['call']*512,q['input_samples']) and e['result']==1
        done=re.search(r'NYX_PCM_DONE samples=(\d+) calls=(\d+) events=(\d+) verified=(\d+) crc32=([0-9a-f]+)',text)
        assert done and int(done[1])==q['input_samples']==n['samples']
        assert int(done[2])==q['input_calls']==n['result']['calls']==(n['samples']+511)//512
        assert int(done[3])==len(events) and int(done[5],16)==int(q['input_crc32'],16)==n['result']['crc32']
        native_events=[{k:e[k] for k in ('call','sample_end','result')} for e in n['raw_events']]
        control=n['initialization']['control']
        comparisons=dict(getter_bits=q['actual_float32_bits']==control['getter_after_set_f32_hex'],
                         prime_returns=prime==control['prime_returns'],
                         positive_events=events==native_events,
                         channel_verified=int(done[4])==n['result']['channel_verified_calls'])
        rows.append(dict(index=i,point=n['point'],voice=n['voice'],target=n['target'],form=n['form'],
                         samples=n['samples'],calls=q['input_calls'],pcm_sha256=q['pcm_sha256'],
                         getter_qemu=q['actual_float32_bits'],getter_native=control['getter_after_set_f32_hex'],
                         qemu_events=events,native_events=native_events,exact_comparisons=comparisons,
                         report_sha256=sha(report),uart_sha256=sha(uart)))
    mismatches=[r for r in rows if not all(r['exact_comparisons'].values())]
    result=dict(schema='hey-qemu-native-consumed-grid-comparison-v1',plan_sha256=sha(PLAN),
                auditor_sha256=sha(Path(__file__)),native_report_sha256=sha(native_path),
                streams=156,input_calls=sum(r['calls'] for r in rows),prime_returns=156*63,
                qemu_events=sum(len(r['qemu_events']) for r in rows),native_events=sum(len(r['native_events']) for r in rows),
                streams_with_any_difference=len(mismatches),rows=rows,
                limits='Saved complete-event comparison on consumed12PCM/13config only. Classic ESP32 and P4 remain distinct runtimes; this is not fresh recognition/FAR/acceptance or universal equivalence.')
    with OUTPUT.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':audit()
