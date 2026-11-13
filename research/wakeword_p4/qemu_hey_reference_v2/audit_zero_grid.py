"""Independent complete zero-UART/getter/raw-return audit; no speech access."""
import json,re,struct
from pathlib import Path
from run_zero_grid import PLAN,sha

def main():
    p=json.loads(PLAN.read_text());root=Path(p['output']);c=json.loads((root/'complete.json').read_text())
    assert c['plan_sha256']==sha(PLAN) and c['cases']==26 and c['retries']==0
    rows=[]
    for i,(case,r) in enumerate(zip(p['cases'],c['reports'])):
        text=(root/f'{i:02}-serial.log').read_text()
        assert sha(root/f'{i:02}-serial.log')==r['serial_sha256']
        after=re.search(r'HEY_REFERENCE_AFTER_PRIME actual_bits=([0-9a-f]+) actual=([0-9.eE+-]+)',text)
        assert after and after[1]==r['actual_float32_bits']==r['post_prime_getter_bits']
        assert struct.pack('<f',float(after[2]))==struct.pack('<I',int(after[1],16))
        raw=[(int(a),int(b),int(v)) for a,b,v in re.findall(r'HEY_REFERENCE_RETURN call=(\d+) sample_end=(\d+) result=(-?\d+)',text)]
        assert len(raw)==32 and all(a==j and b==min(j*512,16001) and v in (-1,0) for j,(a,b,v) in enumerate(raw,1))
        trace=root/f'{i:02}-raw-returns.bin';assert trace.read_bytes()==bytes(v+1 for a,b,v in raw) and sha(trace)==r['raw_returns_sha256']
        prime=[(int(a),int(v)) for a,v in re.findall(r'HEY_REFERENCE_PRIME call=(\d+) result=(-?\d+)',text)]
        assert len(prime)==63 and prime==[tuple(v) for v in r['prime_returns']]
        assert r['model']==('wn9_heywillow_tts' if case['model']=='willow' else 'wn9_hiesp')
        assert r['stage']==104 and r['input_samples']==16001 and not r['events']
        rows.append(dict(case=case,actual_bits=after[1],report_sha256=sha(root/f'{i:02}-report.json'),uart_sha256=r['serial_sha256']))
    result=dict(schema='hey-qemu-zero-grid-audit-v2',plan_sha256=sha(PLAN),auditor_sha256=sha(__file__),cases=26,prime_returns=26*63,input_returns=26*32,all_post_prime_getters_stable=True,records=rows,limits='Synthetic-zero API/complete-return check only; no speech/FAR or universal P4 equivalence.')
    path=Path('research/wakeword_p4/results/4090-hey-qemu-zero-grid-audit-v2.json')
    with path.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(cases=26,all_post_prime_getters_stable=True)))
if __name__=='__main__':main()
