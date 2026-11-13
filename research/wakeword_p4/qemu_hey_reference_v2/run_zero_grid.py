"""Exactly26 prelocked API configurations on synthetic zeros; no audio splits."""
import hashlib,json,subprocess,sys
from pathlib import Path
PLAN=Path('research/wakeword_p4/results/4090-hey-qemu-zero-grid-plan-v2.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes();p=json.loads(remote)
    for name,h in p['files'].items():assert sha(name)==h
    dest=Path(p['output']);dest.mkdir(exist_ok=False)
    reports=[]
    for i,case in enumerate(p['cases']):
        out=Path(f'/tmp/hy-zg-v2-{i:02}')
        command=[sys.executable,p['runner']]
        for name,path in p['runtime'].items():command+=['--'+name,path]
        command+=['--model',case['model'],'--threshold',case['threshold'],'--output',str(out)]
        with (dest/f'{i:02}.log').open('xb') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        for name in ('report.json','serial.log','stage-monitor.txt','raw-returns.bin'):(dest/f'{i:02}-{name}').write_bytes((out/name).read_bytes())
        reports.append(json.loads((out/'report.json').read_text()))
        print(json.dumps(dict(completed=i+1,total=26)),flush=True)
    (dest/'complete.json').write_text(json.dumps(dict(schema='hey-qemu-zero-grid-completion-v2',plan_sha256=sha(PLAN),cases=26,retries=0,reports=reports),indent=2)+'\n')
if __name__=='__main__':main()
