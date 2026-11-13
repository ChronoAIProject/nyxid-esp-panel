"""Execute exactly one frozen156-stream consumed speech witness; no retries."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PLAN=Path('research/wakeword_p4/results/4090-hey-qemu-consumed-grid-plan-v1.json')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes();plan=json.loads(remote)
    for p,h in plan['files'].items():assert sha(p)==h,p
    dest=Path(plan['output']);dest.mkdir(exist_ok=False)
    manifest=list(map(json.loads,Path(plan['manifest']).read_text().splitlines()))
    assert len(manifest)==156
    common=[sys.executable,plan['runner']]
    for name,path in plan['runtime'].items():common+=['--'+name,path]
    for index,row in enumerate(manifest):
        output=f'/tmp/hy-cg-v1-{index:03}'
        threshold='DEFAULT' if row['requested_decimal'] is None else str(row['requested_decimal'])
        command=common+['--model','willow','--threshold',threshold,'--record-index',str(index),'--output',output]
        with (dest/f'{index:03}.log').open('xb') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
        for name in ('report.json','serial.log','stage-monitor.txt'):
            (dest/f'{index:03}-{name}').write_bytes((Path(output)/name).read_bytes())
        print(json.dumps(dict(completed=index+1,total=156)),flush=True)
    (dest/'complete.json').write_text(json.dumps(dict(streams=156,plan_sha256=sha(PLAN),retries=0))+'\n')


if __name__=='__main__':main()
