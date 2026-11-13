"""Wait for original bound schedule; publish verified lock; run two fixed arms."""
import json
from pathlib import Path
import subprocess
import sys
import time

PID=1034876
START='19738022'
ROOT=Path('research/wakeword_p4')
LOCK=ROOT/'results/4090-hey-broad-formal-training-lock-v1.json'


def main():
    proc=Path(f'/proc/{PID}/stat')
    while proc.exists():
        try:current=proc.read_text().rsplit(')',1)[1].split()[19]
        except FileNotFoundError:break
        if current!=START:break
        time.sleep(5)
    # No restart/retry of original generation, training or failed output dirs.
    report=ROOT/'results/4090-hey-broad-schedules-v1.json'
    assert report.exists(),'Original schedule did not produce complete report; stop'
    d=json.loads(report.read_text());assert len(d['arms'])==2 and all(a['batches']==120100 for a in d['arms'])
    assert not LOCK.exists(),'Existing formal lock requires explicit inspection; stop'
    subprocess.run([sys.executable,str(ROOT/'experiments/lock_hey_broad_formal_v1.py'),'--write'],check=True)
    subprocess.run(['git','diff','--cached','--quiet'],check=True)
    subprocess.run(['git','add',str(LOCK)],check=True)
    subprocess.run(['git','commit','-m','research: freeze complete broad225 formal120k training lock'],check=True)
    subprocess.run(['git','push','origin','HEAD:research-hye-nyx-voice-gpu'],check=True)
    for target in ('hey_willow','hey_nyx'):
        log=Path('.local/wakeword')/f'hey-broad-formal-{target}-v1.log'
        with log.open('x') as stream:
            print(json.dumps(dict(starting=target,log=str(log))),flush=True)
            subprocess.run([sys.executable,str(ROOT/'training/gpu_python.py'),
                str(ROOT/'experiments/run_hey_broad_formal_v1.py'),'--training-lock',str(LOCK),'--target',target],
                stdout=stream,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(completed=target)),flush=True)


if __name__=='__main__':main()
