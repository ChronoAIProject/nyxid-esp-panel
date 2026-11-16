"""Tracked single GPU child with PID/starttime and actual terminal exit receipt."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path('.local/wakeword/daily-context-supervisor-v1')
def main():
    ROOT.mkdir(exist_ok=False)
    s=dict(pid=os.getpid(),starttime=Path('/proc/self/stat').read_text().split()[21],status='starting')
    def save():
        p=ROOT/'state.tmp';p.write_text(json.dumps(s,indent=2)+'\n');p.replace(ROOT/'state.json')
    save()
    command=['/usr/bin/python3','research/wakeword_p4/training/gpu_python.py','research/wakeword_p4/experiments/train_chatterbox_daily_context_v1.py','run']
    with (ROOT/'training.log').open('xb') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        s.update(status='running',child_pid=child.pid,child_starttime=Path(f'/proc/{child.pid}/stat').read_text().split()[21],command=command);save()
        code=child.wait();s.update(status='completed-await-export' if code==0 else 'failed',exit_code=code,ended_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
    return code
if __name__=='__main__':raise SystemExit(main())
