"""Durable serial GPU supervisor; explicit child exit and process identity receipts."""
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path('.local/wakeword/context-pair-supervisor-v1')
def save(state):
    tmp=ROOT/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(ROOT/'state.json')
def main():
    ROOT.mkdir(exist_ok=False)
    state=dict(pid=os.getpid(),proc_starttime=Path('/proc/self/stat').read_text().split()[21],status='starting',arms=[])
    save(state)
    for condition in ['silence','speech']:
        command=['/usr/bin/python3','research/wakeword_p4/training/gpu_python.py','research/wakeword_p4/experiments/train_chatterbox_context_pair_v1.py','run',condition]
        with (ROOT/f'{condition}.log').open('xb') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            arm=dict(condition=condition,pid=child.pid,proc_starttime=Path(f'/proc/{child.pid}/stat').read_text().split()[21],status='running',command=command,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            state['arms'].append(arm);state['status']='running';save(state)
            arm['exit_code']=child.wait();arm['status']='complete' if arm['exit_code']==0 else 'failed';arm['ended_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save(state)
            if arm['exit_code']!=0:state['status']='failed';save(state);return arm['exit_code']
    state['status']='completed-both-await-export';save(state);return 0
if __name__=='__main__':raise SystemExit(main())
