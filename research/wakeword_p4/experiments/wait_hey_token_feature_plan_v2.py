"""Observe original token task, then write separate v2 feature plan; never restart."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import time


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return fields[0],fields[19]
    except FileNotFoundError:return None


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid',type=int,required=True)
    parser.add_argument('--starttime',required=True)
    a=parser.parse_args()
    current=identity(a.pid)
    if current is None or current[1]!=a.starttime or current[0]=='Z':
        raise ValueError('Must attach to the original live process')
    if b'extract_hey_token_durations_v1.py\0extract-training' not in Path(f'/proc/{a.pid}/cmdline').read_bytes():
        raise ValueError('Unexpected observed command')
    builder=Path(__file__).with_name('build_hey_training_features_v2.py')
    key=str(builder.resolve().relative_to(Path.cwd()))
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+key])
    if remote!=builder.read_bytes():raise ValueError('Feature builder must already be pushed')
    expected=sha(builder)
    print(f'observing pid={a.pid} starttime={a.starttime}; no restart',flush=True)
    while True:
        current=identity(a.pid)
        if current is None or current[0]=='Z':break
        if current[1]!=a.starttime:raise ValueError('PID reused; original task is terminal, do not follow replacement')
        time.sleep(2)
    if sha(builder)!=expected:raise ValueError('Builder changed during observation')
    # Builder fails closed on missing/partial/mismatched token records or any
    # unsupported prefix. Never creates features or launches training here.
    subprocess.run([sys.executable,str(builder),'write-plan'],check=True)
    print('feature plan written; push exact plan before build; no training launched',flush=True)


if __name__=='__main__':main()
