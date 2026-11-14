"""Wait for original material replay, publish locks, export and numerical screen once."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('research/wakeword_p4')
BRANCH='research-hye-nyx-voice-gpu'
MATERIAL=ROOT/'results/4090-hey-broad-ptq-material-result-v1.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None

def publish(paths,message):
    subprocess.run(['git','add','--',*[str(p) for p in paths]],check=True)
    subprocess.run(['git','commit','-m',message,'--',*[str(p) for p in paths]],check=True)
    subprocess.run(['git','push','origin',BRANCH],check=True)

def run(script,action):
    subprocess.run([sys.executable,str(ROOT/'training/gpu_python.py'),str(ROOT/'experiments'/script),action],check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pid',type=int,required=True);p.add_argument('--starttime',required=True);a=p.parse_args()
    me=Path(__file__).relative_to(Path.cwd()) if Path(__file__).is_absolute() else Path(__file__)
    assert subprocess.check_output(['git','show',f'origin/{BRANCH}:{me}'])==me.read_bytes()
    lockpath=ROOT/'results/4090-hey-broad-bounded-continuation-v1.json'
    assert subprocess.check_output(['git','show',f'origin/{BRANCH}:{lockpath}'])==lockpath.read_bytes()
    lock=json.loads(lockpath.read_text());assert lock['pid']==a.pid and lock['starttime']==a.starttime
    for path,h in lock['files'].items():assert sha(path)==h,path
    while True:
        current=identity(a.pid)
        if current!=a.starttime:break
        print(json.dumps(dict(waiting_original_pid=a.pid,starttime=current)),flush=True)
        time.sleep(10)
    assert MATERIAL.exists(),'Original replay terminated without completion report; no retry'
    material=json.loads(MATERIAL.read_text());assert len(material['arms'])==2
    for row in material['arms']:
        assert row['verified_batches']==120100 and row['shape']==[100,1,225,40]
        assert sha(row['path'])==row['sha256']
    publish([MATERIAL],'research: authenticate original PTQ calibration tensors for bounded control')
    run('export_hey_broad_bounded_v1.py','lock')
    exportlock=ROOT/'results/4090-hey-broad-bounded-export-execution-v1.json'
    publish([exportlock],'research: pin actual bounded export execution dependencies')
    run('export_hey_broad_bounded_v1.py','run')
    export=ROOT/'results/4090-hey-broad-bounded-export-result-v1.json'
    publish([export],'research: retain controlled bounded head export and tensor metadata')
    if not all(r['static_gate_passed'] for r in json.loads(export.read_text())['arms']):
        print('Static export gate rejected; no inference or retry',flush=True);return
    run('screen_hey_broad_bounded_v1.py','lock')
    screenlock=ROOT/'results/4090-hey-broad-bounded-screen-execution-v1.json'
    publish([screenlock],'research: bind bounded models before original 270 numerical gates')
    run('screen_hey_broad_bounded_v1.py','run')
    result=ROOT/'results/4090-hey-broad-bounded-screen-result-v1.json'
    publish([result],'research: preserve complete bounded model numerical screen results')
    print(json.dumps(dict(completed=True,all_numerical_passed=json.loads(result.read_text())['all_passed'],development_authorized=False)),flush=True)

if __name__=='__main__':main()
