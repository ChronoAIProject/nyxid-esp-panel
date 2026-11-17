"""One training child, persistent actual exit receipt, no retries or restart."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('.local/wakeword/c161-envelope-supervisor-v1')
REPORT=Path('research/wakeword_p4/results/4090-c161-envelope-supervisor-original-v1.json')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    ROOT.mkdir(exist_ok=False)
    command=['/usr/bin/python3','research/wakeword_p4/training/gpu_python.py','research/wakeword_p4/experiments/train_c161_envelope_v1.py','run']
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TF_NUM_INTEROP_THREADS='1')
    identity=dict(supervisor_pid=os.getpid(),supervisor_starttime=Path('/proc/self/stat').read_text().split()[21],command=command,script_sha256=sha(__file__))
    start=time.monotonic()
    with (ROOT/'training.log').open('xb') as log:
        child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        identity.update(child_pid=child.pid,child_starttime=Path(f'/proc/{child.pid}/stat').read_text().split()[21])
        (ROOT/'state.json').write_text(json.dumps(dict(identity=identity,status='running'),indent=2)+'\n')
        code=child.wait()
    result=dict(identity=identity,status='completed-await-export' if code==0 else 'failed',actual_exit_code=code,elapsed_seconds=time.monotonic()-start,log_sha256=sha(ROOT/'training.log'))
    (ROOT/'state.json').write_text(json.dumps(result,indent=2)+'\n')
    with REPORT.open('x') as out:json.dump(result,out,indent=2)
    print(json.dumps(result),flush=True)
    raise SystemExit(code)
if __name__=='__main__':main()
