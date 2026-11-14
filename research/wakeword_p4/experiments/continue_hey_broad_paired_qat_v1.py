"""Wait original bound process; publish/audit/export/screen once, fail closed."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
R=Path('research/wakeword_p4/results');HERE=Path(__file__).parent;LOCK=R/'4090-hey-broad-paired-qat-continuation-execution-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def proc_identity(pid):
 p=Path(f'/proc/{pid}/stat')
 try:return p.read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
def publish(paths,message):
 subprocess.run(['git','add',*map(str,paths)],check=True);subprocess.run(['git','commit','-m',message],check=True);subprocess.run(['git','push','origin','research-hye-nyx-voice-gpu'],check=True)
def command(script,*args,gpu=False):
 cmd=[sys.executable]
 if gpu:cmd.append('research/wakeword_p4/training/gpu_python.py')
 subprocess.run(cmd+[str(HERE/script),*args],check=True)
def main():
 lock=json.loads(LOCK.read_text());assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
 for p,h in lock['files'].items():assert sha(p)==h,p
 pid=lock['pid'];identity=lock['process_starttime']
 while proc_identity(pid)==identity:time.sleep(10)
 print('Original formal process terminal, checking complete result',flush=True)
 formal=R/'4090-hey-broad-paired-qat-formal-result-v1.json';assert formal.exists(),'Original run missing complete result; no retry'
 command('audit_hey_broad_paired_qat_formal_v1.py')
 publish([formal,R/'4090-hey-broad-paired-qat-formal-independent-audit-v1.json'],'research: publish original paired fixed10000 training and independent audit')
 command('export_hey_broad_paired_qat_v1.py','lock')
 publish([R/'4090-hey-broad-paired-qat-export-execution-v1.json'],'research: lock actual four final weights and single matched export')
 command('export_hey_broad_paired_qat_v1.py','run',gpu=True)
 publish([R/'4090-hey-broad-paired-qat-export-result-v1.json'],'research: preserve four original final INT8 exports and static state identities')
 command('screen_hey_broad_paired_qat_v1.py','lock')
 publish([R/'4090-hey-broad-paired-qat-screen-execution-v1.json'],'research: lock actual final INT8 bytes for complete matched540 numerical screen')
 command('screen_hey_broad_paired_qat_v1.py','run',gpu=True)
 publish([R/'4090-hey-broad-paired-qat-screen-result-v1.json'],'research: preserve complete matched540 final numerical outcomes')
 print('Complete original postprocessing sequence; no freshdev/acceptance or promotion',flush=True)
if __name__=='__main__':main()
