"""Sequential final-export/replay evidence continuation; never starts training."""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/binary40-continuation-v1')
TRAIN=Path('.local/wakeword/binary40-training-v1')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    B.mkdir(exist_ok=False)
    state=dict(schema='binary40-continuation-original-v1',pid=os.getpid(),status='waiting-original-training',stages=[])
    dest=R/'4090-binary40-continuation-original-v1.json'
    def save():
        (B/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    def stage(name,args):
        path=B/(name+'.log');start=time.time()
        with path.open('w') as f:result=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
        row=dict(name=name,args=args,actual_exit_code=result.returncode,seconds=time.time()-start,log=str(path),log_sha256=sha(path));state['stages'].append(row);save()
        print(json.dumps(row),flush=True)
        if result.returncode:raise RuntimeError(f'{name} actual exit {result.returncode}')
    def publish(paths,message):
        existing=[str(p) for p in paths if Path(p).exists()]
        subprocess.run(['git','add','--',*existing],check=True)
        # The root may independently publish a verified terminal receipt. Do not
        # fail or manufacture an empty commit when those files are already clean.
        changed=subprocess.run(['git','diff','--cached','--quiet','--',*existing]).returncode
        if changed:
            subprocess.run(['git','commit','-m',message],check=True)
            subprocess.run(['git','push','origin','research-hye-nyx-voice-gpu'],check=True)
    try:
        launch=json.loads((TRAIN/'launch.json').read_text());state['training_launch']=launch;save()
        stat=Path(f'/proc/{launch["pid"]}/stat')
        while stat.exists():
            try:fields=stat.read_text().split()
            except FileNotFoundError:break
            if fields[21]!=launch['proc_starttime'] or fields[2]=='Z':break
            time.sleep(5)
        # Process disappearance is not an actual exit-code receipt. Require
        # completed artifacts for safe next work; root separately polls8373.
        report=R/'4090-binary40-training-original-v1.json';tr=json.loads(report.read_text())
        assert tr['steps']==20000 and tr['launch']==launch
        for name,h in tr['files'].items():assert sha(TRAIN/name)==h
        state.update(status='running',training_terminal_observation='original PID/starttime no longer running; completed final20000 artifact hashes verified; actual training exit code not observed by this non-parent')
        save();publish([report],'research: preserve completed Binary40 training evidence')
        py='/usr/bin/python3';base='research/wakeword_p4/experiments/'
        stage('export',[py,'research/wakeword_p4/training/gpu_python.py',base+'export_binary40_trained_v1.py','run'])
        publish([R/'4090-binary40-trained-export-original-v1.json'],'research: publish final trained Binary40 integer export evidence')
        cohorts=['builtin','daily','continuous','primed','jenny','joe','melo','mac']
        for c in cohorts:stage('plan-'+c,[py,base+'replay_binary40_v1.py','plan',c])
        publish([R/f'4090-binary40-{c}-replay-plan-v1.json' for c in cohorts],'research: lock all final Binary40 consumed replay identities')
        for c in cohorts:
            stage('replay-'+c,[py,base+'replay_binary40_v1.py','run',c])
            stage('score-'+c,[py,base+'score_binary40_v1.py',c])
            publish([R/f'4090-binary40-{c}-comparison-v1.json',R/f'4090-binary40-{c}-evidence-v1.tar.gz'],f'research: preserve Binary40 {c} complete original evidence')
        state['status']='completed-await-review'
    except Exception:
        state['status']='failed-preserved';state['error']=traceback.format_exc();(B/'failure.log').write_text(state['error'])
    finally:
        save();dest.write_text(json.dumps(state,indent=2)+'\n')
        if state['status']!='waiting-original-training':publish([dest],'research: preserve Binary40 continuation execution receipts')
    if state['status']!='completed-await-review':raise SystemExit(1)
if __name__=='__main__':main()
