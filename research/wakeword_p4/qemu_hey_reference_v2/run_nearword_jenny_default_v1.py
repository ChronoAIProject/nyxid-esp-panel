"""One official native DEFAULT point on existing Jenny development PCM."""
import argparse,ast,hashlib,json,subprocess,sys,shutil,os
from pathlib import Path
ROOT=Path('research/wakeword_p4');PLAN=ROOT/'results/4090-nearword-jenny-official-default-plan-v1.json'
SOURCE=Path(__file__).with_name('run_development.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def register():
    parent=ROOT/'results/4090-causal-nearword-jenny-plan-v1.json';custom=json.loads(parent.read_text());old=ROOT/'results/4090-hey-official-development-execution-v1.json';runtime=json.loads(old.read_text())['runtime'];rows=[]
    files={str(p):sha(p) for p in [Path(__file__),SOURCE,parent,old]+[Path(p) for p in runtime.values()]}
    for phase,material in custom['materials'].items():
        p=Path(material['path']);assert sha(p)==material['sha256'];files[str(p)]=sha(p)
        for row in json.loads(p.read_text())['records']:
            r=dict(row,phase=phase)
            if phase=='background':r.update(path=row['wav_path'],sha256=row['wav_sha256'])
            assert sha(r['path'])==r['sha256'];files[r['path']]=r['sha256'];rows.append(r)
    assert len(rows)==408
    p=dict(schema='nearword-jenny-official-default-plan-v1',files=files,runtime=runtime,records=rows,threshold_requests=['DEFAULT'],models=['hiesp'],streams=408,output='.local/wakeword/nearword-jenny-official-default-v1',policy='One fixed official native DEFAULT DET_MODE_95 operating point, no grid or threshold equality. Same396 original whole synthetic PCM and12 complete9.614h background as candidate/v30. Native512sample chunks and existing63zero-call32256sample official warmup, versus candidate81600prime: initialization differs explicitly, not exact identical timing protocol or P4 equivalence. Complete native returns/events retained; generated boundaries only synthesis reference. Development only, no final acceptance/training/newTTS. Original HiESP target utterances scored as target; HeyNyx utterances as cross negatives. No selection or promotion.')
    with PLAN.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    print(sha(PLAN))
def case():
    tree=ast.parse(SOURCE.read_text());count=0
    for n in ast.walk(tree):
        if isinstance(n,ast.Constant) and n.value=='research/wakeword_p4/results/4090-hey-official-development-execution-v1.json':n.value=str(PLAN);count+=1
    assert count==1
    exec(compile(ast.fix_missing_locations(tree),str(SOURCE),'exec'),dict(__name__='__main__',__file__=__file__))
def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=json.loads(PLAN.read_text())
    for f,h in p['files'].items():assert sha(f)==h,f
    subprocess.run([sys.executable,str(ROOT/'tools/check_disk_budget.py'),'--expected-gib','3'],check=True)
    out=Path(p['output']);out.mkdir(exist_ok=False);(out/'launch.json').write_text(json.dumps(dict(pid=os.getpid(),starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(PLAN)))+'\n')
    for i,row in enumerate(p['records']):
        tmp=Path(f'/tmp/nj-hiesp-{i:03}');assert not tmp.exists()
        cmd=[sys.executable,str(__file__),'case']
        for name,path in p['runtime'].items():cmd+=['--'+name,path]
        cmd+=['--model','hiesp','--threshold','DEFAULT','--record-index',str(i),'--output',str(tmp)]
        with (out/f'{i:03}.log').open('xb') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        for name in ['report.json','serial.log','stage-monitor.txt','raw-returns.bin']:
            source=tmp/name;dest=out/f'{i:03}-{name}';shutil.copyfile(source,dest);assert sha(source)==sha(dest)
        shutil.rmtree(tmp)
        print(json.dumps(dict(completed=i+1,total=408,phase=row['phase'])),flush=True)
    (out/'complete.json').write_text(json.dumps(dict(streams=408,plan_sha256=sha(PLAN),acceptance_accessed=False))+'\n')
if __name__=='__main__':
    action=sys.argv.pop(1)
    {'register':register,'run':run,'case':case}[action]()
