"""Wait for the existing supervisor, then execute one fixed research pipeline."""
import hashlib,json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path('.local/wakeword/c161-envelope-continuation-v1')
SUP=Path('.local/wakeword/c161-envelope-supervisor-v1')
R=Path('research/wakeword_p4/results');E=Path('research/wakeword_p4/experiments')
BRANCH='research-hye-nyx-voice-gpu';REPO='ChronoAIProject/nyxid-esp-panel'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def live(pid,start):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split()
        return fields[21]==start and fields[2]!='Z'
    except FileNotFoundError:return False
def main():
    ROOT.mkdir(exist_ok=False);stages=[]
    identity=dict(pid=os.getpid(),starttime=Path('/proc/self/stat').read_text().split()[21],script_sha256=sha(__file__))
    def state(status,**kw):
        target=ROOT/'state.json';temp=ROOT/'state.tmp'
        temp.write_text(json.dumps(dict(identity=identity,status=status,stages=stages,**kw),indent=2)+'\n');temp.replace(target)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TF_NUM_INTEROP_THREADS='1')
    def command(name,argv):
        assert not any(r['name']==name for r in stages)
        log=ROOT/(name+'.log');start=time.monotonic()
        with log.open('xb') as out:
            process=subprocess.Popen(argv,env=env,stdout=out,stderr=subprocess.STDOUT)
            state('stage-running',stage=name,child_pid=process.pid,argv=argv)
            code=process.wait()
        stages.append(dict(name=name,argv=argv,actual_exit_code=code,seconds=time.monotonic()-start,log_sha256=sha(log)))
        state('between-stages')
        if code:raise RuntimeError(f'{name} actual exit {code}; no retry')
    def python(name,script,*args,gpu=False):
        argv=['/usr/bin/python3']
        if gpu:argv.append('research/wakeword_p4/training/gpu_python.py')
        command(name,argv+[str(E/script),*args])
    def publish(name):
        assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==BRANCH
        files=sorted({str(p) for pattern in ['4090-c161-envelope-training-original-v1.json','4090-c161-envelope-supervisor-original-v1.json','4090-context-envelope-*'] for p in R.glob(pattern)})
        assert files
        command(name+'-add',['git','add','--',*files])
        changed=subprocess.run(['git','diff','--cached','--quiet','--',*files]).returncode
        assert changed in [0,1]
        if changed:command(name+'-commit',['git','commit','--only','-m','research: '+name.replace('-',' '),'--',*files])
        command(name+'-push',['git','push','origin',BRANCH])
    try:
        supervisor=json.loads((SUP/'state.json').read_text());owner=supervisor['identity']
        while live(owner['supervisor_pid'],owner['supervisor_starttime']):
            state('waiting-existing-training',supervisor=owner)
            time.sleep(10)
        terminal=json.loads((SUP/'state.json').read_text())
        assert terminal['identity']==owner and terminal['actual_exit_code']==0 and terminal['status']=='completed-await-export'
        receipt=R/'4090-c161-envelope-training-original-v1.json';r=json.loads(receipt.read_text())
        assert r['all_training_checks_passed'] and r['steps']==r['optimizer_iterations']==20000
        assert r['plan_sha256']==sha(R/'4090-c161-envelope-training-plan-v1.json')
        output=Path('.local/wakeword/c161-envelope-training-v1')
        for name,h in r['files'].items():assert sha(output/name)==h
        python('export-plan','export_chatterbox_envelope_v1.py','plan')
        publish('envelope-training-complete-export-locked')
        python('export-run','export_chatterbox_envelope_v1.py','run',gpu=True)
        for cohort in ['builtin','daily','continuous']:
            python('plan-'+cohort,'replay_context_envelope_initial_v1.py','plan',cohort)
        for cohort in ['primed','jenny','joe','melo','mac']:
            python('plan-'+cohort,'replay_context_envelope_coverage_v1.py','plan',cohort)
        publish('envelope-single-export-and-all-coverage-locks')
        for cohort in ['builtin','daily','continuous']:
            python('replay-'+cohort,'replay_context_envelope_initial_v1.py','run',cohort)
        python('score-initial','score_context_envelope_initial_v1.py')
        publish('envelope-initial-fixed-profile-evidence')
        for cohort in ['primed','jenny','joe','melo','mac']:
            python('replay-'+cohort,'replay_context_envelope_coverage_v1.py','run',cohort)
            if cohort=='mac':python('score-mac','score_context_envelope_mac_v1.py')
            else:python('score-'+cohort,'score_context_envelope_coverage_v1.py',cohort)
            publish('envelope-'+cohort+'-fixed-profile-evidence')
        body=ROOT/'issue-body.txt'
        body.write_text('Envelope固定研究流程已完成，全部阶段真实exit0，原始报告/UINT8重评分/逐成员SHA归档已推送research-hye-nyx-voice-gpu。证据为results/4090-context-envelope-*-comparison-v1.json；本条是执行收据，不是候选晋升结论。已完成builtin/daily/continuous及Legacy/Jenny/Joe/Melo/Mac全部既有消费材料；与保存C161及其他已完成候选逐条对照，无基线重复推理。下一步审核正常命中收益与损失、常用词冲突、命令覆盖及背景，形成结论。复用背景时长不累加，Mac host TTS不是P4，C161基准保持不变。无阈值/检查点/校准救援或独立验收访问。\n')
        command('issue-receipt',['gh','issue','comment','18','--repo',REPO,'--body-file',str(body)])
        state('completed-await-review')
        with (R/'4090-c161-envelope-continuation-original-v1.json').open('x') as f:json.dump(dict(identity=identity,stages=stages,status='completed-await-review'),f,indent=2)
    except Exception:
        (ROOT/'failure.log').write_text(traceback.format_exc());state('failed-no-retry');raise
if __name__=='__main__':main()
