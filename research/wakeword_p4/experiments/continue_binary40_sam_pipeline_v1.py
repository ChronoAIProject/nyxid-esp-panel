"""Resume a live SAM run through authenticated final-only downstream stages.

No training restarts, rescue exports, model promotion or heldout access.
"""
import gzip,json,os,subprocess,sys,time
from pathlib import Path
R=Path('research/wakeword_p4/results');E=Path('research/wakeword_p4/experiments')
ENV=os.environ.copy();ENV.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
COHORTS=['builtin','daily','continuous','primed','jenny','joe','melo','mac']
def read(p):return json.loads(Path(p).read_text())
def python(script,*args):return ['/usr/bin/python3','research/wakeword_p4/training/gpu_python.py',str(E/script),*args]
def publish(paths,message):
 assert paths and all(Path(p).is_file() for p in paths)
 subprocess.run(['git','add','--',*map(str,paths)],check=True,env=ENV)
 subprocess.run(['git','commit','-m',message,'--',*map(str,paths)],check=True,env=ENV)
 subprocess.run(['git','push','origin','research-hye-nyx-voice-gpu'],check=True,env=ENV)
def stage(name,cmd):
 receipt=R/f'4090-binary40-sam-{name}-execution-v1.json'
 assert not receipt.exists()
 start=time.time()
 with Path(f'.local/wakeword/binary40-sam-pipeline-{name}-v1.log').open('x') as log:
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=ENV);code=p.wait()
 with receipt.open('x') as f:json.dump(dict(command=cmd,pid=p.pid,actual_exit_code=code,start=start,end=time.time()),f,indent=2);f.write('\n')
 assert code==0,(name,code)
 return receipt
def main():
 assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='research-hye-nyx-voice-gpu'
 # Wait for an already registered reviewer, authenticating its process identity.
 launch=read('.local/wakeword/binary40-sam-pipeline-launch-v1.json')
 proc=Path('/proc')/str(launch['review_supervisor_pid'])/'stat'
 expected=launch['review_supervisor_starttime'];ready=R/'4090-binary40-sam-training-review-execution-v1.json'
 while not ready.exists():
  assert proc.exists(),'Reviewer handle missing; no restart'
  assert proc.read_text().rsplit(')',1)[1].split()[19]==expected
  time.sleep(15)
 assert read(ready)['actual_exit_code']==0
 review=R/'4090-binary40-sam-training-review-v1.json'
 assert read(review)['all_sampling_checks_passed'] and read(review)['steps_verified']==20000
 publish([ready,review,R/'4090-binary40-sam-training-receipt-v1.json',R/'4090-binary40-sam-sam-training-original-v1.json'],'research: preserve completed SAM training and exhaustive ledger review')
 subprocess.run(python('export_binary40_sam_final_v1.py','plan'),check=True,env=ENV)
 export_plan=R/'4090-binary40-sam-sam-export-plan-v1.json'
 publish([export_plan],'research: lock final SAM INT8 export after completed training audit')
 export_receipt=stage('export',python('export_binary40_sam_final_v1.py','run'))
 export_report=R/'4090-binary40-sam-sam-export-original-v1.json';export=read(export_report)
 assert export['all_neural_tensors_integer'] and export['fresh_reset_exact']
 model=Path(export['model']['path']);packed=R/'4090-binary40-sam-sam-int8-original-v1.tflite.gz'
 with packed.open('xb') as target,gzip.GzipFile(filename='',mode='wb',fileobj=target,mtime=0) as gz:gz.write(model.read_bytes())
 assert gzip.decompress(packed.read_bytes())==model.read_bytes()
 publish([export_receipt,export_report,packed],'research: preserve original final SAM INT8 model and export evidence')
 subprocess.run(python('run_binary40_sam_evaluation_v1.py','plan'),check=True,env=ENV)
 plans=[R/f'4090-binary40-sam-sam-{c}-replay-plan-v1.json' for c in COHORTS]+[R/'4090-binary40-sam-sam-context-plan-v1.json']
 publish(plans,'research: lock all eight complete SAM consumed cohorts and three contexts')
 execution=stage('full-evaluation',python('run_binary40_sam_evaluation_v1.py','run'))
 independent=stage('consumed-independent-review',python('review_binary40_sam_consumed_v1.py'))
 assert read(R/'4090-binary40-sam-consumed-review-v1.json')['all_passed']
 paths=[execution,independent,R/'4090-binary40-sam-evaluation-receipts-v1.json',R/'4090-binary40-sam-consumed-review-v1.json',R/'4090-binary40-sam-sam-context-original-v1.json']
 for c in COHORTS:paths.extend([R/f'4090-binary40-sam-sam-{c}-comparison-v1.json',R/f'4090-binary40-sam-sam-{c}-evidence-v1.tar.gz'])
 publish(paths,'research: publish full SAM consumed evaluation and independent completeness audit')
 body=Path('/tmp/nyx-sam-pipeline-completed-v1.md')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 body.write_text('SAM固定20000步训练、全批账本复建、单次最终INT8导出及8完整cohort/3context回放和独立审核均实际exit0，原报告/模型/逐条证据已推送 '+commit+'。详见4090-binary40-sam-consumed-review-v1.json及各comparison；尚未作候选升级结论，C161及Mac固件保持。复用背景非新增暴露，未访问留出、未调阈值或重试失败配方。下一步解读逐句正常命中/常用词冲突与C161得失，不能以训练loss或单项误报宣布成功。\n')
 subprocess.run(['gh','issue','comment','18','--repo','ChronoAIProject/nyxid-esp-panel','--body-file',str(body)],check=True,env=ENV)
if __name__=='__main__':main()
