"""Attach to existing complete-replay continuation, review and publish findings."""
import json,os,subprocess,time,traceback
from pathlib import Path
from export_binary40_trained_v1 import sha
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/binary40-regularization-finalization-v1')
LOCK=R/'4090-binary40-regularization-finalization-plan-v1.json'
def verify():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
 plan=json.loads(LOCK.read_text())
 for path,h in plan['files'].items():assert sha(path)==h,path
 return plan

def main():
 plan=verify();parent=plan['continuation_pid'];identity=plan['continuation_starttime'];stat=Path(f'/proc/{parent}/stat')
 assert stat.read_text().split()[21]==identity
 B.mkdir(exist_ok=False)
 state=dict(schema='binary40-regularization-finalization-v1',pid=os.getpid(),proc_starttime=Path('/proc/self/stat').read_text().split()[21],parent_pid=parent,parent_starttime=identity,plan_sha256=sha(LOCK),status='waiting-existing-continuation',stages=[])
 def save():(B/'state.json').write_text(json.dumps(state,indent=2)+'\n')
 def stage(name,args):
  verify();log=B/(name+'.log');start=time.monotonic()
  with log.open('x') as f:process=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
  row=dict(name=name,args=args,actual_exit_code=process.returncode,elapsed_seconds=time.monotonic()-start,log=str(log),log_sha256=sha(log));state['stages'].append(row);save();print(json.dumps(row),flush=True)
  assert process.returncode==0,row
 def publish(paths,message):
  subprocess.run(['git','add','--',*[str(p) for p in paths]],check=True)
  subprocess.run(['git','commit','-m',message],check=True)
  subprocess.run(['git','push','origin','research-hye-nyx-voice-gpu'],check=True)
 try:
  save()
  while stat.exists():
   try:fields=stat.read_text().split()
   except FileNotFoundError:break
   if fields[21]!=identity or fields[2]=='Z':break
   time.sleep(5)
  verify();receipt=R/'4090-binary40-regularization-continuation-original-v1.json';r=json.loads(receipt.read_text())
  assert r['pid']==parent and r['status']=='completed-await-independent-review'
  assert r['continuation_plan_sha256']==sha(R/'4090-binary40-regularization-continuation-plan-v1.json')
  assert len(r['stages'])==53 and all(s['actual_exit_code']==0 for s in r['stages'])
  state['status']='reviewing-complete-consumed-evidence';save()
  py='/usr/bin/python3';base='research/wakeword_p4/experiments/'
  stage('independent-consumed-review',[py,base+'review_binary40_regularization_consumed_v1.py'])
  stage('complete-comparison',[py,base+'summarize_binary40_regularization_v1.py'])
  evidence=[R/'4090-binary40-regularization-consumed-review-v1.json',R/'4090-binary40-regularization-final-summary-v1.json',R/'4090-binary40-regularization-final-table-v1.md']
  publish(evidence,'research: publish independently reviewed complete regularization comparison')
  commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();state['result_commit']=commit
  table=evidence[-1].read_text();body=B/'issue18-body.md'
  body.write_text('匹配正则化实验的两臂训练、最终INT8导出和全部16组既有评测已结束；独立归档/成员校验、来源身份及训练核验通过。结果已推送 '+commit+'。\n\n'+table+'\n\n仅 control 与 decay 是本次 AdamW矩阵衰减的匹配因果对照；历史 teacher 使用 Adam。尚无自动晋升或实机合格结论：C161仍为基准，后续需审查逐条普通命中损失、常用词冲突，并结合Mac独立P4数值、首句命令、资源、实时性和稳定性。完整原始报告、逐条得失和独立复核均已保留。\n')
  stage('issue18-update',['gh','issue','comment','18','--repo','ChronoAIProject/nyxid-esp-panel','--body-file',str(body)])
  state['status']='completed-await-research-decision'
 except Exception:
  state['status']='failed-preserved';state['error']=traceback.format_exc();(B/'failure.log').write_text(state['error'])
 finally:
  save();dest=R/'4090-binary40-regularization-finalization-original-v1.json';dest.write_text(json.dumps(state,indent=2)+'\n')
  import tarfile
  archive=R/'4090-binary40-regularization-finalization-logs-v1.tar.gz'
  with tarfile.open(archive,'x:gz') as t:
   for f in sorted(B.iterdir()):
    if f.is_file():t.add(f,arcname=f.name)
  publish([dest,archive],'research: preserve regularization independent review and publication receipts')
 if state['status']!='completed-await-research-decision':raise SystemExit(1)
if __name__=='__main__':main()
