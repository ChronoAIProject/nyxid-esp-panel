"""Attach to original runtime PID/starttime, then independently review/publish."""
import json,os,subprocess,time,traceback,hashlib
from pathlib import Path
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/c161-binary40-parallel-runtime-finalization-v1');LOCK=R/'4090-c161-binary40-parallel-runtime-finalization-plan-v1.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();plan=json.loads(LOCK.read_text())
 for f,h in plan['files'].items():assert sha(f)==h,f
 return plan
def main():
 plan=verify();pid=plan['runtime_pid'];identity=plan['runtime_starttime'];stat=Path(f'/proc/{pid}/stat');assert stat.read_text().split()[21]==identity
 B.mkdir(exist_ok=False);state=dict(pid=os.getpid(),starttime=Path('/proc/self/stat').read_text().split()[21],status='waiting-existing-runtime',plan_sha256=sha(LOCK),stages=[])
 def save():(B/'state.json').write_text(json.dumps(state,indent=2)+'\n')
 def publish(paths,message):
  subprocess.run(['git','add','--',*[str(p) for p in paths]],check=True);subprocess.run(['git','commit','-m',message],check=True);subprocess.run(['git','push','origin','research-hye-nyx-voice-gpu'],check=True)
 try:
  save()
  while stat.exists():
   try:s=stat.read_text().split()
   except FileNotFoundError:break
   if s[21]!=identity or s[2]=='Z':break
   time.sleep(5)
  receipt=R/'4090-c161-binary40-parallel-runtime-execution-v1.json'
  for attempt in range(12):
   if receipt.exists():break
   time.sleep(5)
  rc=json.loads(receipt.read_text());assert rc['pid']==pid and rc['actual_exit_code']==0
  verify();state['status']='independent-review';save();log=B/'review.log';args=['/usr/bin/python3','research/wakeword_p4/experiments/review_c161_binary40_parallel_runtime_v1.py'];started=time.monotonic()
  with log.open('x') as f:process=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
  state['stages'].append(dict(args=args,actual_exit_code=process.returncode,elapsed_seconds=time.monotonic()-started,log=str(log),log_sha256=sha(log)));save();assert process.returncode==0
  evidence=[receipt,R/'4090-c161-binary40-parallel-runtime-original-v1.json',R/'4090-c161-binary40-parallel-runtime-review-v1.json',R/'4090-c161-binary40-parallel-runtime-evidence-v1.tar.gz'];publish(evidence,'research: preserve complete exact dual interpreter replay and independent review')
  commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();body=B/'issue18.md';body.write_text('固定C161+原Binary40的实际双解释器完整8cohort/1760输入回放已结束，真实exit0；独立全部双轨迹codes/clock/source及事件/共享67call冷却重算、归档成员校验通过，推送 '+commit+'。\n\n逐模型输出和实际在线合并/抑制事件均逐元素复现先前保存事件模拟，因而此前互补计数已通过主机实际双解释器执行验证。仍有Nick日常冲突及Melo背景事件，全部保留；不新增独立曝光，不调阈值，不自动晋升。\n\n这是BUILTIN_REF各1thread主机执行，不代表P4算力/arena/整机资源、首句保留或稳定性通过。Mac仍需独立验证这两原始模型及固定仲裁策略的数值与资源。C161继续基准。\n')
  process=subprocess.run(['gh','issue','comment','18','--repo','ChronoAIProject/nyxid-esp-panel','--body-file',str(body)],capture_output=True,text=True);state['issue_update']=dict(actual_exit_code=process.returncode,stdout=process.stdout,stderr=process.stderr);assert process.returncode==0;state['status']='completed-await-mac-p4-and-research-decision'
 except Exception:state['status']='failed-preserved';state['error']=traceback.format_exc()
 finally:
  save();dest=R/'4090-c161-binary40-parallel-runtime-finalization-original-v1.json';dest.write_text(json.dumps(state,indent=2)+'\n');publish([dest],'research: preserve dual runtime independent review publication receipts')
 if state['status']!='completed-await-mac-p4-and-research-decision':raise SystemExit(1)
if __name__=='__main__':main()
