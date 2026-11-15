"""Unchanged official HiESP, all new development PCM, continuous total81600prime."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path('research/wakeword_p4');PLAN=ROOT/'results/4090-melo-official-execution-v2.json'
SOURCE=Path(__file__).with_name('run_development.py')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def register():
 collection=ROOT/'results/4090-melo-development-int8-execution-v2.json';c=json.loads(collection.read_text())
 parent=ROOT/'results/4090-jenny-official-matched-prime-plan-v1.json';prior=json.loads(parent.read_text());runtime=prior['runtime']
 files=dict(c['files']);files.update({str(f):sha(f) for f in [Path(__file__),SOURCE,collection,parent,ROOT/'experiments/score_melo_official_v2.py']+[Path(x) for x in runtime.values()]})
 p=dict(schema='melo-official-execution-v2',files=files,runtime=runtime,records=c['records'],threshold_requests=['DEFAULT'],models=['hiesp'],streams=474,output='.local/wakeword/melo-official-v2',firmware_prime_samples=32256,additional_continuous_zero_samples=49344,total_prime_samples=81600,
  policy='Unchanged official wn9_hiesp DEFAULT DET_MODE_95, fixed getter0.6299999952316284. Same432Melo+42complete background PCM as two fixed candidates/v30, total81600zeroPCM continuous prime, native512block with192pending hostprefix samples. No transition reset or readiness censoring. Preserve every raw return, prime event, UART, stage104, CRC, exact runtime and source identities. QEMU classicESP32 reference, not P4 hardware/numerical equivalence, human phonetic truth or promotion. Acceptance remains unopened.')
 with PLAN.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
 print(json.dumps(dict(streams=474,plan_sha256=sha(PLAN))),flush=True)
def case():
 tree=ast.parse(SOURCE.read_text());counts=dict(plan=0,insert=0,report=0)
 class Adapter(ast.NodeTransformer):
  def visit_Constant(self,n):
   if n.value=='research/wakeword_p4/results/4090-hey-official-development-execution-v1.json':counts['plan']+=1;return ast.copy_location(ast.Constant(str(PLAN)),n)
   return n
  def visit_Expr(self,n):
   self.generic_visit(n)
   if ast.unparse(n).startswith('a.output.mkdir('):
    counts['insert']+=1
    return ast.parse('source_samples=samples\nsource_raw_sha256=hashlib.sha256(raw).hexdigest()\nextra_prime_samples=49344\nraw=bytes(extra_prime_samples*2)+raw\nsamples+=extra_prime_samples').body+[n]
   return n
  def visit_Assign(self,n):
   self.generic_visit(n)
   if any(isinstance(t,ast.Name) and t.id=='report' for t in n.targets):
    counts['report']+=1
    return [n]+ast.parse("report.update(schema='melo-official-original-v2',source_input_samples=source_samples,source_pcm_sha256=source_raw_sha256,input_pcm_sha256=hashlib.sha256(raw).hexdigest(),firmware_prime_samples=32256,additional_prime_samples=extra_prime_samples,total_prime_samples=81600,continuous_model=True,transition_reset=False,real_events=[dict(e,relative_sample=e['sample_end']-extra_prime_samples) for e in events if e['sample_end']>extra_prime_samples],additional_priming_events=[e for e in events if e['sample_end']<=extra_prime_samples])").body
   return n
 tree=Adapter().visit(tree);assert counts==dict(plan=1,insert=1,report=1),counts
 exec(compile(ast.fix_missing_locations(tree),str(SOURCE),'exec'),dict(__name__='__main__',__file__=__file__))
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=json.loads(PLAN.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 subprocess.run([sys.executable,str(ROOT/'tools/check_disk_budget.py'),'--expected-gib','2'],check=True)
 out=Path(p['output']);out.mkdir(exist_ok=False);(out/'launch.json').write_text(json.dumps(dict(pid=os.getpid(),starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(PLAN)))+'\n')
 for i,row in enumerate(p['records']):
  tmp=Path(f'/tmp/melo-hiesp-{i:04}');assert not tmp.exists();cmd=[sys.executable,str(__file__),'case']
  for name,path in p['runtime'].items():cmd+=['--'+name,path]
  cmd+=['--model','hiesp','--threshold','DEFAULT','--record-index',str(i),'--output',str(tmp)]
  with (out/f'{i:04}.log').open('xb') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
  for name in ['report.json','serial.log','stage-monitor.txt','raw-returns.bin']:
   source=tmp/name;dest=out/f'{i:04}-{name}';shutil.copyfile(source,dest);assert sha(source)==sha(dest)
  report=json.loads((out/f'{i:04}-report.json').read_text());assert report['actual_threshold']==0.6299999952316284
  shutil.rmtree(tmp);print(json.dumps(dict(completed=i+1,total=474,phase=row['phase'])),flush=True)
 (out/'complete.json').write_text(json.dumps(dict(streams=474,plan_sha256=sha(PLAN),acceptance_accessed=False))+'\n')
if __name__=='__main__':
 action=sys.argv.pop(1);{'register':register,'case':case,'run':run}[action]()
