"""Observe the locked runs without restarting; score only complete receipts."""
import gzip
import json
from pathlib import Path
import subprocess
import time
from collect_melo_development_v2 import OUT,sha
from melo_synthetic_viability_v2 import assess
ROOT=Path('research/wakeword_p4');R=ROOT/'results'

def wait_for(path,launch,failure):
 while not path.exists():
  if failure.exists():raise RuntimeError(failure.read_text())
  identity=json.loads(launch.read_text());pid=identity['pid']
  proc=Path(f'/proc/{pid}/stat')
  if not proc.exists() or proc.read_text().split()[21]!=identity['starttime']:
   raise RuntimeError(f'Original process {pid} stopped without {path}; no restart')
  time.sleep(10)

def main():
 wait_for(OUT/'result.json',OUT/'launch.json',OUT/'failure.log')
 subprocess.run(['.local/wakeword/melo-runtime-v1/bin/python',str(ROOT/'experiments/score_melo_development_v2.py')],check=True)
 metrics=OUT/'metrics';joint=metrics/'joint-selection.json'
 subprocess.run(['/usr/bin/python3',str(ROOT/'experiments/select_melo_joint_development_v2.py'),'--c161',str(metrics/'c161-table.json'),'--speecht5',str(metrics/'speecht5-final20000-table.json'),'--output',str(joint)],check=True)
 official=Path('.local/wakeword/melo-official-v2')
 wait_for(official/'complete.json',official/'launch.json',official/'failure.log')
 subprocess.run(['/usr/bin/python3',str(ROOT/'experiments/score_melo_official_v2.py')],check=True)
 selected=json.loads(joint.read_text());v30=json.loads((metrics/'v30-frozen-table.json').read_text());reference=json.loads((official/'point.json').read_text())
 if selected['selected_candidate'] is None:
  outcome=dict(passed=False,reason='No background-feasible development point')
 else:outcome=assess(selected['selected_point'],v30['points'][0],reference['point'],v30['background_hours'])
 report=dict(schema='melo-joint-development-findings-v2',selection=selected,viability=outcome,official=reference,v30=v30,acceptance_accessed=False,promotion_qualified=False,v30_retained=True,
  inputs={str(p):sha(p) for p in [joint,metrics/'completed.json',official/'complete.json',official/'point.json']},
  next_action='Separately lock independent acceptance before generating it' if outcome['passed'] else 'Development failed; preserve all evidence, do not open acceptance or fall back to runner-up')
 path=R/'4090-melo-joint-development-findings-v2.json'
 with path.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 # Preserve exact tables and every signed event detail; no filtered successes.
 paths=[path]
 for source in sorted(metrics.iterdir()):
  if source.is_file():
   dest=R/('4090-melo-development-v2-'+source.name+( '' if source.suffix=='.gz' else '.gz'))
   with dest.open('xb') as raw:
    if source.suffix=='.gz':raw.write(source.read_bytes())
    else:
     with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as f:f.write(source.read_bytes())
   paths.append(dest)
 for name in ['point.json','event-details.jsonl.gz']:
  dest=R/('4090-melo-official-v2-'+name+('' if name.endswith('.gz') else '.gz'))
  with dest.open('xb') as raw:
   if name.endswith('.gz'):raw.write((official/name).read_bytes())
   else:
    with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as f:f.write((official/name).read_bytes())
  paths.append(dest)
 subprocess.run(['git','add','--',*map(str,paths)],check=True)
 subprocess.run(['git','commit','-m','research: preserve complete fresh Melo joint development findings'],check=True)
 subprocess.run(['git','push','origin','HEAD'],check=True)
 print(json.dumps(report),flush=True)

if __name__=='__main__':main()
