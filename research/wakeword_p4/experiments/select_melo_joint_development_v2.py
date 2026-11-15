"""Exactly two preregistered candidate tables; no inference or promotion."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
from select_unconsumed_threshold_v1 import select
MODELS=('c161','speecht5-final20000')
STRATA={'EN-US/bare':9,'EN-US/command':27,'EN-BR/bare':9,'EN-BR/command':27}
C161='c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81'

def joint_select(tables):
 if set(tables)!=set(MODELS):raise ValueError('Exactly the two preregistered models required')
 common=None;results={};eligible=[]
 for name in MODELS:
  t=tables[name]
  if t['candidate_id']!=name:raise ValueError('Candidate identity mismatch')
  for field in ['model_sha256','population_sha256','background_manifest_sha256']:
   if not re.fullmatch('[0-9a-f]{64}',t[field]):raise ValueError('Invalid identity hash')
  if name=='c161' and t['model_sha256']!=C161:raise ValueError('Changed c161 model')
  signature=(t['population_sha256'],t['background_manifest_sha256'],t['background_hours'])
  if common is None:common=signature
  if common!=signature:raise ValueError('Different source population/exposure between models')
  if set(t['expected_positive_strata'])!=set(STRATA):raise ValueError('Wrong prospective strata')
  for point in t['points']:
   if {s:v['clips'] for s,v in point['positive'].items()}!=STRATA or point['confusion_clips']!=288:raise ValueError('Changed prospective denominators')
  result=select(t);results[name]=dict(result,model_sha256=t['model_sha256'])
  if result['selected_threshold'] is None:continue
  point=next(p for p in t['points'] if p['threshold']==result['selected_threshold'])
  worst=min(Fraction(v['hits'],v['clips']) for v in point['positive'].values())
  total=Fraction(sum(v['hits'] for v in point['positive'].values()),72)
  rank=(worst,total,-Fraction(point['confusion_activated_clips'],288),point['threshold'],name=='c161')
  eligible.append((rank,name,point))
 if not eligible:return dict(status='no_feasible_background_point',selected_candidate=None,per_candidate=results,promotion_qualified=False)
 _,name,point=max(eligible,key=lambda x:x[0])
 return dict(status='joint_development_point_selected',selected_candidate=name,selected_threshold=point['threshold'],selected_model_sha256=tables[name]['model_sha256'],selected_point=point,per_candidate=results,promotion_qualified=False,viability_checked=False,limits='Fixed two-model development selection only. No runner-up fallback after failed viability; independent acceptance and fixed reference gates remain required. Repeated models/grid do not add background exposure.')

def main():
 p=argparse.ArgumentParser();p.add_argument('--c161',type=Path,required=True);p.add_argument('--speecht5',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 paths=dict(zip(MODELS,[a.c161,a.speecht5],strict=True));tables={n:json.loads(path.read_text()) for n,path in paths.items()};result=joint_select(tables)
 result.update(tables={n:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for n,path in paths.items()},selector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 with a.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
