"""Conservative role-conflict check; passing is NOT source/data admission."""
import hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results')
ALIAS=R/'4090-libritts-config-reader-alias-overlap-v1.json'
HISTORY=R/'4090-historical-piper-reader-mapping-v1.json'
def reader_key(voice,speaker_id,aliases):
 for row in aliases['pairs']:
  for family,side in zip(aliases['families'],['left','right'],strict=True):
   entry=row[side]
   if voice==family and speaker_id==entry['conditioning_id']:return 'libritts-reader:'+row['reader_id']
 raise ValueError('Unknown configuration/speaker identity')
def conflicts(proposed,aliases,history):
 # Consumed historical training identities are immutable regardless of proposal.
 roles={'libritts-reader:'+row['reader_id']:{'training'} for row in history['rows']}
 for row in proposed:
  if row['role'] not in {'training','development','acceptance'}:raise ValueError('Unknown role')
  key=reader_key(row['voice'],row['speaker_id'],aliases);roles.setdefault(key,set()).add(row['role'])
 return {k:sorted(v) for k,v in roles.items() if len(v)>1}
def check():
 aliases=json.loads(ALIAS.read_text());history=json.loads(HISTORY.read_text());assert history['training_ids_mapped']==672
 checks=[]
 for family,side in zip(aliases['families'],['left','right'],strict=True):
  mapping={row['reader_id']:row[side]['conditioning_id'] for row in aliases['pairs']}
  for role in ['development','acceptance']:
   proposed=[dict(voice=family,speaker_id=mapping[r['reader_id']],role=role) for r in history['rows']]
   rejected=conflicts(proposed,aliases,history);assert len(rejected)==672
   checks.append(dict(family=family,role=role,rejected_reader_conflicts=len(rejected)))
 training=[dict(voice='en_US-libritts-high',speaker_id=r['libritts_high_conditioning_id'],role='training') for r in history['rows']];assert not conflicts(training,aliases,history)
 try:conflicts([dict(voice='en_US-libritts-high',speaker_id=-1,role='development')],aliases,history)
 except ValueError:unknown=True
 else:raise AssertionError('Unknown ID accepted')
 result=dict(schema='piper-reader-role-conflict-negative-checks-v1',files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ALIAS,HISTORY,Path(__file__)]},negative_checks=checks,unknown_identity_rejected=unknown,matching_training_identity_no_conflict=True,source_admission_created=False,limits='Constructed proposals against actual historical identities. Detector checks known reader aliases only, not whole-family exclusions, other engines/readers/ancestry or complete training eligibility. No-conflict is not admission or permission to generate. No roles assigned or media read.')
 (R/'4090-piper-reader-role-conflict-negative-checks-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(checks))
if __name__=='__main__':check()
