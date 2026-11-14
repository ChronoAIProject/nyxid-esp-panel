"""Validate prospective synthesis reservations; never generate or score audio."""
import argparse,hashlib,json,math
from pathlib import Path
ROLES={'training','development','acceptance'}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(plan):
 assert plan['schema']=='synthetic-role-reservation-v1'
 ledger=Path(plan['consumed_ledger']['path']);assert digest(ledger)==plan['consumed_ledger']['sha256']
 consumed=json.loads(ledger.read_text());assert consumed['schema']=='synthetic-consumed-identities-v1'
 old={k:set(consumed[k]) for k in ['voice_ids','model_sha256','physical_speaker_ids']}
 assert plan['consumed_ledger']['coverage_limits'].strip()
 seen={};ids=set();counts={r:0 for r in ROLES}
 assert plan['requests']
 for row in plan['requests']:
  assert row['role'] in ROLES and row['request_id'] not in ids;ids.add(row['request_id']);counts[row['role']]+=1
  assert row['text'].strip() and row['pronunciation_input'] and row['voice_id'] and row['upstream_revision'] and row['upstream_url']
  assert row['runtime_versions'] and all(isinstance(v,str) and v for v in row['runtime_versions'].values())
  assert isinstance(row['native_rate'],int) and row['native_rate']>0
  assert row['native_return_saved_before_transform']is True and row['trimmed']is False
  assert row['duration_metadata_interface'] and row['phoneme_metadata_interface']
  assert row['physical_speaker_id'] is None or isinstance(row['physical_speaker_id'],str)
  assert row['pretraining_overlap_status'] in ['unknown','documented_overlap','documented_disjoint']
  assert all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in row['scales'].values()) and row['scales']
  for name in ['model','config']:
   assert digest(row[name]['path'])==row[name]['sha256']
  identities=[('voice_ids',row['voice_id']),('model_sha256',row['model']['sha256'])]
  if row['physical_speaker_id']:identities.append(('physical_speaker_ids',row['physical_speaker_id']))
  for kind,value in identities:
   key=(kind,value)
   if key in seen:assert seen[key]==row['role'],'Cross-role source identity: '+str(key)
   seen[key]=row['role']
   if row['role']!='training':assert value not in old[kind],'Consumed holdout identity: '+str(key)
 assert all(counts.values()),'Reserve all three roles explicitly'
 assert plan['failure_policy']=='retain-all-stop-pack-no-substitution'
 return dict(all_mechanical_reservation_checks_passed=True,counts=counts,requests=len(ids),limits='Validates declared identities and authenticated files/ledger only. Ledger coverage, physical-speaker identity and upstream claims require independent provenance review. Unknown pretraining overlap remains unknown. No audio quality, phonetic truth, generation, detector calls or independent acceptance claim.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('plan');a=p.parse_args();print(json.dumps(audit(json.loads(Path(a.plan).read_text())),indent=2))
