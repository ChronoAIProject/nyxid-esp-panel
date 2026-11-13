"""Bounded tracked-metadata role inventory; no synthesis, audio or model access."""
import hashlib,json,re,subprocess
from pathlib import Path

ROOT=Path('research/wakeword_p4');CATALOG=ROOT/'results/4090-hey-voice-coverage-catalog-v1.json'
OUT=Path('.local/wakeword/hey-voice-role-catalog-audit-v1')
RESULT=ROOT/'results/4090-hey-voice-catalog-role-audit-v1.json'
def sha(data):return hashlib.sha256(data).hexdigest()
def family(key):return key.rsplit('-',1)[0]
def role(value):
    if not isinstance(value,str):return None
    value=value.lower()
    if any(w in value for w in ('acceptance','testing','test-set','final-set')):return 'protected-testing-or-acceptance'
    if any(w in value for w in ('development','validation')):return 'protected-development'
    if value=='training' or value.startswith('training-'):return 'training'
    return None

def main():
    catalog=json.loads(CATALOG.read_text());keys=[r['key'] for r in catalog['english_entries']]
    pattern=re.compile('|'.join(re.escape(k) for k in sorted(keys,key=len,reverse=True)))
    inventory=[];evidence=[]
    paths=subprocess.check_output(['git','ls-files',str(ROOT)],text=True).splitlines()
    paths=[p for p in paths if p.endswith(('.json','.jsonl'))]
    def walk(node,path,where,context=None):
        if isinstance(node,dict):
            current=role(node.get('role')) or role(node.get('split')) or context
            # Only immediate scalar identities are role evidence; do not interpret all descendant text as this voice.
            for key in ('voice','engine','model','model_name','model_path','path','source'):
                value=node.get(key)
                if current and isinstance(value,str):
                    for voice in sorted(set(pattern.findall(value))):
                        evidence.append(dict(path=path,location=where,field=key,voice=voice,family=family(voice),observed_role=current,identity=value))
            for key,value in node.items():walk(value,path,where+'/'+str(key),current)
        elif isinstance(node,list):
            for i,value in enumerate(node):walk(value,path,where+'/'+str(i),context)
    for name in paths:
        data=Path(name).read_bytes();inventory.append(dict(path=name,bytes=len(data),sha256=sha(data)))
        if not pattern.search(data.decode('utf-8',errors='replace')):continue
        if name.endswith('.jsonl'):
            for i,line in enumerate(data.splitlines()):
                if line.strip():walk(json.loads(line),name,'line'+str(i+1))
        else:walk(json.loads(data),name,'')
    reservation=ROOT/'results/4090-hey-paired-voice-role-reservation-v1.json'
    assigned=json.loads(reservation.read_text())['roles']
    for r in assigned:
        assert any(e['voice']==r['voice'] and e['observed_role']==role(r['role']) and e['path']==str(reservation) for e in evidence)
    groups=[]
    for fam in sorted({family(k) for k in keys}):
        found=[e for e in evidence if e['family']==fam];protected=[e for e in found if e['observed_role'].startswith('protected-')]
        groups.append(dict(family=fam,catalog_variants=[k for k in keys if family(k)==fam],role_evidence_count=len(found),protected_evidence_count=len(protected),status='exclude-from-next-training' if protected else 'unresolved-source-and-role-review-required',training_or_generation_authorized=False))
    OUT.mkdir(exist_ok=False)
    for name,data in [('inventory.json',inventory),('role-evidence.json',evidence)]:
        (OUT/name).write_text(json.dumps(data,indent=2)+'\n')
    result=dict(schema='hey-voice-catalog-role-audit-v1',auditor_sha256=sha(Path(__file__).read_bytes()),catalog_sha256=sha(CATALOG.read_bytes()),tracked_metadata_files=len(paths),catalog_entries=len(keys),voice_families=len(groups),families=groups,evidence_path=str(OUT/'role-evidence.json'),evidence_sha256=sha((OUT/'role-evidence.json').read_bytes()),inventory_path=str(OUT/'inventory.json'),inventory_sha256=sha((OUT/'inventory.json').read_bytes()),sources_reserved=False,models_downloaded=False,limits='Bounded current tracked JSON/JSONL structured-role evidence only, not all history/branches/local files or complete voice/physical-speaker/pretraining identity. Absence is unresolved, never admission. Low/medium/high variants share one family. Old Joe/Mike/VCTK and all other protocol exclusions remain mandatory even when this bounded matcher finds none.')
    with RESULT.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(metadata_files=len(paths),families=len(groups),protected=[r['family'] for r in groups if r['protected_evidence_count']],sources_reserved=False)))
if __name__=='__main__':main()
