"""Original research bytes only; public archive and every member verification."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword');ARCHIVE=B/'matchbox-long-research-evidence-v1.tar.gz';MANIFEST=R/'4090-matchbox-long-evidence-manifest-v1.json';RECEIPT=R/'4090-matchbox-long-evidence-publication-v1.json'
URL='https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/'+ARCHIVE.name

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pack():
 subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','.3'],check=True)
 for arm in ['control','pooled']:
  assert json.loads((R/f'4090-matchbox-long-{arm}-consumed-melo-original-v1.json').read_text())['all474streams_complete']
 commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();paths=set()
 directories=[B/n for n in ['matchbox-long-control-v1','matchbox-long-control-recovery-v1','matchbox-long-control-resumed-v1','matchbox-long-pooled-recovery-v1','matchbox-long-material-v1','matchbox-long-control-int8-v1','matchbox-long-pooled-int8-v1','matchbox-long-control-consumed-melo-v1','matchbox-long-pooled-consumed-melo-v1']]
 for directory in directories:paths.update(p for p in directory.rglob('*') if p.is_file())
 paths.update(R.glob('4090-matchbox-long-*.json'));paths.update(R.glob('4090-matchbox-long-*.gz'));paths.update(R.glob('4090-finite-pcm-long-*.json'))
 names=['matchbox_long_findings_v1.md','matchbox_long_protocol_v1.md','train_matchbox_long_v1.py','matchbox_long_batch_v1.py','prepare_matchbox_long_sources_v1.py','recover_matchbox_long_training_v1.py','resume_matchbox_long_control_v1.py','export_matchbox_long_recovered_v1.py','collect_matchbox_long_recovered_melo_v1.py','finite_pcm_long_window_v1.py','uint8_event_policy_v1.py','compare_matchbox_long_behavior_v1.py','audit_matchbox_long_execution_v1.py']
 paths.update(Path('research/wakeword_p4/experiments')/n for n in names);paths.update(Path('research/wakeword_p4/training')/n for n in ['matchboxnet_pretrained_v1.py','matchboxnet_long_window_v1.py'])
 paths.add(B/'compact-pretrained-feasibility-v1/matchboxnet3x1x64-v2-1.0.0rc1.nemo');paths.add(Path(__file__))
 assert all(p.suffix not in ['.wav','.flac','.pcm'] for p in paths)
 rows=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),source_commit=commit,source_kind='original-local-research-artifact' if str(p).startswith('.local/') else 'repository-path-research-metadata-or-source') for p in sorted(paths)]
 manifest=dict(schema='matchbox-long-evidence-manifest-v1',source_commit=commit,files=rows,research_only=True,promotion_qualified=False,limits='Failed/unqualified research evidence only, no PCM audio or P4 flash handoff. Both originalfinalmodels, originaltraining/restart/checkpoints/optimizer/RNG/representatives/sourcehashes/all948traces and metadata. Commit identifies packaging source state; local tensor provenance is bound by originalreports, not a claim that gitstoresweights. Interruptedgzip remains originalincomplete bytes.')
 with MANIFEST.open('x') as f:f.write(json.dumps(manifest,indent=2)+'\n')
 with tarfile.open(ARCHIVE,'w:gz') as tar:
  for p in sorted(paths):tar.add(p,arcname=str(p),recursive=False)
  tar.add(MANIFEST,arcname='manifest.json')
 receipt=dict(schema='matchbox-long-evidence-publication-v1',path=str(ARCHIVE),bytes=ARCHIVE.stat().st_size,sha256=sha(ARCHIVE),manifest_sha256=sha(MANIFEST),members=len(rows)+1,source_commit=commit,url=URL,uploaded=False,anonymous_verified=False,research_only=True,promotion_qualified=False)
 RECEIPT.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
def publish():
 receipt=json.loads(RECEIPT.read_text());assert sha(ARCHIVE)==receipt['sha256']
 subprocess.run(['gh','release','upload','research-hey-nyx-20261110',str(ARCHIVE),'--repo','ChronoAIProject/nyxid-esp-panel'],check=True)
 receipt['uploaded']=True;RECEIPT.write_text(json.dumps(receipt,indent=2)+'\n')
def verify():
 receipt=json.loads(RECEIPT.read_text());assert receipt['uploaded'];download=B/'matchbox-long-evidence-anonymous-download-v1.tar.gz';assert not download.exists()
 subprocess.run(['curl','-q','--fail','--location','--silent','--show-error',URL,'--output',str(download)],check=True)
 assert download.stat().st_size==receipt['bytes'] and sha(download)==receipt['sha256']
 manifest=json.loads(MANIFEST.read_text());expected={r['path']:r for r in manifest['files']}
 with tarfile.open(download,'r:gz') as tar:
  members=tar.getmembers();assert len(members)==receipt['members'] and len({m.name for m in members})==len(members)
  assert set(m.name for m in members)==set(expected)|{'manifest.json'}
  assert tar.extractfile('manifest.json').read()==MANIFEST.read_bytes()
  for m in members:
   assert m.isfile()
   if m.name=='manifest.json':continue
   row=expected[m.name];assert m.size==row['bytes'] and hashlib.file_digest(tar.extractfile(m),'sha256').hexdigest()==row['sha256'],m.name
 receipt.update(anonymous_verified=True,verified_members=len(members),all_member_hashes_passed=True,anonymous_duplicate_deleted=True,manifest_repository_path=str(MANIFEST));download.unlink();RECEIPT.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['pack','publish','verify']);a=p.parse_args();dict(pack=pack,publish=publish,verify=verify)[a.action]()
