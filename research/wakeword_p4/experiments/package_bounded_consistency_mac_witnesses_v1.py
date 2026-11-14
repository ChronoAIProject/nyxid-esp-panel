"""Package existing final models and four prospectively fixed witnesses, no inference."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
R=Path('research/wakeword_p4/results');PREFIX='4090-hey-broad-bounded-consistency-';OUT=Path('.local/wakeword/bounded-consistency-mac-handoff-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def published(p):
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(p)],stderr=subprocess.PIPE)==p.read_bytes(),str(p)
def main():
 names=['formal-execution','formal-result','formal-independent-audit','fixed-grid-compile-plan','fixed-grid-compile-result','fixed-grid-constant-audit','fixed-grid-screen-execution','fixed-grid-screen-result','fixed-grid-screen-independent-audit','formal-evidence-scope']
 reports={n:R/(PREFIX+n+'-v1.json') for n in names}
 # No output directory or model/witness read before complete published reports exist.
 for p in [Path(__file__),*reports.values()]:published(p)
 data={k:json.loads(p.read_text()) for k,p in reports.items()};formal=data['formal-result'];compiled=data['fixed-grid-compile-result'];screen=data['fixed-grid-screen-result'];audit=data['fixed-grid-screen-independent-audit']
 assert formal['all_passed'] and len(formal['cases'])==len(compiled['arms'])==4 and len(screen['cases'])==540
 assert audit['result_sha256']==sha(reports['fixed-grid-screen-result']) and data['formal-independent-audit']['result_sha256']==sha(reports['formal-result'])
 assert formal['execution_sha256']==sha(reports['formal-execution']) and compiled['plan_sha256']==sha(reports['fixed-grid-compile-plan']) and screen['execution_sha256']==sha(reports['fixed-grid-screen-execution'])
 expected={(t,c) for t in ['hey_willow','hey_nyx'] for c in ['raw_logit_control','bounded_logit_consistency']}
 assert {(c['target'],c['condition']) for c in formal['cases']}=={(a['target'],a['condition']) for a in compiled['arms']}==expected
 assert len({(c['target'],c['condition'],c['request_order']) for c in screen['cases']})==540
 files=[]
 def add(path,name,receipt,expected_sha=None):
  p=Path(path);h=sha(p)
  if expected_sha is not None:assert h==expected_sha,str(p)
  commit=subprocess.check_output(['git','log','-1','--format=%H','--',str(receipt)],text=True).strip();assert commit
  files.append((p,dict(name=name,source_path=str(p),bytes=p.stat().st_size,sha256=h,source_receipt=str(receipt),source_receipt_sha256=sha(receipt),source_receipt_commit=commit)))
 for target,condition in sorted(expected):
  arm,=[a for a in compiled['arms'] if (a['target'],a['condition'])==(target,condition)];case,=[c for c in formal['cases'] if (c['target'],c['condition'])==(target,condition)];f=case['final_repeat'];assert case['steps']==f['steps']==10000 and arm['final_h5_sha256']==f['weights_sha256']
  dest=f'models/{target}-{condition}';add(arm['path'],dest+'/hey_nyx.tflite',reports['fixed-grid-compile-result'],arm['sha256']);add(f['weights_path'],dest+'/final.weights.h5',reports['formal-result'],f['weights_sha256'])
  training=Path(f['weights_path']).parent/'training.json';assert json.loads(training.read_text())==f;add(training,dest+'/training.json',reports['formal-result'])
  order=14 if target=='hey_willow' else 48394;w,=[w for w in screen['cases'] if (w['target'],w['condition'],w['request_order'])==(target,condition,order)];add(w['arrays_path'],f'witnesses/{target}-{condition}-request{order}.npz',reports['fixed-grid-screen-result'],w['arrays_sha256'])
 for p in reports.values():add(p,'receipts/'+p.name,p)
 for name in ['run_hey_broad_bounded_consistency_formal_v1.py','hey_broad_bounded_consistency_formal_v1.md','compile_hey_broad_bounded_consistency_fixed_grid_v1.py','screen_hey_broad_bounded_consistency_fixed_grid_v1.py','audit_hey_broad_bounded_consistency_fixed_grid_screen_v1.py']:
  p=Path(__file__).with_name(name);published(p);add(p,'source/'+name,p)
 manifest=dict(schema='gpu-bounded-consistency-fixed-grid-mac-witnesses-v1',files=[r for _,r in files],fixed_requests=dict(hey_willow=14,hey_nyx=48394),limits='Diagnostic original-byte handoff regardless numerical pass/fail. Formal one fresh run/arm, smoke-only repeat evidence. No extra inference, retry, selection, fresh development, acceptance or promotion.')
 assert len({r['name'] for _,r in files})==len(files);OUT.mkdir(exist_ok=False);mp=OUT/'manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n');archive=OUT/'gpu-bounded-consistency-fixed-grid-mac-witnesses-v1.tar.gz'
 with tarfile.open(archive,'w:gz') as t:
  for p,row in files:t.add(p,arcname=row['name'],recursive=False)
  t.add(mp,arcname='manifest.json',recursive=False)
 with tarfile.open(archive) as t:
  assert len(t.getnames())==len(files)+1 and set(t.getnames())=={'manifest.json'}|{r['name'] for _,r in files}
  for _,r in files:
   b=t.extractfile(r['name']).read();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
 print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),payloads=len(files),local_archive_verified=True,anonymous_redownload_required_after_upload=True)))
if __name__=='__main__':main()
