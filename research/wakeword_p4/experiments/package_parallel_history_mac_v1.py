"""Exact original optional triple handoff; preserve existing dual Release asset."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
R=Path('research/wakeword_p4/results');ROOT=Path('.local/wakeword/parallel-history-mac-handoff-v1');ASSET='c161-binary40-history-mac-handoff-v1.tar.gz'
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def commit(p):return subprocess.check_output(['git','log','-1','--format=%H','--',str(p)],text=True).strip()
def main():
 assert read(R/'4090-parallel-history-portable-check-v1.json')['all_passed'] and read(R/'4090-parallel-history-abi-original-v1.json')['all_passed'];assert all(s['exit_code']==0 for s in read(R/'4090-parallel-history-portable-abi-execution-v1.json')['stages']);ROOT.mkdir(exist_ok=False);files={};ledger=[];head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 def add(name,data,source,revision):
  assert name not in files and not Path(name).is_absolute() and '..' not in Path(name).parts;files[name]=data;ledger.append(dict(path=name,bytes=len(data),sha256=sha(data),source_path=source,source_commit=revision))
 # Authenticate inherited original bytes against the existing dual manifest.
 old=read(R/'4090-c161-binary40-parallel-mac-handoff-manifest-v1.json')
 for spec in old['files']:
  if spec['path']=='README.md':continue
  raw=Path(spec['source_path']).read_bytes();assert len(raw)==spec['bytes'] and sha(raw)==spec['sha256'];add(spec['path'],raw,spec['source_path'],spec['source_commit'])
 export=R/'4090-frozen-history-teacher-teacher-export-original-v1.json';m=read(export)['model'];raw=Path(m['path']).read_bytes();assert len(raw)==64800 and sha(raw)==m['sha256']=='d92b95278334043ff70c20c6e8f267e13fe83acac7dbcf705cc2177df9130950';add('models/history_expert.tflite',raw,m['path'],commit(export))
 names=['results/4090-frozen-history-teacher-teacher-training-original-v1.json','results/4090-frozen-history-teacher-teacher-export-original-v1.json','results/4090-frozen-history-teacher-teacher-export-plan-v1.json','results/4090-parallel-history-expert-runtime-original-v1.json','results/4090-parallel-history-expert-runtime-review-v1.json','results/4090-parallel-history-expert-runtime-execution-v1.json','results/4090-parallel-history-portable-plan-v1.json','results/4090-parallel-history-portable-check-v1.json','results/4090-parallel-history-portable-scores-v1.bin.gz','results/4090-parallel-history-abi-plan-v1.json','results/4090-parallel-history-abi-original-v1.json','results/4090-parallel-history-portable-abi-execution-v1.json','results/4090-c161-binary40-static-abi-original-v1.json','results/4090-c161-binary40-static-abi-review-v1.json','portable/c161_binary40_history_events_v1.h','portable/check_c161_binary40_history_events_v1.cpp','portable/c161_binary40_history_policy_readme_v1.md','experiments/parallel_history_expert_scope_findings_v1.md','experiments/parallel_history_expert_preflight_findings_v1.md','experiments/parallel_history_expert_runtime_findings_v1.md','experiments/parallel_history_portable_abi_findings_v1.md']
 for name in names:
  p=Path('research/wakeword_p4')/name;revision=commit(p);assert revision and subprocess.check_output(['git','show',revision+':'+str(p)])==p.read_bytes();add(str(p),p.read_bytes(),str(p),revision)
 text='''# Optional fixed triple research handoff\n\nThree exact original models: C161, original Binary40 and original frozen-history-teacher treatment (not control). Existing published dual-model handoff remains unchanged; this optional package is not a promotion or proof of P4 feasibility.\n\nRead research/wakeword_p4/portable/c161_binary40_history_policy_readme_v1.md first. It documents independent caches/input quantization, signed+128 probability decoding, fixed235/256/smoothing5/67call policy, tie order and score-only C++ self-test. Both the inherited dual policy and new triple policy/fixtures are included, clearly separately named. The old dual findings describe the earlier candidate; triple findings and readme describe this optional candidate.\n\nAll1763 host streams/3961012calls and independent review passed; portable C++ events exactly match every call. No PCM or Linux executable is shipped. Mac generates PCM self-tests, verifies actual P4 outputs and measures all three resident arenas, full-panel internal/PSRAM peaks, real-time scheduling, first-command retention and stability. Use Issue#18 with firmware commit/model SHA/profile.\n\nModel files190552bytes and retained-state logical payload19108bytes exclude arena/scratch/resource/frontend/audio/panel overhead. C161 resource lifecycle and listed op versions require device support. Host allocation/timing is not P4 evidence.\n\nConsumed synthetic diagnostics remain imperfect: new context builtin11/12,US11/12,BR9/12; Joe23/72,MeloBR25/36,Mac host2/4. Added Muse/Mix/Nick non-target clips, dailyNick1/96 and reused Melo background1event remain disclosed; no new FAR exposure, human-generalization claim or guarantee all baseline events are preserved on unseen streams. Rare constructed nearwords are diagnostic, common-name Nick remains practical risk.\n\nmanifest.json records exact size/SHA/source commit for all other members. Model source commits identify original export publications for locally preserved model bytes; no retraining/reexport. Full raw triple evidence archive stays on research branch (not duplicated here).\n'''
 add('README.md',text.encode(),'generated handoff metadata',head)
 manifest=dict(schema='c161-binary40-history-mac-handoff-v1',source_commit=head,baseline='original C161',role='optional fixed triple research candidate; preserve original dual handoff',profile=dict(threshold=.91796875,smoothing_steps=5,refractory_calls=67,priority=['c161','binary40','history_expert']),files=ledger)
 encoded=(json.dumps(manifest,indent=2)+'\n').encode();files['manifest.json']=encoded
 archive=ROOT/ASSET
 with tarfile.open(archive,'x:gz') as tf:
  for name,raw in files.items():
   t=tarfile.TarInfo(name);t.size=len(raw);t.mode=0o644;t.mtime=0;tf.addfile(t,io.BytesIO(raw))
 with tarfile.open(archive,'r:gz') as tf:
  assert len(tf.getnames())==len(set(tf.getnames())) and set(tf.getnames())==set(files)
  for name,raw in files.items():assert tf.extractfile(name).read()==raw
 with (R/'4090-parallel-history-mac-handoff-manifest-v1.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
 receipt=dict(schema='parallel-history-mac-handoff-package-v1',archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive.read_bytes())),members=len(files),all_original_members_exact=True,manifest_sha256=sha(encoded))
 with (R/'4090-parallel-history-mac-handoff-package-v1.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
