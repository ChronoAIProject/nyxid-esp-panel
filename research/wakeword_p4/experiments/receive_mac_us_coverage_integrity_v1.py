"""Authenticate immutable Mac report and unchanged-script Linux replay."""
import hashlib,json,shutil
from pathlib import Path
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
root=Path('.local/wakeword/mac-us-coverage-integrity-v1');out=Path('research/wakeword_p4/results')
mac=json.loads((root/'report.json').read_text());linux=json.loads((root/'linux-replay.json').read_text());receipt=json.loads((root/'receipt.json').read_text())
assert sha(root/'report.json')==receipt['audit_report_sha256']=='12ff0b6b3cac3cf70e6743f9e16719fc3f741d82e086278eb744aec9f68cc07f'
assert sha(root/'audit_us_coverage_mac.py')==mac['script_sha256']==linux['script_sha256']=='8f6232deeb1ff97fc8f880671e646547d645020a9acc6a378143d809ffc86afd'
assert sha(root/'plan.json')==mac['plan_sha256']==linux['plan_sha256']
assert mac['input_sha256']==linux['input_sha256'] and mac['host']['versions']==linux['host']['versions']
assert mac['clips']==linux['clips']==108 and not mac['all_process_checks_pass'] and linux['all_process_checks_pass']
assert mac['process_agreement_clips']==0 and linux['process_agreement_clips']==108
changed=0;maxdiff=0.;other=0
for a,b in zip(mac['details'],linux['details']):
 assert a['path']==b['path'] and a['samples']==b['samples'] and a['native_dtype']==b['native_dtype']=='float32' and a['float16k_dtype']==b['float16k_dtype']=='float32'
 assert not a['checks']['resample_exact'] and all(v for k,v in a['checks'].items() if k!='resample_exact')
 assert all(b['checks'].values()) and b['resample_changed_samples']==b['pcm_changed_samples']==0
 assert a['pcm_changed_samples']==0
 changed+=a['resample_changed_samples'];maxdiff=max(maxdiff,a['resample_max_abs_difference']);other+=1
assert changed==194115 and maxdiff==8.940696716308594e-8 and other==108
for name,target in [('report.json','4090-mac-us-coverage-integrity-original-v1.json'),('receipt.json','4090-mac-us-coverage-receipt-original-v1.json'),('linux-replay.json','4090-mac-us-coverage-integrity-linux-replay-v1.json')]:shutil.copyfile(root/name,out/target)
r=dict(scope='Mac original failure report authenticated and same frozen audit script rerun on Linux originals; no model scoring',mac_commit='7936daa',audit_plan_commit='054d7e0',mac_report_sha256=sha(root/'report.json'),mac_receipt_sha256=sha(root/'receipt.json'),linux_replay_sha256=sha(root/'linux-replay.json'),audit_source_sha256=sha(root/'audit_us_coverage_mac.py'),plan_sha256=sha(root/'plan.json'),inputs_identical=True,package_versions_identical=True,mac_strict_process_pass=0,linux_strict_process_pass=108,mac_other_checks_pass=108,mac_changed_resample_samples=changed,mac_max_abs_resample_difference=maxdiff,mac_max_difference_pcm_lsb=maxdiff*32768,original_saved_float_to_pcm_changed_samples_both_hosts=0,script_sha256=sha(Path(__file__)),limits='Mac exact resampling failure preserved, no tolerance relaxation. Same version Python packages do not establish cross-platform bit identity; platform/build/compiler differences not isolated causally. Linux exact replay108/108 corroborates fixed-host conversion, not universal reproducibility. Original Linux float/PCM unchanged and used for v44; no regeneration, training restart, model inference or new-development/final access.')
(out/'4090-mac-us-coverage-integrity-receipt-v1.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
