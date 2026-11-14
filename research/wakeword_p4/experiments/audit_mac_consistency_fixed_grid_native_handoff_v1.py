"""Read-only independent comparison of authenticated Mac and original GPU arrays."""
import json,hashlib,tarfile,io
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');ARCHIVE=Path('.local/wakeword/mac-consistency-fixed-grid-native-import-v1/original.tar.gz')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(data):return hashlib.sha256(data).hexdigest()
def main():
 assert sha(ARCHIVE)=='d99e0125145976650e7bce3be43078b16df9c1df8573f73a4b5bd0c568a49a06'
 screen=R/'4090-hey-broad-consistency-fixed-grid-screen-result-v1.json';recursive=R/'4090-hey-broad-consistency-fixed-grid-recursive-result-v1.json';models=json.loads((R/'4090-hey-broad-consistency-fixed-grid-compile-result-v1.json').read_text())['arms'];gpurows=json.loads(screen.read_text())['cases'];refs=json.loads(recursive.read_text())['cases'];cases=[]
 with tarfile.open(ARCHIVE,'r:gz') as tar:
  manifest=json.load(tar.extractfile('manifest.json'));assert len(manifest['files'])==15;assert len(tar.getnames())==16 and set(tar.getnames())=={'manifest.json',*[r['name'] for r in manifest['files']]}
  for row in manifest['files']:
   member=tar.getmember(row['name']);assert member.isfile();data=tar.extractfile(member).read();assert len(data)==row['bytes'] and digest(data)==row['sha256']
  manifestbytes=tar.extractfile('manifest.json').read();(R/'4090-mac-consistency-fixed-grid-native-import-original-manifest-v1.json').write_bytes(manifestbytes)
  reportbytes=tar.extractfile('reports/mac-consistency-fixed-grid-native-four-witness-replay-v1.json').read();report=json.loads(reportbytes);planbytes=tar.extractfile('source/results/mac-consistency-fixed-grid-witness-replay-plan-v1.json').read();assert digest(planbytes)=='c9843db3c403923da4c5b08d22590ddf645c3101a1f22e785e93af284090d594';assert report['plan_sha256']==digest(planbytes);assert report['script_sha256']==digest(tar.extractfile('source/experiments/replay_consistency_fixed_grid_witnesses_mac_v1.py').read());assert len(report['cases'])==4
  for c in report['cases']:
   key=(c['target'],c['condition'],c['request_order']);g,=[r for r in gpurows if (r['target'],r['condition'],r['request_order'])==key];reference,=[r for r in refs if (r['target'],r['condition'],r['request_order'])==key];model,=[r for r in models if (r['target'],r['condition'])==key[:2]]
   assert model['sha256']==c['model_sha256'] and model['final_h5_sha256']==c['final_h5_sha256'];assert sha(g['arrays_path'])==g['arrays_sha256'] and sha(reference['arrays_path'])==reference['arrays_sha256'];data=tar.extractfile('arrays/'+c['arrays']).read();assert digest(data)==c['arrays_sha256']
   with np.load(io.BytesIO(data),allow_pickle=False) as a,np.load(g['arrays_path'],allow_pickle=False) as b,np.load(reference['arrays_path'],allow_pickle=False) as ref:
    for field in ['raw','quantized_input','uint8']:assert np.array_equal(a[field],b[field])
    for field,repeat in [('float','float_repeat'),('uint8','uint8_repeat'),('embedding','embedding_repeat'),('actual_fc','actual_fc_repeat')]:assert np.array_equal(a[field],a[repeat])
    assert np.array_equal(a['actual_fc'],a['registered_fc']) and np.array_equal(a['embedding'],ref['64'].reshape(-1,1800)) and np.array_equal(a['actual_fc'],ref['actual_65'].reshape(-1));assert len(a['uint8'])==c['steps']
    scale,zero=g['output_quantization'];signed=(a['uint8'].astype(np.float64)-zero)*scale-a['float'];five=np.abs(np.convolve(signed,np.ones(5)/5,'valid'));stats=dict(mean=float(five.mean()),p99=float(np.quantile(five,.99)),maximum=float(five.max()));phase=float(np.max(np.abs(a['offline_phase']-a['float'][74:])))
    for name,value in stats.items():assert np.isclose(value,c['smoothed_abs_error'][name],rtol=0,atol=1e-14)
    assert phase==c['float_offline_phase_max_abs'];passed=bool(phase<=1e-4 and stats['mean']<=.02 and stats['p99']<=.1 and stats['maximum']<=.25);assert passed==c['witness_meets_numerical_bounds']==g['passed']
    receipt=c['actual_registration_receipt'];assert receipt['prepare_status']==receipt['invoke_status']==receipt['temporary_count']==0
    cases.append(dict(target=c['target'],condition=c['condition'],request_order=c['request_order'],steps=c['steps'],input_codes_output_embedding_fc_cross_host_exact=True,all_saved_repeats_exact=True,mac_numerical_passed=passed,gpu_numerical_passed=g['passed'],mac_five_stats=stats,cross_host_float_max_abs=float(np.max(np.abs(a['float']-b['float']))),cross_host_phase_max_abs=float(np.max(np.abs(a['offline_phase']-b['offline_phase'])))))
  dest=R/'4090-mac-consistency-fixed-grid-native-import-original-report-v1.json'
  with dest.open('xb') as f:f.write(reportbytes)
 result=dict(archive_url='https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/mac-consistency-fixed-grid-native-four-witness-replay-v1.tar.gz',archive_sha256=sha(ARCHIVE),archive_bytes=ARCHIVE.stat().st_size,anonymous_download=True,verified_payloads=15,original_mac_report_sha256=digest(reportbytes),gpu_screen_sha256=sha(screen),gpu_recursive_sha256=sha(recursive),cases=cases,total_steps=sum(c['steps'] for c in cases),limits='Read-only saved-array crosscheck, no model invoke/export/training. Registered FC execution remains original Mac source/log evidence; independently verifies its saved codes vs both Mac actualFC and original GPU recursive actualFC. Four consumed predetermined witnesses only; all540 models remain rejected, no newdev/P4wholemodel/resource/human-generalization admission.')
 with (R/'4090-mac-consistency-fixed-grid-native-linux-cross-audit-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
