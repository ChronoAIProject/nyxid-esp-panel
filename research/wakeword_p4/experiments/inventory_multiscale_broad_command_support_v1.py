"""Existing data availability versus active sampler support, metadata only."""
import collections,hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results');paths=[Path('.local/wakeword/hey-broad-training-features-v1/hey_nyx/features.json'),R/'4090-causal-command-prefix-reference-map-v1.json',Path('research/wakeword_p4/experiments/causal_positive_coverage_batch_v1.py'),Path('research/wakeword_p4/experiments/matchbox_command_material_findings_v1.md')]
rows=json.loads(paths[0].read_text());refs={r['id'] for r in json.loads(paths[1].read_text())['records']};groups={}
for r in rows:
 if r['label']!=1:continue
 key=r['engine']+'/'+r['source_pair_key']['carrier'];g=groups.setdefault(key,{'rows':0,'conditioning_ids':set(),'source_ids':set(),'frames_over846':0,'existing_aligned_map_matches':0,'generated_prefix_present':0})
 assert r['split']=='training';g['rows']+=1;g['conditioning_ids'].add(r['conditioning_id']);g['source_ids'].add(r['source_id']);g['frames_over846']+=r['frames']>846;g['existing_aligned_map_matches']+=r['source_id'] in refs;g['generated_prefix_present']+='generated_prefix_end_sample' in r
for g in groups.values():
 for k in ['conditioning_ids','source_ids']:g[k]=len(g[k])
out={'schema':'multiscale-broad-command-support-inventory-v1','groups':groups,'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'interpretation':'Current broad insertion explicitly selects carrierbare only. Command rows already exist and include generated prefix metadata; zero source-id matches to the current MFA aligned map does not prove missing acoustic timing or label defect. Generated boundaries and MFA estimates are different provenance, neither phonetic truth. Rows exceeding846 cannot enter current sampler unchanged. Prior wholecommand/alignedprefix/positivecoverage failures retained; this is not authorization to silently truncate or relabel, nor evidence that broader commands cure generalization.'}
(R/'4090-multiscale-broad-command-support-inventory-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(groups))
