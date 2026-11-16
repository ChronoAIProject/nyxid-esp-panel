import hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 items={
 'frozen_v30':dict(path='.local/wakeword/model-gpu-v30-ami-extra/hey_nyx.tflite',required_sha256='04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c',status='available'),
 'fresh_threshold_development':dict(status='missing',evidence=str(R/'4090-unused-development-inventory-audit-v1.json')),
 'sealed_acceptance':dict(status='missing',reason='No candidate has passed reused-development viability; protected final sets remain closed'),
 'candidate_int8_qualification':dict(status='missing',reason='All recent causal controls rejected before export'),
 'official_hiesp_matched_protocol':dict(status='incomplete',reason='Existing synthetic references are diagnostic and not a qualified candidate comparison'),
 'human_generalization':dict(status='out_of_scope',reason='Objective explicitly limits claims to synthetic protocol')}
 for v in items.values():
  if 'path' in v: v['bytes']=Path(v['path']).stat().st_size;v['sha256']=sha(v['path']);assert v['sha256']==v['required_sha256']
 out=dict(schema='current-causal-completion-gap-audit-v1',branch='research-hye-nyx-voice-gpu',latest_commit='e280e6ae',items=items,executed_controls=['contextual','normalized','styleaug','conditioning-risk','allocation-margin','channel-mask'],policy='Do not promote or handoff any rejected control; do not open protected final sets; next work requires a genuinely unconsumed causal-compatible development pack or a separately justified new source protocol.',limits='Read-only state audit; no model/audio/threshold calls.')
 p=R/'4090-current-causal-completion-gap-audit-v1.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
