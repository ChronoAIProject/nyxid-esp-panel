"""Inventory existing training-only metadata; no audio/model calls or admission claim."""
import collections
import hashlib
import json
from pathlib import Path

ROOT=Path('.local/wakeword/hey-broad-training-features-v1/hey_nyx')
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    path=ROOT/'features.json';rows=json.loads(path.read_text())
    selected=[r for r in rows if r['label']==1 and r['source_pair_key']['carrier']!='bare']
    assert len(selected)==18684 and all(r['split']=='training' and r['target_id']=='hey_nyx' for r in selected)
    counts=collections.Counter((r['engine'],r['source_pair_key']['carrier']) for r in selected)
    scripts=[Path('research/wakeword_p4/experiments')/name for name in ['causal_positive_coverage_batch_v1.py','causal_source_balanced_batch_v1.py','causal_aligned_prefix_batch_v1.py']]
    result=dict(schema='existing-broad-command-support-v1',inputs=[dict(path=str(p),sha256=sha(p)) for p in [path,ROOT/'target-provenance.json',*scripts]],existing_training_command_rows=len(selected),maximum_frames=max(r['frames'] for r in selected),counts=[dict(engine=e,carrier=c,rows=n,conditioning_ids=len({r['conditioning_id'] for r in selected if r['engine']==e and r['source_pair_key']['carrier']==c})) for (e,c),n in sorted(counts.items())],current_broad_positive_sampler=dict(selected_carrier='bare',command_rows_selected_from_this_feature_store=0,other_command_training_exists=True),references=dict(generated_prefix_present=sum(r.get('generated_prefix_end_sample') is not None for r in selected),phonetic_truth_claimed=False,existing_automatic_alignment_map_scope='432 separate native training commands used by causal_aligned_prefix_batch_v1; not automatically applicable to this broader feature store'),records=[{k:r[k] for k in ['source_id','engine','conditioning_id','source_pair_key','variant','text','offset','frames','samples','sha256','pcm_sha256','generated_prefix_end_sample','generated_container_end_sample']} for r in selected],limits='Metadata inventory only. Existing complete command rows fit846 frames, but presence of generated prefix metadata does not validate phonetic boundary or authorize copying unrelated automatic alignments. Prior whole-command-end/bare-only/common-carrier failures retained. No new synthesis, inference, training, selection by score, dev/final use or acceptance claim. No claim missing command support causes errors.')
    # Compact gzip preserves all source identities without another audio/features copy.
    import gzip
    payload=(json.dumps(result,indent=2)+'\n').encode();dest=R/'4090-existing-broad-command-support-v1.json.gz'
    with dest.open('xb') as f:f.write(gzip.compress(payload,mtime=0))
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','inputs']}))
    print(json.dumps(dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest))))
if __name__=='__main__':main()
