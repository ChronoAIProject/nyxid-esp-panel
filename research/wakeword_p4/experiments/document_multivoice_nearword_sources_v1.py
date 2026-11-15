"""Document already-used training source roles and coverage, no detector calls."""
import collections,hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/hey-common-carrier-features-v1')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    metadata=B/'features.json';binary=B/'features.bin';rows=json.loads(metadata.read_text());original=R/'4090-hey-common-carrier-feature-result-v1.json';receipt=json.loads(original.read_text())
    assert sha(metadata)==receipt['metadata_sha256'] and sha(binary)==receipt['binary_sha256']
    assert len(rows)==4860 and all(r['split']=='training' and r['label']==0 for r in rows)
    counts=lambda key:dict(sorted(collections.Counter(str(r[key]) for r in rows).items()))
    cells=collections.Counter((r['conditioning_id'],r['anchor_text'],r['carrier'],r['speed'],r['variant']) for r in rows)
    assert len(cells)==4860 and set(cells.values())=={1}
    refs=[original,R/'4090-hey-common-carrier-kokoro-generation-result-v1.json',R/'4090-hey-common-carrier-render-result-v1.json',R/'4090-hey-common-carrier-render-independent-audit-v1.json',R/'4090-hey-common-carrier-raw-formal-result-v1.json',R/'4090-hey-common-carrier-raw-fp32-waveform-result-v1.json',R/'4090-causal-multivoice-nearword-plan-v1.json',metadata,binary]
    result=dict(schema='causal-multivoice-nearword-source-coverage-v1',references={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in refs},training_rows=4860,unique_full_factorial_cells=len(cells),counts={k:counts(k) for k in ['conditioning_id','anchor_text','carrier','speed','variant']},fullwave_frames=dict(minimum=min(r['frames'] for r in rows),maximum=max(r['frames'] for r in rows),padded=846,cropped=False),role='existing training only; historical failed common-carrier use retained',reused_audio=True,new_synthesis=False,detector_calls=0,protected_acceptance_accessed=False,jenny_added_to_training=False,independence_claim=False,limitations='20 Kokoro conditioning IDs, not proof of20 independent physical speakers or disjoint pretraining. Existing two-engine training sources remain; source-family count is not physical-speaker count. Jenny remains consumed development and is not introduced into training. Preserved native/render reports establish process quality only, not phonetic truth. This source summary does not establish recognition performance or promotion.')
    dest=R/'4090-causal-multivoice-nearword-source-coverage-v1.json'
    with dest.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows),cells=len(cells),binary_sha256=sha(binary))))
if __name__=='__main__':main()
