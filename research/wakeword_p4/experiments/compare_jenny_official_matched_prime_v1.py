"""Combine completed behavioral reports under common totalzero PCM warmup."""
import hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    previous=R/'4090-nearword-jenny-official-default-comparison-v1.json';matched=R/'4090-jenny-official-matched-prime-behavior-v1.json'
    old=json.loads(previous.read_text());new=json.loads(matched.read_text());assert new['total_prime_samples']==81600 and not new['acceptance_accessed']
    models=[m for m in old['models'] if m['model']!='official-wn9_hiesp-DEFAULT'];assert len(models)==3 and all(m['prime_samples']==81600 for m in models)
    models.append(dict(model='official-wn9_hiesp-DEFAULT-matched-prime',source=str(matched),source_sha256=sha(matched),prime_samples=81600,native_identity=new['native_identity'],background=new['background'],positive=new['positive'],shared_common=new['shared_common'],target_details=new['target_details']))
    out=dict(schema='jenny-official-matched-prime-comparison-v1',source_reports={str(p):sha(p) for p in [previous,matched]},manifest_sha256=old['manifest_sha256'],common_zero_pcm_samples=81600,common_original_synthetic_inputs=396,common_original_background_inputs=12,models=models,previous_official_native_warmup_reference=[m for m in old['models'] if m['model']=='official-wn9_hiesp-DEFAULT'],threshold_selected=False,acceptance_accessed=False,candidate_promoted=False,limits='Same original full396-input synthesis protocol and12complete background recordings, common81600zero PCM before real input; continuous state/pending blocks and application-ignore-prime policy retained. Original official DEFAULT and candidate.91796875/smoothing5 operating points remain detector-specific; native512/480chunk clocks/frontends/events remain native, no numerical equivalence or P4 parity claim. Target phrases differ by intended wakeword. Shared288negative source/variant cohort excludes all three target families. Signed latency/complete counts use generated coordinates, not human phonetic truth; before-reference events are not an automatic veto. Consumed single-Jenny-voice development, not fresh threshold selection/multi-voice independent acceptance. Prior32256official reference retained separately. Raw behavior, misses and FAR bounds determine engineering interpretation; no promotion here.')
    for m in models:
        assert m['shared_common']['clips']==288 and m['positive']['all']['clips']==36 and m['positive']['command']['clips']==27
        assert abs(m['background']['hours']-9.614419583333333)<1e-10
    dest=R/'4090-jenny-official-matched-prime-comparison-v1.json'
    with dest.open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
    for m in models:print(json.dumps(dict(model=m['model'],positive=m['positive']['all'],commands=m['positive']['command'],shared_common=m['shared_common'],background={k:m['background'][k] for k in ['events','hours','far_upper95']})))
if __name__=='__main__':main()
