"""Supersede historical role flags with committed actual scoring evidence."""
import gzip
import hashlib
import json
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    prior=R/'4090-synthetic-source-role-reconciliation-v1.json'
    finals=R/'4090-independent-data-usage-ledger-v1.json'
    inputs={str(prior):sha(prior),str(finals):sha(finals)};scored=[]
    # These are known completed interventions, not a claim of exhaustive history.
    conditions=['source-balanced','character-auxiliary','phone-auxiliary','phrase-auxiliary','nonlinear-readout','bn-affine','latent-command','negative-peak','lexical-competition']
    expected={'primed':372,'jenny':408,'joe':360}
    for condition in conditions:
        for group,count in expected.items():
            manifest=R/f'4090-causal-{condition}-{group}-evidence-manifest-v1.json'
            receipt=json.loads(manifest.read_text());inputs[str(manifest)]=sha(manifest)
            reports=[x for x in receipt['files'] if x.get('original_path','').endswith('/report.json')]
            assert len(reports)==1
            artifact=Path(reports[0]['path']);data=artifact.read_bytes()
            assert hashlib.sha256(data).hexdigest()==reports[0]['sha256']
            original=gzip.decompress(data);assert hashlib.sha256(original).hexdigest()==reports[0]['original_sha256']
            report=json.loads(original);assert report['all_passed'] and len(report['records'])==count
            scored.append(dict(condition=condition,group=group,complete_input_records=count,report_path=str(artifact),report_sha256=sha(artifact),evidence_manifest=str(manifest),role='consumed development; not fresh threshold selection or final acceptance'))
    protected=json.loads(finals.read_text())['datasets']
    result=dict(schema='causal-development-usage-reconciliation-v2',supersedes_role_status_in=str(prior),inputs=inputs,completed_model_group_evidence=scored,sources=[dict(source='legacy synthetic plus existing continuous background',role='consumed development',fresh_threshold_selection_allowed=False),dict(source='author-direct Jenny plus earnings12 background',role='consumed development',fresh_threshold_selection_allowed=False,supersedes='Old reserved/process-audited status is historical; actual scoring is now established.'),dict(source='Piper Joe',role='consumed development',fresh_threshold_selection_allowed=False),dict(source='Piper Mike',role='protected reserved acceptance; freshness/independence not established',fresh_threshold_selection_allowed=False,material_accessed=False)],historical_finals=[dict(dataset=x['dataset'],usable_for_training_or_threshold_selection=x['usable_for_training_or_threshold_selection'],usable_as_untouched_future_acceptance=x['usable_as_untouched_future_acceptance']) for x in protected],unconsumed_development_found_in_this_scope=False,threshold_selection_performed=False,new_audio_or_detector_calls=0,limits='Known actual report metadata only; nine interventions and three groups prove consumption, not full historical coverage or absence of other unconsumed sources. No PCM/features/model outputs/final or Mike material opened. Reports contain observations from reused correlated inputs, never independent additive exposure. This does not alter training roles or add an admission gate.',next_action='Freeze a genuinely unconsumed development source and threshold/tie protocol before selecting a candidate threshold; separately keep independent final sources protected. Existing Jenny/Joe/legacy cannot supply this missing evidence.')
    out=R/'4090-causal-development-usage-reconciliation-v2.json'
    with out.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(completed_model_group_reports=len(scored),known_consumed_groups=len(expected),unconsumed_development_found=False,model_calls=0)))
if __name__=='__main__':main()
