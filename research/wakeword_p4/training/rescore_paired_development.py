"""Recount existing paired development events at an independently calibrated point."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate import event_indices
from compare_paired_validation import classify, summarize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('evaluation','calibration','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    path=a.evaluation/'report.json';parent=json.loads(path.read_text())
    calibration=json.loads(a.calibration.read_text())
    if (not calibration.get('validation_only') or not calibration.get('not_for_final_acceptance')
            or calibration['threshold'] is None or parent['independent_final_audio_accessed']
            or parent['nyx_model_sha256']!=calibration['model_sha256']):
        raise ValueError('Need same-model diagnostic validation calibration')
    reference=calibration['reference_hiesp']
    if (parent['hiesp_threshold_requested']!=reference['threshold_requested']
            or parent['hiesp_threshold_getter_values']!=[reference['threshold_getter']]):
        raise ValueError('Different Hi ESP reference threshold')
    rows=parent['records'];window=calibration['sliding_window']
    for i,result in enumerate(rows):
        row=result['record']
        if row['split']!='validation' or row['role']!='development-paired-validation':
            raise ValueError('Only paired development validation')
        trace=a.evaluation/f'{i}-nyx-scores.npy'
        if sha(trace)!=result['nyx']['score_sha256']:raise ValueError('Changed score trace')
        scores=np.load(trace,allow_pickle=False)
        if scores.ndim!=1 or not len(scores) or not np.isfinite(scores).all():raise ValueError('Invalid scores')
        previous=[float((3*(j+window-1)+5)*.01) for j in event_indices(scores,parent['nyx_threshold'])]
        if previous!=result['nyx']['event_seconds']:raise ValueError('Original event count not reproduced')
        times=[float((3*(j+window-1)+5)*.01) for j in event_indices(scores,calibration['threshold'])]
        result['nyx']['event_seconds']=times
        result['nyx']['classification']=classify(times,row,'hey_nyx')
    report=dict(parent,role='Paired synthetic development at empirical Hi ESP event-budget point; not acceptance',
                source_sha256=sha(Path(__file__)),parent_report_sha256=sha(path),calibration_sha256=sha(a.calibration),
                parent_nyx_calibration_sha256=parent['nyx_calibration_sha256'],nyx_calibration_sha256=sha(a.calibration),
                nyx_threshold=calibration['threshold'],threshold_chosen_using_paired_scores=False,
                summaries={engine:summarize(rows,engine) for engine in ('nyx','hiesp')},
                not_for_final_acceptance=True,existing_zero_event_gate_passed=calibration['selected']['passes_existing_zero_event_gate'],
                empirical_reference_negative_events=calibration['reference_event_budget'],
                limits=parent['limits']+' Diagnostic threshold calibrated on separate continuous validation traces against observed reference event budget, not paired positives. Equal development counts do not establish equal population FAR or authorize final testing.')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summaries']))


if __name__=='__main__':main()
