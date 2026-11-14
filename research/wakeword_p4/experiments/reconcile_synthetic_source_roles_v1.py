"""Reconcile documented roles, without reading audio or opening acceptance data."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path('research/wakeword_p4/results')


def main():
    names = [
        '4090-joe-fixed-baseline-summary-v1.json',
        '4090-joe-known-source-overlap-supplement-v1.json',
        '4090-jenny-development-native-generation-v1.json',
        '4090-jenny-development-render-independent-audit-v1.json',
        '4090-jenny-consistency-development-lock-preflight-v1.json',
        '4090-mike-acceptance-generation-reservation-v1.json',
        '4090-joe-mike-generation-plan-v1.json',
    ]
    evidence = {}
    data = {}
    for name in names:
        raw = (ROOT / name).read_bytes()
        data[name] = json.loads(raw)
        evidence[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    joe = data[names[0]]
    native = data[names[2]]
    render = data[names[3]]
    assert joe['complete_records'] == 360 and joe['candidate_scored'] is False
    assert native['requests'] == native['quality_passes'] == 132
    assert render['cases'] == render['independently_qualified'] == 396
    assert all(r['role'] == 'acceptance' for r in data[names[5]]['requests'])
    report = dict(
        schema='synthetic-source-role-reconciliation-v1',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        evidence=evidence,
        sources=[
            dict(source='Piper Joe', role='consumed development diagnostic',
                 evidence=names[:2], fresh_holdout_allowed=False,
                 reason='360 actual fixed-baseline records; not training or fresh acceptance.'),
            dict(source='author-direct Jenny', role='reserved development; generated and process audited',
                 evidence=names[2:5], fresh_holdout_allowed=False,
                 reason='132 native requests, 396 renders; retain development reservation. Historical failed preflight does not prove current scoring absence.'),
            dict(source='Piper Mike', role='reserved acceptance; generation/scoring not established by these receipts',
                 evidence=[names[1], names[5], names[6]], fresh_holdout_allowed=False,
                 reason='Keep protected. Shared Joe/Mike parent and unknown physical/pretraining overlap prevent independent-acceptance admission from reservation alone.'),
        ],
        next_action='Find and authenticate additional source families and actual historical usage before reserving a new multi-voice training/development/acceptance plan; do not recycle Joe or use Mike as independent by default.',
        limits='Curated evidence-linked role reconciliation, not an exhaustive usage census or proof of no later scoring. No audio, sealed acceptance material, detector outputs, or model bytes read. Does not relax conservative exclusion guard or qualify any new source.',
        generation_run=False, detector_invocations=0, acceptance_opened=False,
    )
    dest = ROOT / '4090-synthetic-source-role-reconciliation-v1.json'
    with dest.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(sources=len(report['sources']), evidence_files=len(evidence), output=str(dest))))


if __name__ == '__main__':
    main()
