"""Exercise review validation with synthetic in-memory fixtures, never human labels."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from collect_review_annotations import agreement, load_pack, validate_export


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack',nargs=2,type=Path,action='append',required=True,metavar=('HTML','AUDIT'))
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh check output required')
    checks = []
    for html,audit in a.pack:
        pack = load_pack(html,audit)
        fixture = dict(protocol=pack['protocol'],pack_id=pack['pack_id'],
                       parent_manifest_sha256=pack['parent_manifest_sha256'],
                       annotator='SYNTHETIC_VALIDATION_FIXTURE_A',
                       detector_outputs_shown=False,automatic_boundaries_shown=False,
                       review_complete=False,
                       records=[{k:r[k] for k in ('id','wav_sha256','pcm_sha256','samples')} for r in pack['records']])
        fixture['records'][0].update(status='matches',lower_sample=1000,upper_sample=1200,notes='')
        first = validate_export(fixture,pack)
        assert not first['complete'] and first['records'][pack['records'][0]['id']]['complete']
        invalid = []
        def bad(name,mutate):
            trial = copy.deepcopy(fixture)
            mutate(trial)
            try:
                validate_export(trial,pack)
            except ValueError:
                invalid.append(name)
            else:
                raise AssertionError('Invalid export accepted: '+name)
        bad('wrong-pack',lambda x:x.update(pack_id='0'*64))
        bad('wrong-parent',lambda x:x.update(parent_manifest_sha256='0'*64))
        bad('detector-exposed',lambda x:x.update(detector_outputs_shown=True))
        bad('automatic-boundaries-exposed',lambda x:x.update(automatic_boundaries_shown=True))
        bad('known-automated-ui-fixture',lambda x:x.update(annotator='AUTOMATED_UI_TEST_NOT_HUMAN_LABELS'))
        bad('empty-annotator',lambda x:x.update(annotator=' '))
        bad('changed-waveform',lambda x:x['records'][0].update(pcm_sha256='0'*64))
        bad('duplicate-record',lambda x:x['records'].__setitem__(1,copy.deepcopy(x['records'][0])))
        bad('missing-record',lambda x:x['records'].pop())
        bad('negative-boundary',lambda x:x['records'][0].update(lower_sample=-1))
        bad('boundary-beyond-audio',lambda x:x['records'][0].update(upper_sample=x['records'][0]['samples']+1))
        bad('boolean-boundary',lambda x:x['records'][0].update(lower_sample=True))
        bad('reversed-boundaries',lambda x:x['records'][0].update(lower_sample=1300))
        bad('false-complete-claim',lambda x:x.update(review_complete=True))
        bad('unknown-status',lambda x:x['records'][0].update(status='automatic-gold'))
        bad('invalid-notes-type',lambda x:x['records'][0].update(notes=[]))
        other = copy.deepcopy(fixture)
        other['annotator'] = 'SYNTHETIC_VALIDATION_FIXTURE_B'
        other['records'][0].update(lower_sample=1100,upper_sample=1300)
        pair = agreement(pack,[first,validate_export(other,pack)])[0]
        assert pair['descriptive_agreement']=='matches'
        assert pair['conservative_interval_union']==dict(lower_sample=1000,upper_sample=1300,intervals_overlap=True,separation_samples=0)
        other['records'][0].update(lower_sample=1500,upper_sample=1700)
        disjoint = agreement(pack,[first,validate_export(other,pack)])[0]
        assert disjoint['conservative_interval_union']==dict(lower_sample=1000,upper_sample=1700,intervals_overlap=False,separation_samples=300)
        other['records'][0].update(status='mismatch',lower_sample=None,upper_sample=None,notes='Synthetic incomplete-phrase fixture')
        dissent = agreement(pack,[first,validate_export(other,pack)])[0]
        assert dissent['descriptive_agreement']=='disagreement' and dissent['conservative_interval_union'] is None
        single = agreement(pack,[first])[0]
        assert single['descriptive_agreement']=='single-review' and single['conservative_interval_union'] is None
        try:
            agreement(pack,[first,first])
        except ValueError:
            pass
        else:
            raise AssertionError('Repeated reviewer counted twice')
        checks.append(dict(pack_id=pack['pack_id'],clips=len(pack['records']),rejected_invalid_cases=invalid,
                           valid_partial_export_checked=True,union_preserves_uncertainty=True,
                           disjoint_intervals_not_hidden=True,pronunciation_dissent_retained=True,
                           single_review_not_consensus=True,repeated_identity_rejected=True))
    code = Path(__file__)
    result = dict(role='Synthetic review-collector checks only; not listening annotations',
                  source_sha256=sha(code),collector_source_sha256=sha(code.with_name('collect_review_annotations.py')),
                  checks=checks,fixtures_are_synthetic_not_human=True,fixture_annotation_files_written=False,
                  human_annotations_present=False,gold_endpoints_created=False)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
