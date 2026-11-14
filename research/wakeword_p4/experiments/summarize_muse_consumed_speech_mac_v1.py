"""Summarize all saved Muse PCM cases after a verified native transaction."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def sha(data):
    return hashlib.sha256(data).hexdigest()


def summarize(report, raw, independent, protocol):
    assert report['backend'] == 'hardware' and report['capture_origin_bound'] and report['restored']
    assert 'error' not in report and 'restoration_error' not in report
    assert report['reduction'] == independent and independent['capture_sha256'] == sha(raw)
    assert report['binding'] == protocol
    cases = independent['cases']
    assert len(cases) == 216 and len(independent['repeat_exact']) == 108
    assert sum(c['result']['samples'] for c in cases) == 15298266
    assert independent['input_invocations'] == 29990 and independent['prime_invocations'] == 13608
    groups = []
    for literal in protocol['fixtures']['selected_literals']:
        for repeat in range(2):
            rows = [c for c in cases if c['original']['literal'] == literal and c['repeat'] == repeat]
            assert len(rows) == 36
            for dimension, key in (('all', None), ('carrier', 'carrier'), ('variant', 'variant'), ('length_scale', 'length_scale')):
                values = [None] if key is None else sorted({c['original'][key] for c in rows})
                for value in values:
                    selected = rows if key is None else [c for c in rows if c['original'][key] == value]
                    observed = []
                    for case in selected:
                        source = case['original']
                        times = [event['sample_end'] for event in case['events']]
                        after = [t for t in times if t >= source['lead_samples']]
                        observed.append(dict(any_after_source=bool(after),
                            at_or_after_generated_prefix=any(t >= source['generated_prefix_end_sample'] for t in after),
                            before_source_events=sum(t < source['lead_samples'] for t in times),
                            first_after_source_signed_ms=(after[0]-source['generated_prefix_end_sample'])/16 if after else None))
                    signed = [row['first_after_source_signed_ms'] for row in observed if row['first_after_source_signed_ms'] is not None]
                    groups.append(dict(literal=literal, repeat=repeat, dimension=dimension, value=value, clips=len(selected),
                        clips_with_events=sum(bool(c['events']) for c in selected), events=sum(len(c['events']) for c in selected),
                        any_after_source=sum(row['any_after_source'] for row in observed),
                        at_or_after_generated_prefix=sum(row['at_or_after_generated_prefix'] for row in observed),
                        before_source_events=sum(row['before_source_events'] for row in observed),
                        first_after_source_signed_ms=dict(observed=len(signed),minimum=min(signed) if signed else None,
                            median=median(signed) if signed else None,maximum=max(signed) if signed else None)))
    return dict(schema='official-muse-consumed108-native-summary-v1', protocol_sha256=sha(json.dumps(protocol,indent=2).encode()+b'\n'),
        capture_sha256=sha(raw), capture_bytes=len(raw), source_sha256=sha(Path(__file__).read_bytes()),
        image_sha256=report['binding']['image_sha256'], capture_reducer_sha256=report['capture_reducer_sha256'],
        phases=report['phases'], executions=216, input_detect_calls=29990, prime_detect_calls=13608,
        all_repeats_exact=independent['all_repeats_exact'], max_detect_us=independent['max_detect_us'],
        all_input_detect_us=sum(c['result']['total_us'] for c in cases), groups=groups,
        original_slot_sha256=report['original_slot_sha256'], restored_slot_sha256=report['restored_slot_sha256'],
        partition_sha256=report['partition_sha256'], restored_partition_sha256=report['restored_partition_sha256'],
        otadata_sha256=report['otadata_sha256'], restored_otadata_sha256=report['restored_otadata_sha256'],
        full_slot_partition_otadata_restored_exact=True, independent_saved_reduction_exact=True,
        readiness=report['execution_parameters']['restore_readiness'],
        restored_app_sha256=report['after']['app_sha256'], restored_slot=report['after']['ota']['running'],
        restored_voice_service_unchanged=report['before']['voice']['service_id']==report['after']['voice']['service_id'],
        restored_wake_enabled_listening=all(report['after']['voice']['wake'][k] for k in ('enabled','listening')),
        before_nyx_state=report['before']['nyx_state'], after_nyx_state=report['after']['nyx_state'],
        candidate_qualified=False, threshold_selected=False, microphone_speaker_used=False,
        fresh_development_accessed=False, acceptance_accessed=False,
        limits=[
            'All108 original consumed clips,36 per literal,one synthetic Semaine voice,all original variants/carriers/speeds. DEFAULT-only, no background exposure or threshold selection.',
            'Generated-prefix support is not a human phonetic boundary; signed offsets and full-clip detection do not establish live wake latency or first-utterance delivery.',
            'No background FAR, source generalization, direct raw-five-score policy equivalence, full-panel memory or voice-connectivity acceptance.',
            'Per-call times are standalone wn->detect timings, excluding creation/priming/UART and complete application work.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'capture', 'independent-reduction', 'protocol', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.report.read_text()), args.capture.read_bytes(),
        json.loads(args.independent_reduction.read_text()), json.loads(args.protocol.read_text()))
    assert result['protocol_sha256'] == sha(args.protocol.read_bytes())
    for original, restored in (('original_slot_sha256','restored_slot_sha256'),('partition_sha256','restored_partition_sha256'),('otadata_sha256','restored_otadata_sha256')):
        assert result[original] == result[restored]
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(executions=216, all_repeats_exact=result['all_repeats_exact'], max_detect_us=result['max_detect_us'],
        groups=[g for g in result['groups'] if g['dimension']=='all'])))


if __name__ == '__main__':
    main()
