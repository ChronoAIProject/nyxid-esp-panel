"""Audit retained Hi ESP bracket reports/UART silently, without device access.

Matching aggregate counters is not sample-level input or latency evidence.
This helper reads existing files only; it cannot play audio or flash a board.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import wave


TRIAL_INPUTS = ('id', 'voice', 'kind', 'category', 'text', 'path', 'sha256',
                'samples', 'keyword_end', 'source_onset_samples')
WAKE_MESSAGE = re.compile(r'^I \(\d+\) panel\.wake: Hi ESP detected;', re.MULTILINE)
MODEL_MESSAGE = re.compile(r'^MC Quantized wakenet9: (.+)$', re.MULTILINE)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identities(trials):
    values = [{key: row[key] for key in TRIAL_INPUTS} for row in trials]
    if not values or len({row['id'] for row in values}) != len(values):
        raise ValueError('Empty or duplicate trial identities')
    return values


def audit_pair(before, after, before_uart, after_uart, plan, plan_sha256):
    expected = plan['arms']['hi_esp']
    frozen = identities(expected['trials'])
    arms = {}
    for name, report, uart in (('before', before, before_uart), ('after', after, after_uart)):
        if (report['arm'] != 'hi_esp' or report['complete'] is not True
                or report['plan_sha256'] != plan_sha256
                or report['app_elf_sha256'] != expected['app_elf_sha256']
                or identities(report['trials']) != frozen):
            raise ValueError('Incomplete or changed frozen control arm')
        trials = report['trials']
        if any(row['rearmed'] is not True or row['playback_conditions_unchanged'] is not True
               for row in trials):
            raise ValueError('Control report does not retain rearm/playback checks')
        if any(not row['source_playback_completed'] or row['playback_returncode_before_cleanup'] != 0
               for row in trials if row['event'] is None):
            raise ValueError('A miss lacks completed source playback')
        observed = sum(row['event'] is not None for row in trials)
        uart_events = len(WAKE_MESSAGE.findall(uart))
        if uart_events != observed:
            raise ValueError('Retained UART and report event totals differ')
        models = sorted(set(MODEL_MESSAGE.findall(uart)))
        boots = re.findall(r'^rst:0x[^\r\n]+', uart, re.MULTILINE)
        if len(models) != 1 or len(boots) != 1:
            raise ValueError('Model identity or reset accounting is ambiguous')
        if '\ufffd' in uart[uart.index(boots[0]):]:
            raise ValueError('Invalid UART text after boot could conceal runtime messages')
        arms[name] = {'trials': len(trials), 'observed_events': observed,
            'uart_events': uart_events, 'wake_model': models[0],
            'boot_reasons': boots,
            'uart_utf8_replacement_characters': uart.count('\ufffd'),
            'invalid_utf8_confined_to_boot_prefix': True,
            'no_event_trials_with_completed_source_playback': len(trials) - observed,
            'missed_positive_trials': sum(row['event'] is None and row['category'] == 'positive'
                                          for row in trials),
            'detected_trials_stopped_before_source_completion': sum(
                not row['source_playback_completed'] for row in trials if row['event'] is not None),
            'reported_rearm_and_playback_checks_passed': True}
    if arms['before']['wake_model'] != arms['after']['wake_model']:
        raise ValueError('Control model identity changed')
    changed = [{'id': a['id'], 'category': a['category'],
                'before_triggered': a['event'] is not None, 'after_triggered': b['event'] is not None}
               for a, b in zip(before['trials'], after['trials'])
               if (a['event'] is None) != (b['event'] is None)]
    return {'scope': __doc__, 'plan_sha256': plan_sha256,
        'same_trial_order_and_source_metadata': True, 'same_declared_app_and_wake_model': True,
        'arms': arms, 'changed_trials': changed, 'captured_microphone_audio_compared': False,
        'accepted': False,
        'interpretation': [
            'Every miss retained completed source playback; recorded Mac playback failure does not explain these misses.',
            'All retained UART detection messages agree with aggregate HTTP report counts. This checks aggregate accounting, not per-trial clock alignment or microphone continuity.',
            'Both controls retain the same app, model, source identities and trial order. Each UART contains only its initial boot; no extra reset is recorded.',
            'Successful trials may terminate source playback as soon as a wake is observed. They do not measure complete first-utterance delivery or conversational success.',
            'These reports do not contain sample-aligned microphone input. Input level, timing/state and room variation cannot be causally separated from these artifacts.',
            'Keep all changed control outcomes. No waveform, threshold, model or acceptance rule was adjusted, and no replay was performed.'
        ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'source-root', 'before-report', 'after-report',
                 'before-uart', 'after-uart', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file')
    plan = json.loads(args.plan.read_text())
    root = args.source_root.resolve()
    for trial in plan['arms']['hi_esp']['trials']:
        path = (root / trial['path']).resolve()
        if not path.is_relative_to(root) or sha(path) != trial['sha256']:
            raise ValueError('Frozen source path/hash differs')
        with wave.open(str(path), 'rb') as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) != (
                    16000, 1, 2, trial['samples']):
                raise ValueError('Frozen source timebase/format differs')
    result = audit_pair(json.loads(args.before_report.read_text()),
        json.loads(args.after_report.read_text()), args.before_uart.read_text(errors='replace'),
        args.after_uart.read_text(errors='replace'), plan, sha(args.plan))
    result['source_waveforms_verified'] = True
    result['input_files'] = {name: sha(getattr(args, name)) for name in (
        'before_report', 'after_report', 'before_uart', 'after_uart')}
    result['script_sha256'] = sha(Path(__file__))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'arms': result['arms'], 'changed_trials': result['changed_trials']}))


if __name__ == '__main__':
    main()
