"""Audit pure-translation consistency without wake scores or endpoint retuning."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('original','shifted','shift-inputs','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    old=json.loads(a.original.read_text());new=json.loads(a.shifted.read_text());inputs=json.loads(a.shift_inputs.read_text())
    if (inputs['parent_input_manifest_sha256']!=old['input_manifest_sha256']
            or new['input_manifest_sha256']!=sha(a.shift_inputs) or inputs['added_lead_seconds']!=.5
            or not inputs['remaining_original_pcm_verified'] or old['model_sources']!=new['model_sources']
            or old['mfa_version']!=new['mfa_version'] or old['command_policy']!=new['command_policy']):
        raise ValueError('Changed inputs/models/alignment configuration')
    for d in (old,new):
        if d['wake_scores_accessed'] or d['independent_final_audio_accessed']:raise ValueError('Invalid scope')
    by_path={r['source']['path']:r for r in new['records']}
    if len(by_path)!=len(new['records']) or set(by_path)!={r['source']['path'] for r in old['records']}:
        raise ValueError('Incomplete/duplicate alignment coverage')
    rows=[];tolerance_ms=20
    for previous in old['records']:
        current=by_path[previous['source']['path']]
        if (current['source']['parent_wav_sha256']!=previous['source']['sha256']
                or current['source']['parent_pcm_sha256']!=previous['source']['pcm_sha256']
                or current['source']['samples']!=previous['source']['samples']+8000):raise ValueError('Changed waveform body')
        words_old=[r for r in previous['tiers']['words'] if r['label']]
        words_new=[r for r in current['tiers']['words'] if r['label']]
        if [r['label'] for r in words_old]!=[r['label'] for r in words_new]:raise ValueError('Changed word sequence')
        endpoint_error=(current['automatic_word_end_seconds']-.5-previous['automatic_word_end_seconds'])*1000
        last_old=previous['automatic_last_phone'];last_new=current['automatic_last_phone']
        same_last_phone=last_old['label']==last_new['label']
        phone_start_error=(last_new['start']-.5-last_old['start'])*1000
        phone_end_error=(last_new['end']-.5-last_old['end'])*1000
        rows.append(dict(path=previous['source']['path'],source_kind=previous['source']['source_kind'],
                         endpoint_translation_error_ms=endpoint_error,
                         original_endpoint_seconds=previous['automatic_word_end_seconds'],shifted_endpoint_seconds=current['automatic_word_end_seconds'],
                         terminal_phone_label_unchanged=same_last_phone,terminal_phone_start_error_ms=phone_start_error,
                         terminal_phone_end_error_ms=phone_end_error,
                         endpoint_within_prespecified20ms=abs(endpoint_error)<=tolerance_ms+1e-8,
                         terminal_phone_boundaries_within20ms=same_last_phone and max(abs(phone_start_error),abs(phone_end_error))<=tolerance_ms+1e-8,
                         word_boundary_errors_ms=[dict(word=a['label'],start=(b['start']-.5-a['start'])*1000,end=(b['end']-.5-a['end'])*1000) for a,b in zip(words_old,words_new)]))
    result=dict(role='Automatic alignment500ms translation consistency audit, not phonetic truth',
                source_sha256=sha(Path(__file__)),original_report_sha256=sha(a.original),shifted_report_sha256=sha(a.shifted),
                shifted_input_manifest_sha256=sha(a.shift_inputs),tolerance_ms=tolerance_ms,records=rows,
                files=len(rows),endpoint_failures=sum(not r['endpoint_within_prespecified20ms'] for r in rows),
                terminal_phone_boundary_failures=sum(not r['terminal_phone_boundaries_within20ms'] for r in rows),
                max_endpoint_error_ms=max(abs(r['endpoint_translation_error_ms']) for r in rows),
                wake_scores_accessed=False,independent_final_audio_accessed=False,old_windows_or_labels_changed=False,
                limits='Same utterance bodies under exact extra zero lead, fixed models/config and20ms endpoint tolerance. Consistency does not establish correct pronunciation, true phone boundaries, independent population quality or latency superiority. Failures retained; no clip filtering or posthoc window change.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
