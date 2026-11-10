"""Compare an automatic endpoint to the pinned public LDC phone reference."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--alignment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh report required')
    inputs = json.loads(a.inputs.read_text())
    alignment = json.loads(a.alignment.read_text())
    if (alignment['input_manifest_sha256'] != sha(a.inputs) or len(alignment['records']) != 1
            or inputs['wake_scores_accessed'] or inputs['independent_final_audio_accessed']
            or alignment['wake_scores_accessed'] or alignment['independent_final_audio_accessed']
            or not inputs['endpoint_tolerance_prespecified'] or inputs['endpoint_tolerance_ms'] != 20):
        raise ValueError('Changed reference or invalid diagnostic scope')
    row = alignment['records'][0]
    if row['source'] != inputs['records'][0] or row['source']['source_kind'] != 'published-human-phonetic-reference':
        raise ValueError('Reference and automatic alignment use different audio')
    last = inputs['reference_last_phone']
    expected = inputs['reference_speech_end_seconds']
    if last['end_samples']/16000 != expected:
        raise ValueError('Inconsistent published boundary')
    error = 1000*(row['automatic_word_end_seconds']-expected)
    phone_error = 1000*(row['automatic_last_phone']['end']-expected)
    result = dict(role='One published human-reference endpoint quality control; not wake-word acceptance',
                  source_sha256=sha(Path(__file__)), input_manifest_sha256=sha(a.inputs),
                  alignment_report_sha256=sha(a.alignment), reference_last_phone=last,
                  reference_endpoint_seconds=expected,
                  automatic_word_end_seconds=row['automatic_word_end_seconds'],
                  automatic_last_phone=row['automatic_last_phone'], endpoint_error_ms=error,
                  terminal_phone_endpoint_error_ms=phone_error, prespecified_tolerance_ms=20,
                  expected_words_preserved=row['expected_words_preserved'],
                  endpoint_within20ms=abs(error) <= 20+1e-8,
                  terminal_phone_endpoint_within20ms=abs(phone_error) <= 20+1e-8,
                  wake_scores_accessed=False, independent_final_audio_accessed=False,
                  old_windows_or_labels_changed=False, models_or_transcript_retuned=False,
                  limits='One public human utterance. Referenceaxr and MFA phone inventories differ; endpoint comparison does not assert identical phonetic segmentation. Neither passing nor failing this sample measures Nyx/Hi ESP endpoint accuracy. Keep original tolerance, labels and model configuration; no posthoc offset correction, population uncertainty interval, wake recall or latency superiority claim.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
