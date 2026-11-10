"""Verify a quarter-TTS source-activity ablation without changing evaluation."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from compare_lexical_auxiliary import read_arm
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('control', 'treatment', 'reference-complete', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    control, _, left = read_arm(args.control)
    treatment, _, right = read_arm(args.treatment)
    difference = 'focus_negative_activity'
    if control['plan'][difference] is not False or treatment['plan'][difference] is not True:
        raise ValueError('Expected explicit off/on source-activity experiment')
    if ({k: v for k, v in control['plan'].items() if k != difference} !=
            {k: v for k, v in treatment['plan'].items() if k != difference}):
        raise ValueError('Plans differ beyond quarter-TTS negative endpoint sampling')
    if control['plan']['activity_stride'] != 4:
        raise ValueError('Expected one quarter of TTS negative windows')
    for key in ('main_sources_batches_sha256', 'main_unmodified_batches_sha256',
                'auxiliary_batches_sha256', 'auxiliary_source_batches_sha256', 'engine_labels_batches_sha256',
                'initial_weights_sha256', 'auxiliary_manifest_sha256', 'negative_activity_manifest_sha256',
                'feature_inputs', 'parameters', 'training_sources', 'training_frames'):
        if control[key] != treatment[key]:
            raise ValueError('Source, unchanged histories or auxiliary evidence differs: ' + key)
    if (control['main_batches_sha256'] == treatment['main_batches_sha256']
            or left['representatives_sha256'] != right['representatives_sha256']
            or left['export']['ops'] != right['export']['ops']
            or left['export']['model_bytes'] != right['export']['model_bytes']):
        raise ValueError('Intervention absent, or PTQ/exported binary topology differs')
    reference = json.loads(args.reference_complete.read_text())
    if sha(args.reference_complete) != control['plan']['retained_reference_complete_sha256']:
        raise ValueError('Frozen retained control artifact differs')
    for key in ('main_batches_sha256', 'auxiliary_batches_sha256'):
        if control[key] != reference[key]:
            raise ValueError('New control does not reproduce retained training batches')
    if control['checkpoints'][0]['model_sha256'] != reference['checkpoints'][0]['model_sha256']:
        raise ValueError('New control does not reproduce retained model bytes')
    for side, complete in [(left, control), (right, treatment)]:
        for key in ('main_sources_batches_sha256', 'main_unmodified_batches_sha256',
                    'auxiliary_source_batches_sha256', 'engine_labels_batches_sha256',
                    'negative_activity_manifest_sha256', 'tts_negative_endpoint_interval_counts'):
            side[key] = complete[key]
    a, b = left['confidence_and_confuser_qualified'], right['confidence_and_confuser_qualified']
    result = {'scope': 'Quarter-TTS endpoint sampling on original training features and reused validation; no acoustic/final acceptance',
              'script_sha256': sha(Path(__file__)), 'control': left, 'treatment': right,
              'all_source_class_draws_auxiliary_ptq_and_unreplaced_histories_match': True,
              'new_control_exactly_reproduces_retained_model_main_and_auxiliary_batches': True,
              'reference_complete_sha256': sha(args.reference_complete),
              'joint_qualified_macro_difference': b['macro_recall'] - a['macro_recall'] if a and b else None,
              'speaker_playback': False, 'microphone_access': False, 'audio_speed_changed': False,
              'accepted': False,
              'limits': ['Energy-span endpoint membership is not speech occupancy or phonetic truth; history can retain earlier speech outside that interval.',
                         'Keep both original joint and confidence-only diagnostic rows. The1%confuser gate is an additional research constraint, not Hi ESP parity.',
                         'Repeated synthetic views and reused validation do not prove real-user recall, latency or population false-wake equality.',
                         'No new runtime VAD, detector policy, microphone/audio producer, AEC/AGC/FIFO or waveform processing.']}
    with args.report.open('x') as out:
        out.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'control_joint_macro': a['macro_recall'] if a else None,
                      'treatment_joint_macro': b['macro_recall'] if b else None,
                      'difference': result['joint_qualified_macro_difference'],
                      'control_confidence_only': left['confidence_only_diagnostic']['macro_recall'],
                      'treatment_confidence_only': right['confidence_only_diagnostic']['macro_recall']}), flush=True)


if __name__ == '__main__':
    main()
