"""Compare the frozen lexical ablation without selecting from development audio."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confidence_point(grid, include_confusers=True):
    points = [r for r in grid if r['continuous_events_per_hour_upper_95'] <= .5
              and (not include_confusers or r['max_corpus_false_rate'] <= .01)]
    return max(points, key=lambda r: (r['macro_recall'], -r['continuous_events_per_hour'],
                                     -r['max_corpus_false_rate'])) if points else None


def read_arm(root):
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    import numpy as np

    training = json.loads((root / 'complete.json').read_text())
    checkpoint = root / 'step-10000'
    calibration = json.loads((checkpoint / 'validation/calibration.json').read_text())
    if (training['plan']['recipe']['checkpoints'] != [10000]
            or len(training['checkpoints']) != 1 or training['checkpoints'][0]['step'] != 10000
            or sha(checkpoint / 'head.tflite') != training['checkpoints'][0]['model_sha256']
            or calibration['model_sha256'] != training['checkpoints'][0]['model_sha256']
            or calibration['feature_inputs'] != training['feature_inputs']
            or calibration['event_policy'] != training['plan']['event_policy']):
        raise ValueError('Frozen final checkpoint, calibration or features differ')
    model = Interpreter(model_path=str(checkpoint / 'head.tflite'),
                        experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    model.allocate_tensors()
    inputs, outputs = model.get_input_details(), model.get_output_details()
    ops = [op['op_name'] for op in model._get_ops_details()]
    if (len(inputs) != 1 or len(outputs) != 1 or tuple(inputs[0]['shape']) != (1, 16, 96, 1)
            or tuple(outputs[0]['shape']) != (1, 1) or outputs[0]['dtype'] != np.int8
            or inputs[0]['quantization'] != (1 / 128, 0) or outputs[0]['quantization'] != (1 / 256, -128)
            or ops != training['checkpoints'][0]['ops'] or 'SOFTMAX' in ops):
        raise ValueError('Exported detector interface or auxiliary pruning differs')
    return training, calibration, {
        'complete_sha256': sha(root / 'complete.json'),
        'calibration_sha256': sha(checkpoint / 'validation/calibration.json'),
        'details_sha256': sha(checkpoint / 'validation/details-private.json'),
        'representatives_sha256': sha(root / 'representatives-training-only.npy'),
        'plan_sha256': training['plan_sha256'], 'training_seconds': training['seconds'],
        'parameters': training['parameters'], 'auxiliary_parameters_training_only': training['auxiliary_parameters_training_only'],
        'export': training['checkpoints'][0], 'main_batches_sha256': training['main_batches_sha256'],
        'auxiliary_batches_sha256': training['auxiliary_batches_sha256'],
        'original_strict_selection': calibration['selection'], 'original_selected': calibration['selected'],
        'confidence_and_confuser_qualified': confidence_point(calibration['grid']),
        'confidence_only_diagnostic': confidence_point(calibration['grid'], include_confusers=False),
        'full_grid': calibration['grid']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--treatment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    control, left, a = read_arm(args.control)
    treatment, right, b = read_arm(args.treatment)
    if control['plan']['auxiliary_loss_weight'] != 0 or treatment['plan']['auxiliary_loss_weight'] != .2:
        raise ValueError('Expected frozen zero/.2 auxiliary weights')
    if ({k: v for k, v in control['plan'].items() if k != 'auxiliary_loss_weight'}
            != {k: v for k, v in treatment['plan'].items() if k != 'auxiliary_loss_weight'}):
        raise ValueError('Training plans differ beyond the auxiliary weight')
    for key in ('main_batches_sha256', 'auxiliary_batches_sha256', 'initial_weights_sha256',
                'feature_inputs', 'auxiliary_manifest_sha256', 'parameters', 'training_sources', 'training_frames'):
        if control[key] != treatment[key]:
            raise ValueError('Matched training evidence differs: ' + key)
    if (a['representatives_sha256'] != b['representatives_sha256']
            or a['export']['ops'] != b['export']['ops']
            or a['export']['model_bytes'] != b['export']['model_bytes']):
        raise ValueError('Matched PTQ or exported binary graph differs')
    joint_a, joint_b = a['confidence_and_confuser_qualified'], b['confidence_and_confuser_qualified']
    result = {'scope': 'Matched final-checkpoint lexical auxiliary ablation on existing validation; proxy-window metrics, not acoustic or final acceptance',
        'comparison_script_sha256': sha(Path(__file__)), 'matched_training_verified': True,
        'same_main_auxiliary_and_ptq_samples': True, 'same_exported_binary_interface_ops_and_size': True,
        'control': a, 'treatment': b,
        'joint_qualified_macro_difference': joint_b['macro_recall'] - joint_a['macro_recall'] if joint_a and joint_b else None,
        'no_new_joint_qualified_point': joint_b is None,
        'speaker_playback': False, 'microphone_access': False, 'audio_speed_changed': False,
        'limits': ['Repeated synthetic source views are not independent speaker trials.',
            'Original synthesis/energy endpoints are retained; proxy-window hits are not real recall or phonetic latency.',
            'Training-only RMS auxiliary windows are uncertain text supervision, not phonetic labels.',
            'The full fixed grid is retained, including failed working points and float/INT8 interface diagnostics.',
            'No development/final scoring, P4 numeric/resource evidence or acoustic/voice stability acceptance.'],
        'accepted': False}
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: {key: value[key] for key in ('original_selected', 'confidence_and_confuser_qualified', 'confidence_only_diagnostic')}
                      for name, value in [('control', a), ('treatment', b)]}
                     | {'joint_qualified_macro_difference': result['joint_qualified_macro_difference'], 'accepted': False}), flush=True)


if __name__ == '__main__':
    main()
