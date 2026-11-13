"""Independently check temporal dependencies and crop support, without inference.

Authenticate the actual upstream files against the GPU plan, then compare
ordered symbolic convolution dependencies. No audio, learned weights, detector,
TTS, TensorFlow, device or acceptance data is loaded.
"""
import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess


GPU_REF = '65c3a23'
BASE = 'research/wakeword_p4/'


def digest(data):
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Node:
    signature: bytes
    low: int | None
    high: int | None


def combine(stage, children):
    spans = [n for n in children if n.low is not None]
    return Node(hashlib.sha256(stage.encode() + b''.join(n.signature for n in children)).digest(),
                min(n.low for n in spans) if spans else None,
                max(n.high for n in spans) if spans else None)


def valid(sequence, kernel, stride, stage):
    return [combine(stage, sequence[i:i + kernel])
            for i in range(0, len(sequence) - kernel + 1, stride)]


def offline(sequence, first, stride, kernels):
    out = valid(sequence, first, stride, 'first')
    for index, kernel in enumerate(kernels):
        out = valid(out, kernel, 1, f'depth{index}')
    return out


def symbolic_comparison(frames, first, stride, kernels):
    slots = (frames - first) // stride + 1 - sum(k - 1 for k in kernels)
    assert slots > 0
    receptive = first + stride * (sum(k - 1 for k in kernels) + slots - 1)
    unused = frames - receptive
    # Match the exact upstream Stream use_one_step=False ring sizes.
    lengths = [max(0, first - stride), *[k - 1 for k in kernels], slots - 1]
    zero = Node(hashlib.sha256(b'initial-state-zero').digest(), None, None)
    states = [[zero] * n for n in lengths]
    inputs = [Node(hashlib.sha256(f'feature{i}'.encode()).digest(), i, i) for i in range(990)]
    poison = Node(hashlib.sha256(b'ignored-poison-a').digest(), -1000, -1000)
    other_poison = Node(hashlib.sha256(b'ignored-poison-b').digest(), -2000, -2000)
    comparisons = nominal_differences = 0
    for end in range(stride, len(inputs) + 1, stride):
        memory = states[0] + inputs[end - stride:end]
        states[0] = memory[-lengths[0]:] if lengths[0] else []
        out = valid(memory, first, stride, 'first')
        assert len(out) == 1
        for index, kernel in enumerate(kernels):
            memory = states[index + 1] + out
            states[index + 1] = memory[-lengths[index + 1]:]
            out = valid(memory, kernel, 1, f'depth{index}')
            assert len(out) == 1
        stream = states[-1] + out
        states[-1] = stream[-lengths[-1]:] if lengths[-1] else []
        if end < frames:
            continue
        nominal = offline(inputs[end - frames:end], first, stride, kernels)
        reference = offline(inputs[end - receptive:end] + [poison] * unused,
                            first, stride, kernels)
        changed_unused = offline(inputs[end - receptive:end] + [other_poison] * unused,
                                 first, stride, kernels)
        assert stream == reference == changed_unused
        assert len(stream) == slots
        assert (stream[0].low, stream[-1].high) == (end - receptive, end - 1)
        comparisons += 1
        nominal_differences += nominal != stream
    assert comparisons and nominal_differences == comparisons
    return dict(context_frames=frames, head_temporal_slots=slots,
                streaming_ring_lengths=lengths, effective_receptive_frames=receptive,
                nominal_unused_trailing_frames=unused, compared_end_clocks=comparisons,
                phase_aware_ordered_dependencies_exact=True,
                changing_only_unused_input_has_no_effect=True,
                nominal_dependency_mismatches=nominal_differences,
                scope='Ordered symbolic temporal dependencies, not learned numeric outputs.')


def geometry(context, unused, maximum_prefix_plus_rir):
    checked = 0
    for rir in (0, 960):
        for prefix in range(1, maximum_prefix_plus_rir - rir + 1):
            # Bare source and a source with a full command tail. Both keep
            # source sample0, not a guessed silence or token-span boundary.
            for command_tail in (0, 16000):
                total_samples = 48000 + prefix + command_tail + rir + 16000
                features = (total_samples + 159) // 160 - 2
                boundary = 48000 + prefix + rir
                keyword_frame = boundary // 160 - 2
                for jitter in range(3, 28):
                    end = min(features, max(context, keyword_frame + jitter))
                    first_raw_sample = (end - context) * 160
                    # Feature i's aperture is [i*160, (i+3)*160). The final
                    # nominal input is unused by this valid-stride model.
                    last_raw_sample_exclusive = (end - unused + 2) * 160
                    assert first_raw_sample <= 48000
                    assert last_raw_sample_exclusive >= boundary
                    checked += 1
    return dict(context_frames=context, nominal_unused_trailing_frames=unused,
                prefix_plus_rir_bound_samples=maximum_prefix_plus_rir,
                lead_samples=48000, tail_samples=16000, rir_tails=[0, 960],
                jitters=list(range(3, 28)), constructed_crops_checked=checked,
                complete_source_start_prefix_and_rir_in_actual_model_aperture=True,
                scope='Constructed integer boundaries only; not actual batch/feature-pack admission.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ref', default=GPU_REF)
    parser.add_argument('--upstream-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ref = subprocess.check_output(['git', 'rev-parse', args.gpu_ref], text=True).strip()
    identities = {}

    def source(path):
        data = subprocess.check_output(['git', 'show', ref + ':' + path])
        identities[path] = digest(data)
        return data

    plan = json.loads(source(BASE + 'results/4090-hey-context225-stream-probe-plan-v3.json'))
    for name in ('mixednet.py', 'layers/stream.py', 'utils.py', 'layers/modes.py'):
        data = (args.upstream_root / name).read_bytes()
        expected = plan['files']['.local/wakeword/microWakeWord/microwakeword/' + name]
        assert digest(data) == expected, name
        identities['upstream/' + name] = expected
    helper = source(BASE + 'experiments/probe_hey_context_transfer_shapes_v1.py')
    arch_node = next(n for n in ast.parse(helper).body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'ARCH' for t in n.targets))
    arch = ast.literal_eval(arch_node.value)
    flags = dict(zip(arch[::2], arch[1::2], strict=True))
    assert ast.literal_eval(flags['--residual_connection']) == (0, 0, 0, 0)
    assert ast.literal_eval(flags['--repeat_in_block']) == (1, 1, 1, 1)
    first, stride = int(flags['--first_conv_kernel_size']), int(flags['--stride'])
    kernels = [max(k) for k in ast.literal_eval(flags['--mixconv_kernel_sizes'])]
    assert (first, stride, kernels) == (5, 3, [5, 9, 13, 21])
    decision = json.loads(source(BASE + 'results/4090-hey-broad-context-decision-v1.json'))
    maximum = max(r['whole_source_prefix_plus_rir_max'] for r in decision['training_only_geometry'].values())
    required = max(150, max(r['frames_with_max_jitter'] for r in decision['training_only_geometry'].values()))
    assert decision['context_frames'] == ((required + 2) // 3) * 3 == 225
    symbolic = [symbolic_comparison(n, first, stride, kernels) for n in (150, 225, 300)]
    selected = next(r for r in symbolic if r['context_frames'] == 225)
    result = dict(schema='mac-independent-context225-phase-and-support-v1', gpu_commit=ref,
        source_sha256=digest(Path(__file__).read_bytes()), input_sha256=identities,
        architecture=flags, symbolic=symbolic,
        geometry=geometry(225, selected['nominal_unused_trailing_frames'], maximum),
        interpretation='The valid first conv and subsequent valid kernels leave one nominal trailing input unused. '
            'At stream end E, the context225 graph matches the latest224 real frames plus one unused dummy, '
            'rather than the nominal225-frame window ending at E. Keep original stream sample clocks; '
            'this reference alignment is not an offset search or a latency correction.',
        learned_output_equivalence_proven=False, media_accessed=False, inference_run=False,
        threshold_selected=False, acceptance_accessed=False, candidate_promoted=False,
        limits='Authenticates declared source bytes and verifies architecture dependencies/integer support only. '
            'Actual float comparisons, unchanged source weights, trained INT8/P4 export and actual batches remain separate evidence.')
    with args.output.open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(symbolic_clocks=sum(r['compared_end_clocks'] for r in symbolic),
                         context225_effective_frames=selected['effective_receptive_frames'],
                         constructed_crops=result['geometry']['constructed_crops_checked'],
                         inference_run=False)))


if __name__ == '__main__':
    main()
