"""Independently summarize published consumed results; no media or inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess


RESULTS = 'research/wakeword_p4/results/'
TARGETS = ('hey_willow', 'hey_nyx')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ref', default='56c0d97561c996488eb39394c9ba204ca956cef6')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', args.gpu_ref], text=True).strip()
    sources = {}

    def read(name):
        data = subprocess.check_output(['git', 'show', commit + ':' + RESULTS + name])
        sources[name] = digest(data)
        return json.loads(data)

    generation = read('4090-hey-paired-generation-plan-v2.json')
    development = read('4090-hey-development-render-contract-v1.json')
    diagnostic = read('4090-hey-consumed-raw-diagnostic-v1.json')
    candidates = read('4090-hey-v2-fixed-final-candidates-v1.json')
    training = [r for r in generation['requests'] if r['role'] == 'training']
    dev = development['requests']
    assert len(training) == 2400 and len(dev) == 120
    assert all(r['role'] == 'development' for r in dev)
    assert {r['voice'] for r in training}.isdisjoint({r['voice'] for r in dev})
    assert len(diagnostic['records']) == 1080 and len(diagnostic['training_cases']) == 4
    assert not diagnostic['inference_run'] and not diagnostic['acceptance_accessed']

    corpus = {}
    for name, rows in [('training', training), ('development', dev)]:
        corpus[name] = dict(
            native_requests=len(rows), rendered_clips=len(rows) * 3,
            voice_models=dict(sorted(Counter(r['voice'] for r in rows).items())),
            native_requests_by_target=dict(sorted(Counter(r['target_id'] for r in rows).items())),
            carriers=sorted({r['carrier'] for r in rows}),
            length_scales=sorted({r['length_scale'] for r in rows}),
            limits='TTS model identities and request counts, not independent human speaker counts.')

    comparisons = []
    for voice in sorted({r['voice'] for r in training + dev}):
        for target in (*TARGETS, 'hi_esp'):
            row = next(r for r in training + dev
                       if r['voice'] == voice and r['target_id'] == target
                       and r['carrier'] == '' and r['length_scale'] == 1.)
            comparisons.append(dict(voice=voice, target=target, literal=row['literal'],
                                    phonemes=''.join(row['phonemes']), token_ids=row['token_ids']))

    group_results = []
    lead = development['rendering']['leading_samples']
    assert lead == 28800
    for model in (*TARGETS, 'v30'):
        target = 'hey_willow' if model == 'hey_willow' else 'hey_nyx'
        rows = [r for r in diagnostic['records'] if r['model'] == model and r['target_id'] == target]
        assert len(rows) == 36 and len({(r['source_id'], r['variant']) for r in rows}) == 36
        groups = {}
        for field in ('variant', 'carrier', 'length_scale'):
            groups[field] = []
            for key in sorted({r[field] for r in rows}, key=str):
                selected = [r for r in rows if r[field] == key]
                groups[field].append(dict(value=key, clips=len(selected),
                    five_mean_peak_min=min(r['five_mean_peak'] for r in selected),
                    five_mean_peak_median=median(r['five_mean_peak'] for r in selected),
                    five_mean_peak_max=max(r['five_mean_peak'] for r in selected),
                    first_global_peak_before_source=sum(r['five_peak_first_support_sample'] < lead
                                                       for r in selected)))
        group_results.append(dict(model=model, target=target, groups=groups,
            observed_training_cases=[r for r in diagnostic['training_cases'] if r['model'] == model],
            limits='Global saved-output peaks and first-maximum clocks only. No threshold/event selection; '
                   'a pre-source first maximum does not prove all later values are lower or identify a cause.'))

    histories = {}
    for model in TARGETS:
        report = read(f'4090-hey-v2-{model}-original-training.json')
        assert report['steps'] == report['selected_step'] == 20000
        histories[model] = dict(history=report['history'], negative_weight=report['negative_weight'],
                                checkpoint_policy=report['checkpoint_policy'],
                                context_frames=report['context_frames'], frozen_batchnorm=report['frozen_batchnorm'])

    result = dict(schema='mac-hey-consumed-generalization-review-v1',
        gpu_commit=commit, source_sha256=digest(Path(__file__).read_bytes()), input_sha256=sources,
        candidate_freeze_schema=candidates['schema'], corpus=corpus,
        bare_scale_one_phoneme_metadata=comparisons, consumed_peak_groups=group_results,
        original_training_histories=histories,
        findings=[
            '2400 training native requests use only two TTS voice models; 120 development requests use one different model.',
            'Training US Willow phoneme metadata uses o while development GB Willow uses schwa; Nyx bare token IDs match. This is metadata, not acoustic truth.',
            'All 12 quiet Willow development positives have the same five-score global peak .0046875; their first maximum precedes source audio.',
            'The only saved Nyx development positive with five-score peak .99609375 is quiet scale .9 set-a-timer; other variants/carriers remain in the report.',
            'Very small final training loss and two saved positive training examples do not establish complete training-pack streaming accuracy or isolate overfitting.',
        ], next_diagnostic='Compare authenticated original final float streaming export, float with rounded INT8 input, '
            'and original INT8 on a predeclared balanced consumed training/development matrix; retain complete streams, '
            'reset boundaries and all outputs. No new checkpoint, threshold selection or acceptance access.',
        media_accessed=False, inference_run=False, candidate_promoted=False,
        threshold_selected=False, acceptance_accessed=False,
        limits='Independent metadata and published saved-output analysis only. Voice/accent, rendering, '
               'training coverage and quantization remain confounded; no single root cause is established.')
    with args.output.open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(gpu_commit=commit, training_voice_models=len(corpus['training']['voice_models']),
                         development_voice_models=len(corpus['development']['voice_models']),
                         media_accessed=False, inference_run=False)))


if __name__ == '__main__':
    main()
