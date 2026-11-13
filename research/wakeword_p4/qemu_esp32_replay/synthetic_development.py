"""Explicit synthetic-duration development admission, separate from final data."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import wave


SCHEMA = 'synthetic-token-duration-paired-development-v1'
REFERENCE = 'generator-token-duration-prefix'
PREREGISTRATION = 'synthetic-token-duration-hiesp-preregistration-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe development path')
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), 'Escaped development path')
    return path


def replay_row(source, native):
    """Preserve source metadata; add explicitly synthetic paired replay fields."""
    return {**source, 'path': 'wav/' + source['path'], 'source_path': source['path'],
            'source_target': source['target'],
            'target': source['target'] if source['target'] in ('hey_nyx', 'hi_esp') else None,
            'role': 'development-paired-validation', 'split': 'validation',
            'phrase_id': native['id'], 'text': source['intended_phrase'],
            'keyword_end': source['generator_prefix_end_sample'],
            'keyword_end_reference_kind': REFERENCE, 'human_phonetic_endpoint': False}


def validate_manifest(path, authenticate_pcm=False):
    manifest = json.loads(path.read_text())
    require(manifest.get('schema') == SCHEMA and manifest.get('source_reference_kind') == REFERENCE,
            'Need explicit synthetic-duration development schema')
    require(manifest.get('role') == 'development-paired-validation'
            and manifest.get('paired_wakephrases') is True
            and manifest.get('independent_final_audio_accessed') is False
            and manifest.get('training') is False, 'Synthetic source cannot become training/final data')
    require(not any(key in manifest for key in ('voice_partition_sha256', 'endpoint_annotation_source_sha256')),
            'Do not impersonate legacy endpoint/voice-partition metadata')
    payloads = {}
    for name in ('source_manifest', 'generation_plan', 'source_audit'):
        source = inside(path.parent, manifest[name + '_path'])
        require(sha(source) == manifest[name + '_sha256'], 'Changed source provenance: ' + name)
        payloads[name] = json.loads(source.read_text())
    source, plan, audit = (payloads[k] for k in ('source_manifest', 'generation_plan', 'source_audit'))
    require(source.get('role') == plan.get('role') == 'synthetic-development-only'
            and source.get('training_data') is False and plan.get('training') is False
            and plan.get('existing_final_data_access') is False, 'Changed source reservation role')
    require(source.get('plan_sha256') == manifest['generation_plan_sha256']
            and source.get('generator_sha256') == plan.get('generator_sha256'), 'Unbound generation recipe')
    require(audit.get('manifest_sha256') == manifest['source_manifest_sha256']
            and audit.get('plan_sha256') == manifest['generation_plan_sha256']
            and audit.get('generator_sha256') == source.get('generator_sha256'), 'Unbound source audit')
    require(source.get('all_mechanical_pass') is True and source.get('mechanical_failures') == []
            and audit.get('independently_reconstructed_wav_bytes_exact') is True
            and audit.get('all_raw_sources_and_rir_tails_preserved') is True,
            'Need the completed original source integrity audit')
    rows, original = manifest['records'], source['records']
    native = {row['key']: row for row in source['native_records']}
    require(len(native) == len(source['native_records']) == plan['native_sources']
            and len(rows) == len(original) == plan['rendered_clips']
            and audit.get('native_files_verified') == len(native)
            and audit.get('wav_files_verified') == len(original), 'Incomplete original cohort')
    require(len({row['path'] for row in rows}) == len(rows), 'Duplicate replay paths')
    require(manifest.get('endpoint_protocol') == plan.get('timing_reference'), 'Changed reference semantics')
    for row, origin in zip(rows, original, strict=True):
        require(origin['source_key'] in native, 'Unknown original native source')
        source_row = native[origin['source_key']]
        alignment = source_row['generator_alignment']
        punctuation = next(i for i, phone in enumerate(alignment) if phone['phone'] in (',', '.', '?', '!'))
        end = alignment[punctuation - 1]['end_sample']
        require(end == source_row['generator_prefix_end_sample']
                and 0 < end <= source_row['source_samples']
                and sum(source_row['phoneme_id_samples']) == source_row['source_samples'],
                'Changed generator prefix duration')
        require(origin['target'] == source_row['target'] and origin['intended_phrase'] == source_row['text']
                and origin['generator_prefix_end_sample'] == origin['lead_samples'] + round(end * 16000 / source_row['source_rate']),
                'Prefix reference does not map to original PCM time')
        require(row == replay_row(origin, source_row), 'Changed source/order/target/endpoint metadata')
        require(0 < row['keyword_end'] <= row['samples'], 'Reference outside source buffer')
        wav = inside(path.parent, row['path'])
        if authenticate_pcm:
            require(sha(wav) == row['sha256'], 'Changed WAV bytes')
            with wave.open(str(wav), 'rb') as stream:
                require((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype(), stream.getnframes())
                        == (16000, 1, 2, 'NONE', row['samples']), 'Changed full PCM geometry')
                pcm = stream.readframes(stream.getnframes())
            require(hashlib.sha256(pcm).hexdigest() == row['pcm_sha256'], 'Changed full PCM payload')
    counts = dict(Counter(row['target'] if row['target'] is not None else 'none' for row in rows))
    require(counts == manifest.get('target_counts'), 'Changed paired target counts')
    return manifest


def validate_preregistration(args, manifest, runner_path):
    """Called before the existing runner creates QEMU or input.pcm."""
    freeze = json.loads(args.development_preregistration.read_text())
    require(freeze.get('schema') == PREREGISTRATION
            and freeze.get('role') == 'development-paired-validation'
            and freeze.get('independent_final_audio_accessed') is False
            and freeze.get('training') is False and not args.final_preregistration,
            'This preregistration cannot admit final/training or mixed replay roles')
    require(sha(args.manifest) == freeze.get('pcm_manifest_sha256'), 'Changed frozen development manifest')
    checked = validate_manifest(args.manifest)
    require(checked == manifest, 'Runner/source manifest differs')
    for name in ('source_manifest', 'generation_plan', 'source_audit'):
        require(freeze.get(name + '_sha256') == manifest[name + '_sha256'], 'Changed frozen source binding')
    require(freeze.get('source_reference_kind') == REFERENCE
            and freeze.get('records') == len(manifest['records'])
            and freeze.get('target_counts') == manifest['target_counts'], 'Changed frozen cohort/reference')
    require(freeze.get('runner_sha256') == sha(runner_path)
            and freeze.get('validator_sha256') == sha(Path(__file__)), 'Changed frozen replay source')
    require(freeze.get('model') == 'wn9_hiesp' and freeze.get('mode') == 'DET_MODE_95'
            and freeze.get('chunk_samples') == 512 and freeze.get('extra_prime_samples') == 0
            and args.threshold == freeze.get('threshold_requested') == .63
            and freeze.get('threshold_getter_expected') == .62496001, 'Changed historical official policy')
    for name in ('qemu', 'elf', 'flash'):
        require(sha(getattr(args, name)) == freeze.get(name + '_sha256'), 'Changed frozen engine: ' + name)
    return freeze
