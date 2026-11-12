"""Prepare complete source-reserved real-speech development, never wake scores."""
import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np
import soundfile as sf

TRAINING = Path(__file__).resolve().parents[1] / 'training'
sys.path.insert(0, str(TRAINING))
from features import continuous_wav_features


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while data := source.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def transcript_audit(path):
    rows = list(csv.DictReader(path.open(), delimiter='|'))
    tokens = [re.sub('[^a-z0-9]', '', row['token'].lower()) for row in rows]
    patterns = [('hey', 'nyx'), ('hey', 'nix'), ('hey', 'nicks'), ('hi', 'esp'), ('hi', 'e', 's', 'p')]
    matches = []
    for pattern in patterns:
        for at in range(len(tokens) - len(pattern) + 1):
            if tuple(tokens[at:at + len(pattern)]) == pattern:
                matches.append(dict(token_index=at, pattern=' '.join(pattern)))
    return dict(tokens=len(rows), speaker_labels=len({r['speaker'] for r in rows}),
                exact_target_text_matches=matches,
                limit='Original transcript screen only; omissions/transcription errors and physical speaker overlap remain unknown.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'sources', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    assert protocol['schema'] == 'earnings22-development-preparation-protocol-v1'
    assert sha(Path(__file__)) == protocol['preparer_sha256']
    assert sha(TRAINING / 'features.py') == protocol['frontend_source_sha256']
    for package, expected in protocol['host_versions'].items():
        assert version(package) == expected, package
    source_plan_path = args.sources / 'source-reservation-v1.json'
    assert sha(source_plan_path) == protocol['source_reservation_sha256']
    reservation = json.loads(source_plan_path.read_text())
    assert not reservation['training'] and not reservation['final_data_accessed'] and not reservation['wake_models_scored']
    for name, key in [('metadata.csv', 'upstream_metadata_sha256'), ('README.md', 'upstream_readme_sha256'),
                      ('LICENSE.md', 'upstream_license_sha256')]:
        assert sha(args.sources / name) == reservation[key]
    raw_metadata = list(csv.DictReader((args.sources / 'metadata.csv').open()))
    order = lambda s: hashlib.sha256(('nyx-earnings22-development-v1:' + s).encode()).hexdigest()
    countries = sorted({r['Country by Ticker'] for r in raw_metadata}, key=lambda c: order('country:' + c))[:12]
    selected = [min([r for r in raw_metadata if r['Country by Ticker'] == c],
                    key=lambda r: order('file:' + r['File ID'])) for c in countries]
    assert [r['metadata'] for r in reservation['records']] == selected
    audits = {}
    for row in reservation['records']:
        identity = row['source_id']
        assert re.fullmatch('[A-Za-z0-9_-]+', identity)
        assert sha(args.sources / row['pointer_path']) == row['pointer_sha256']
        assert sha(args.sources / row['transcript_path']) == row['transcript_sha256']
        audits[identity] = transcript_audit(args.sources / row['transcript_path'])
    assert not any(r['exact_target_text_matches'] for r in audits.values()), 'Preserve whole cohort; review target text before declaring negative readiness'
    ffmpeg, ffprobe = (shutil.which(name) for name in ('ffmpeg', 'ffprobe'))
    assert ffmpeg and ffprobe
    ffmpeg_version = subprocess.check_output([ffmpeg, '-version'], text=True)
    assert ffmpeg_version.splitlines()[0].startswith(protocol['ffmpeg_version_prefix'])
    args.output.mkdir(parents=True)
    for name in ('original', 'pcm', 'features', 'records', 'logs'):
        (args.output / name).mkdir()
    shutil.copyfile(args.protocol, args.output / 'protocol.json')
    shutil.copyfile(source_plan_path, args.output / 'source-reservation-v1.json')
    shutil.copyfile(__file__, args.output / 'preparer.py')
    runtime = dict(ffmpeg_sha256=sha(Path(ffmpeg)), ffprobe_sha256=sha(Path(ffprobe)),
                   ffmpeg_version=ffmpeg_version, host_versions=protocol['host_versions'],
                   platform=sys.platform)
    dump(args.output / 'runtime.json', runtime)
    completed = []
    started = time.monotonic()
    for row in reservation['records']:
        identity = row['source_id']
        source = args.output / 'original' / (identity + '.mp3')
        assert row['source_url'].startswith('https://media.githubusercontent.com/media/revdotcom/speech-datasets/' + reservation['upstream_commit'] + '/earnings22/media/')
        temporary = source.with_suffix('.mp3.partial')
        with urllib.request.urlopen(row['source_url'], timeout=60) as response, temporary.open('xb') as output:
            while data := response.read(1024 * 1024):
                output.write(data)
        assert temporary.stat().st_size == row['source_bytes'] and sha(temporary) == row['source_sha256']
        temporary.rename(source)
        probe_command = [ffprobe, '-v', 'error', '-select_streams', 'a', '-show_streams', '-show_format', '-of', 'json', str(source)]
        probe = json.loads(subprocess.check_output(probe_command, text=True))
        dump(args.output / 'records' / (identity + '-probe.json'), probe)
        assert len(probe['streams']) == 1
        stream = probe['streams'][0]
        native_rate, channels = int(stream['sample_rate']), int(stream['channels'])
        assert native_rate > 0 and channels in (1, 2)
        common = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'warning', '-nostats',
                  '-threads', '1', '-cpuflags', '0', '-fflags', '+bitexact', '-flags:a', '+bitexact',
                  '-i', str(source), '-map', '0:a:0', '-vn', '-sn', '-dn', '-map_metadata', '-1']
        # Count/hash complete native decoded float samples without storing a
        # second large waveform. The same decoder respects native gapless data.
        native_command = common + ['-c:a', 'pcm_f32le', '-f', 'f32le', '-']
        native_hash, native_bytes = hashlib.sha256(), 0
        with (args.output / 'logs' / (identity + '-native.log')).open('wb') as log:
            with subprocess.Popen(native_command, stdout=subprocess.PIPE, stderr=log) as process:
                while data := process.stdout.read(1024 * 1024):
                    native_hash.update(data)
                    native_bytes += len(data)
                assert process.wait() == 0
        assert native_bytes > 0 and native_bytes % (4 * channels) == 0
        native_frames = native_bytes // (4 * channels)
        wav = args.output / 'pcm' / (identity + '.wav')
        decode_command = common + ['-ac', '1', '-ar', '16000', '-af', 'aresample=resampler=swr:dither_method=none',
                                   '-c:a', 'pcm_s16le', '-flags:a', '+bitexact', '-threads', '1', '-f', 'wav', str(wav)]
        with (args.output / 'logs' / (identity + '-16k.log')).open('wb') as log:
            subprocess.run(decode_command, stdout=log, stderr=subprocess.STDOUT, check=True)
        info = sf.info(wav)
        assert info.samplerate == 16000 and info.channels == 1 and info.subtype == 'PCM_16'
        assert abs(info.frames / 16000 - native_frames / native_rate) <= 1 / 16000
        pcm_hash, nonzero, saturation = hashlib.sha256(), 0, 0
        with sf.SoundFile(wav) as audio:
            while len(block := audio.read(160000, dtype='int16')):
                pcm_hash.update(block.astype('<i2').tobytes())
                nonzero += int(np.count_nonzero(block))
                saturation += int(np.count_nonzero((block == -32768) | (block == 32767)))
        assert nonzero > 0
        raw, final_pad = continuous_wav_features(wav)
        assert raw.dtype == np.uint16 and raw.shape == ((info.frames + 159) // 160 - 2, 40)
        assert final_pad == (-info.frames) % 160
        feature_path = args.output / 'features' / (identity + '.npy')
        np.save(feature_path, raw, allow_pickle=False)
        record = dict(source_id=identity, role='continuous-real-speech-development-only',
                      source_sha256=sha(source), source_bytes=source.stat().st_size,
                      source_native_rate=native_rate, source_channels=channels,
                      decoded_native_frames=native_frames, decoded_native_float_sha256=native_hash.hexdigest(),
                      decoded_native_seconds=native_frames / native_rate,
                      metadata_seconds=float(row['metadata']['File Length (seconds)']),
                      native_metadata_rate_matches=native_rate == int(row['metadata']['Sampling Rate (Hz)']),
                      path=str(wav.relative_to(args.output)), sha256=sha(wav), samples=info.frames, sample_rate=16000,
                      pcm_sha256=pcm_hash.hexdigest(), source_seconds=info.frames / 16000,
                      duration_error_16k_samples=info.frames - native_frames * 16000 / native_rate,
                      pcm_nonzero_samples=nonzero, original_saturated_samples=saturation,
                      features_path=str(feature_path.relative_to(args.output)), features_sha256=sha(feature_path),
                      features_array_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                      features_shape=list(raw.shape), frontend_final_zero_pad_samples=final_pad,
                      transcript_audit=audits[identity], native_command=native_command, decode_command=decode_command)
        dump(args.output / 'records' / (identity + '.json'), record)
        completed.append(record)
        print(json.dumps(dict(completed=len(completed), total=12, source_id=identity,
                              source_hours=round(info.frames / 16000 / 3600, 6),
                              elapsed_seconds=round(time.monotonic() - started, 1))), flush=True)
        del raw
    assert len(completed) == 12
    manifest = dict(schema='earnings22-continuous-development-prepared-v1',
                    role=reservation['role'], source_reservation_sha256=sha(source_plan_path),
                    protocol_sha256=sha(args.protocol), preparer_sha256=sha(Path(__file__)),
                    runtime=runtime, records=completed,
                    source_hours=sum(r['source_seconds'] for r in completed) / 3600,
                    complete_recordings=True, original_media_hashes_verified=True,
                    all_decoded_durations_match_native_within_one_sample=True,
                    inserted_intermediate_gaps=False, segmentation=False, normalized=False,
                    wake_models_scored=False, training=False, final_data_accessed=False,
                    accepted=None, far_estimate=None,
                    limits=protocol['limits'])
    dump(args.output / 'manifest.json', manifest)
    print(json.dumps(dict(completed=12, source_hours=manifest['source_hours'],
                          manifest_sha256=sha(args.output / 'manifest.json'), wake_models_scored=False)))


if __name__ == '__main__':
    main()
