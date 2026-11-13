"""Prepare and audit all 126 existing Bryce development WAVs on native P4.

The retained UART probe is separate from the classic ESP32 QEMU profile.
This joins existing v30 results; it never runs or calibrates a Nyx model.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import wave
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'qemu_esp32_replay'))
from synthetic_development import validate_manifest

PROBE_SHA = '28d9fc3d75b25513d4088d8d64780602cf8e8f43fd2fc2a13dcabd1ab0f43a61'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, data):
    with path.open('x') as stream:
        stream.write(json.dumps(data, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'summarize'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--uart-manifest', type=Path, required=True)
    parser.add_argument('--v30-report', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest = validate_manifest(args.manifest, authenticate_pcm=True)
    assert len(manifest['records']) == 126 and sha(args.probe) == PROBE_SHA
    # Keep relative WAV paths and every source/reference field unchanged.
    assert args.uart_manifest.parent.resolve() == args.manifest.parent.resolve()
    uart_bytes = ''.join(json.dumps(row, sort_keys=True) + '\n'
                         for row in manifest['records']).encode()
    bindings = dict(manifest_sha256=sha(args.manifest), probe_sha256=sha(args.probe),
                    runner_sha256=sha(args.runner), v30_report_sha256=sha(args.v30_report),
                    uart_manifest_sha256=hashlib.sha256(uart_bytes).hexdigest(),
                    audit_source_sha256=sha(Path(__file__)))
    if args.action == 'prepare':
        assert not args.profile.exists() and not args.uart_manifest.exists()
        args.uart_manifest.write_bytes(uart_bytes)
        write_new(args.profile, dict(schema='bryce-native-p4-hiesp-profile-v1', **bindings,
            records=126, target_counts=manifest['target_counts'], engine='wn9_hiesp',
            esp_sr='2.6.1', mode='DET_MODE_95', chunk_samples=512,
            threshold_requested=None, threshold_getter_expected=.63,
            zero_prime_samples=32256, uart_baud=1500000,
            classic_esp32_qemu_profile=False, training=False, final_data=False,
            scope='Existing synthetic development only; native P4 standalone detector, no microphone/AEC/full-app acceptance.'))
        print(json.dumps(dict(prepared=126, profile_sha256=sha(args.profile))))
        return
    assert args.report and args.output and not args.output.exists()
    profile = json.loads(args.profile.read_text())
    assert all(profile.get(key) == value for key, value in bindings.items())
    assert args.uart_manifest.read_bytes() == uart_bytes
    report = json.loads(args.report.read_text())
    assert report['probe_sha256'] == PROBE_SHA
    assert report['manifest_sha256'] == bindings['uart_manifest_sha256']
    assert report['engine'] == 'wn9_hiesp' and report['mode'] == 'DET_MODE_95'
    assert report['sample_rate'] == 16000 and report['uart_baud'] == 1500000
    assert report['restored']['nyx_state'] == 'up' and report['restored']['wake_listening'] is True
    assert len(report['clips']) == 126
    for index, (source, row) in enumerate(zip(manifest['records'], report['clips'], strict=True)):
        assert {key: row[key] for key in source} == source and row['threshold'] == .63
        with wave.open(str(args.manifest.parent / source['path']), 'rb') as stream:
            pcm = stream.readframes(stream.getnframes())
        result = row['result']
        assert result['id'] == index and result['complete'] is True
        assert result['samples'] == source['samples'] and result['crc32'] == zlib.crc32(pcm)
        assert result['calls'] == math.ceil(source['samples'] / 512)
        assert result['events'] == len(result['offsets']) <= 64
        assert result['offsets'] == sorted(set(result['offsets']))
        assert all(0 < at <= source['samples'] for at in result['offsets'])
        assert result['total_us'] >= result['max_us'] > 0
    v30 = json.loads(args.v30_report.read_text())
    assert v30['data_manifest_sha256'] == manifest['source_manifest_sha256']
    assert v30['model_sha256'] == '04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c'
    assert v30['threshold'] == .91796875 and len(v30['records']) == 126
    for source, saved in zip(manifest['records'], v30['records'], strict=True):
        assert saved['record']['source_key'] == source['source_key']
        assert saved['record']['condition'] == source['condition']
        assert saved['record']['sha256'] == source['sha256']
        assert saved['record']['pcm_sha256'] == source['pcm_sha256']
        assert saved['record']['samples'] == source['samples']

    def counts(rows, target):
        def count(group):
            return dict(clips=len(group), clips_with_event=sum(bool(r['result']['events']) for r in group),
                        events=sum(r['result']['events'] for r in group))
        positives = [r for r in rows if r['target'] == target]
        latencies = [(at-r['generator_prefix_end_sample'])/16 for r in positives for at in r['result']['offsets']]
        return dict(own_target=count(positives),
                    common_confusers=count([r for r in rows if r['target'] is None]),
                    opposite_wake_phrase=count([r for r in rows if r['target'] not in (None, target)]),
                    generator_reference_latency_ms_median=statistics.median(latencies) if latencies else None,
                    target_events_before_buffered_speech=sum(at < r['lead_samples'] for r in positives for at in r['result']['offsets']))

    rows = report['clips']
    saved_rows = [{**source, 'result': saved['firmware_policy']} for source, saved in zip(manifest['records'], v30['records'], strict=True)]
    def groups(rows, target):
        return dict(all=counts(rows, target),
                    by_voice={voice:counts([r for r in rows if r['voice']==voice], target) for voice in sorted({r['voice'] for r in rows})},
                    by_condition={condition:counts([r for r in rows if r['condition']==condition], target) for condition in sorted({r['condition'] for r in rows})},
                    own_target_by_kind={kind:counts([r for r in rows if r['target']==target and ('command' in r['source_key'])==command], target)['own_target'] for kind,command in [('bare',False),('command',True)]})
    calls = sum(r['result']['calls'] for r in rows)
    result = dict(schema='bryce-native-p4-hiesp-development-result-v1', **bindings,
        profile_sha256=sha(args.profile), raw_report_sha256=sha(args.report),
        records=126, all_complete_crc_and_sample_counts_verified=True,
        original_app_service_and_listener_restored=True,
        hi_esp_native_p4=groups(rows,'hi_esp'), v30_retained_host=groups(saved_rows,'hey_nyx'),
        standalone_inference=dict(calls=calls, mean_us_per_chunk=sum(r['result']['total_us'] for r in rows)/calls,
                                  max_us_per_chunk=max(r['result']['max_us'] for r in rows)),
        background_hours=0, far_estimate=None, accepted=False,
        limits=['Same original PCM and correlated synthetic draws; not human/independent-final acceptance.',
                'Each detector uses its own wake phrase and frozen lifecycle; frontends and priming differ.',
                'P4 default getter .63/32256-sample prime differs from classic ESP32 QEMU .62496001/no extra prime.',
                'Timing is generator duration reference, not phonetic endpoint truth.',
                'No microphone/AEC, acoustic playback, full-app resource or long-call/reconnect test.'])
    write_new(args.output, result)
    print(json.dumps({key:result[key] for key in ('records','hi_esp_native_p4','v30_retained_host','standalone_inference','accepted')},indent=2))


if __name__ == '__main__':
    main()
