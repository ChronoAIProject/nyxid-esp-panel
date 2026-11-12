"""One frozen PocketSphinx feasibility replay, using existing public Mac PCM.

Uses Decoder only: no microphone, playback, live device or model fitting.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import time
import wave
import zlib

from pocketsphinx import Config, Decoder, get_model_path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def decode(model, dictionary, pcm, prime):
    config = Config(hmm=str(model), dict=str(dictionary), lm=None,
                    keyphrase='hey nicks', loglevel='ERROR')
    defaults = {key: config[key] for key in (
        'samprate', 'kws_threshold', 'kws_delay', 'kws_plp',
        'beam', 'pbeam', 'wbeam', 'frate')}
    decoder = Decoder(config)
    decoder.start_utt()
    tail = (-len(pcm)//2) % 480
    padded = bytes(2*prime) + pcm + bytes(2*tail)
    events, seen = [], set()

    def collect(at, flushed):
        for segment in decoder.seg() or []:
            key = (segment.word, segment.start_frame, segment.end_frame)
            if key not in seen:
                require(segment.word == 'hey nicks', 'Unexpected keyword segment')
                seen.add(key)
                events.append(dict(word=segment.word, start_frame=segment.start_frame,
                                   end_frame=segment.end_frame,
                                   returned_source_sample=at-prime,
                                   utterance_flush=flushed))

    started = time.perf_counter()
    for stop in range(960, len(padded)+1, 960):
        decoder.process_raw(padded[stop-960:stop], False, False)
        collect(stop//2, False)
    decoder.end_utt()
    collect(len(padded)//2, True)
    elapsed = time.perf_counter()-started
    return events, elapsed, defaults, tail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sources', 'plan', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    require(platform.system() == 'Darwin', 'This probe uses macOS RSS units')
    require(not args.output.exists(), 'Fresh output required')
    plan = json.loads(args.plan.read_bytes())
    require(version('pocketsphinx') == plan['pocketsphinx_version'], 'Package changed')
    model_root = Path(get_model_path())/'en-us'
    model = model_root/'en-us'
    for name, expected in plan['acoustic_files'].items():
        require(sha(model/name) == expected, 'Acoustic bytes changed: '+name)
    native_dictionary = model_root/'cmudict-en-us.dict'
    require(sha(native_dictionary) == plan['native_dictionary_sha256'], 'Dictionary changed')
    entries = [s for s in native_dictionary.read_text().splitlines()
               if s.split()[0] in ('hey', 'nicks')]
    require(entries == plan['dictionary_entries'], 'Word pronunciations changed')
    manifest, baseline = args.sources/'manifest.jsonl', args.sources/'baseline.json'
    require(sha(manifest) == plan['manifest_sha256']
            and sha(baseline) == plan['baseline_sha256'], 'Source protocol changed')
    rows = [json.loads(s) for s in manifest.read_text().splitlines()]
    official = json.loads(baseline.read_bytes())
    require(len(rows) == len(official['clips']) == 350, 'Incomplete input collection')
    require(official['restored'] and official['manifest_sha256'] == sha(manifest),
            'Retained Hi ESP provenance changed')
    # Authenticate the entire cohort before the first keyword decode.
    pcms = []
    for row, prior in zip(rows, official['clips'], strict=True):
        relative = Path(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe source path')
        path = args.sources/relative
        require(all(prior[k] == v for k, v in row.items())
                and prior['result']['complete'] and sha(path) == row['sha256'],
                'Source identity/Hi ESP completion differs')
        with wave.open(str(path), 'rb') as source:
            require((source.getnchannels(), source.getsampwidth(), source.getframerate(),
                     source.getnframes(), source.getcomptype())
                    == (1, 2, 16000, row['samples'], 'NONE'), 'PCM geometry changed')
            pcm = source.readframes(source.getnframes())
        require(zlib.crc32(pcm) == prior['result']['crc32'], 'PCM CRC differs')
        pcms.append(pcm)
    args.output.mkdir(parents=True)
    dictionary = args.output/'keyword.dict'
    dictionary.write_text('\n'.join(entries)+'\n')
    details, total_seconds = [], 0.0
    for index, (row, pcm, prior) in enumerate(zip(rows, pcms, official['clips'], strict=True)):
        first, elapsed, defaults, tail = decode(model, dictionary, pcm, plan['prime_samples'])
        second, _, repeated_defaults, _ = decode(model, dictionary, pcm, plan['prime_samples'])
        require(defaults == repeated_defaults == plan['decoder_defaults'], 'Decoder policy changed')
        require(first == second, 'Fresh-decoder replay differs')
        source_events = [event for event in first if event['returned_source_sample'] > 0]
        details.append({**row, 'events': first, 'source_events': len(source_events),
                        'retained_hiesp_events': prior['result']['events'],
                        'tail_zero_samples': tail, 'reset_exact': True})
        total_seconds += elapsed
        if index % 25 == 0:
            print(json.dumps(dict(completed=index+1, total=350)), flush=True)
    summary = {}
    for key in ('target', 'text', 'voice'):
        summary[key] = {value: dict(
            clips=sum(r[key] == value for r in details),
            event_clips=sum(r[key] == value and r['source_events'] > 0 for r in details),
            retained_hiesp_event_clips=sum(r[key] == value and r['retained_hiesp_events'] > 0
                                          for r in details))
            for value in sorted({r[key] for r in details})}
    report = dict(scope=plan['scope'], accepted=False,
                  plan_sha256=sha(args.plan), source_sha256=sha(Path(__file__)),
                  pocketsphinx_version=version('pocketsphinx'),
                  platform=platform.platform(), source_count=len(details),
                  original_source_seconds=sum(len(pcm)/32000 for pcm in pcms),
                  first_pass_decode_seconds=total_seconds,
                  process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  prime_events=sum(e['returned_source_sample'] <= 0
                                   for r in details for e in r['events']),
                  all_sources_hash_crc_verified=True, all_reset_replays_exact=True,
                  acoustic_bytes=sum((model/name).stat().st_size for name in plan['acoustic_files']),
                  dictionary_bytes=dictionary.stat().st_size,
                  summary=summary, source_details=details, limits=plan['limits'])
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('summary', 'first_pass_decode_seconds',
                                           'process_peak_rss_bytes', 'acoustic_bytes')}), flush=True)


if __name__ == '__main__':
    main()
