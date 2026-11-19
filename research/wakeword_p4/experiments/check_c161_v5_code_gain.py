"""Double only the three embedded codes in v5. Compare identical room channels and noise."""
import argparse
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import soundfile as sf

import check_c161_coded_cue_v4 as playback
from check_c161_coded_cue_band_v4 import fresh_streams
from check_c161_coded_cue_v5 import COEFF, HEADER, STEM, WAV
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_v5_command_ns import digest
from check_c161_v5_far_command_noise import normalized
from check_c161_v3_two_anchor_v1 import binding
from audit_c161_history_paths_v1 import decode, times
from diagnose_c161_v5_room_misses import REPORT
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha
from replay_c161_pcm_cues_v1 import detect, joined
from replay_c161_vowel_name_pair_v1 import binding as c161_binding
from run_c161_first_command_v5 import transcripts
from select_c161_synthesized_cue_v2 import load_wav
from verify_c161_ambient_stream_v1 import run_stream

PREFIX = f"{STEM}-code-gain"
CANDIDATE = RESULTS / "hye-nyx-c161-synthetic-cue-v5-code-gain.wav"


def waves():
    source = read(RESULTS/f"{STEM}-source.json")
    old = load_wav(dict(path=WAV, sha256=source["wav_sha256"]))
    v3 = load_wav(dict(path=RESULTS/"hye-nyx-c161-synthetic-cue-v3.wav", sha256=source["v3_sha256"]))
    delta = old.astype(np.int32) - v3
    mask = np.zeros(len(old), bool)
    for start in (11840, 14400, 16320):
        mask[start:start+1280] = True
    assert not delta[~mask].any()
    result = old + delta
    assert np.abs(result).max() <= 32767
    return old, result.astype("<i2")


def routes(detector, native, pcm):
    fixed = detect(detector, [pcm])
    codes, clock, _, _ = run_stream(native, pcm)
    base = times(decode(np.asarray(codes[0], np.uint8), np.asarray(clock)))
    return dict(pcm_sha256=digest(pcm), fixed=fixed, c161=base, merged=joined(base, fixed))


def measured(old, new, detector, native):
    prior = read(REPORT)
    rows = []
    with patch.object(playback, "TARGET", "v5"):
        pairs = zip(prior["records"], playback.measured(old), playback.measured(new, gain_reference=old), strict=True)
        for r, (meta, pcm), (paired, other) in pairs:
            assert r["source"] == meta and digest(pcm) == r["pcm_sha256"]
            assert paired == meta and meta["clipped"] == 0
            if meta["role"] != "v5":
                np.testing.assert_array_equal(pcm, other)
                continue
            baseline = routes(detector, native, pcm)
            assert baseline["fixed"] == r["events"]
            rows.append(dict(source=meta, v5=baseline, doubled=routes(detector, native, other)))
            if len(rows) % 100 == 0:
                print("room replays", len(rows), flush=True)
    return dict(frozen_report_sha256=sha(REPORT), records=rows)


def commands(old, new, detector, native):
    plan = RESULTS/f"{STEM}-merged.json"
    carriers = {(r["voice"],r["scene"]): r["carrier"] for r in
                read(RESULTS/"4090-c161-signed-forward-plan-v1.json")["records"] if r["target_id"] == "nyx"}
    model, identity = load_reader()
    rows = []
    with patch.object(playback, "TARGET", "v5"):
        pairs = zip(read(plan)["records"], playback.command_streams(old), playback.command_streams(new, noise_reference=old), strict=True)
        for r, (meta, pcm), (paired, other) in pairs:
            assert meta == paired and json.loads(json.dumps(meta)) == r["source"] and digest(pcm) == r["pcm_sha256"]
            if meta["role"] != "v5":
                np.testing.assert_array_equal(pcm, other)
                assert detect(detector, [pcm]) == r["fixed"]
                continue
            start, end = meta["start"], meta["end"]
            expected = carriers[meta["voice"], meta["scene"]]
            row = dict(source=meta, expected=expected)
            for label, wave in (("v5", pcm), ("doubled", other)):
                outcome = routes(detector, native, wave)
                if label == "v5":
                    assert all(outcome[k] == r[k] for k in outcome)
                # Score the whole received command from the known dry-onset mapping.
                words = transcripts(model, wave[end:])
                outcome.update(command=words, exact=normalized(words["text"]) == normalized(expected),
                               first_notify_in_cue=bool(outcome["merged"] and start <= outcome["merged"][0] < end))
                row[label] = outcome
            rows.append(row)
    return dict(frozen_report_sha256=sha(plan), asr_files=identity, dry_cue_transcripts={
        "v5":transcripts(model, old), "doubled":transcripts(model, new)}, records=rows)


def controls(old, new, detector, native):
    prior_path = RESULTS/f"{STEM}-fresh.json"
    prior = read(prior_path)["records"]
    rooms = []
    for seed in (316523, 561119):
        frozen = fresh_streams(old, "v5", seed)
        doubled = fresh_streams(new, "v5", seed, noise_reference=old)
        for i, ((meta, pcm), (paired, other)) in enumerate(zip(frozen, doubled, strict=True)):
            source = {k:v for k,v in meta.items() if k != "pcm_sha256"}
            assert all(paired[k] == v for k,v in source.items())
            base = routes(detector, native, pcm)
            if seed == 316523:
                assert dict(**meta, events=base["fixed"]) == prior[i]
            if meta["role"] == "silence":
                np.testing.assert_array_equal(pcm, other)
                changed = base
            else:
                changed = routes(detector, native, other)
            rooms.append(dict(source=source, v5=base, doubled=changed))
    backgrounds = {}
    for group in ("daily", "context", "bot", "music"):
        path = RESULTS/f"{STEM}-{group}.json"
        ref = read(path)
        assert (ref["header_sha256"], ref["coefficients_sha256"]) == (sha(HEADER), sha(COEFF))
        records = []
        for expected, (meta, pcm) in zip(ref["records"], playback.inputs(group), strict=True):
            checksum = digest(pcm)
            assert checksum == meta["pcm_sha256"] == expected["pcm_sha256"]
            events = detect(detector, [pcm])
            assert events == expected["events"]
            records.append(dict(pcm_sha256=checksum, events=events))
        backgrounds[group] = dict(original_report_sha256=sha(path), records=records)
    return dict(frozen_room_report_sha256=sha(prior_path), rooms=rooms, fixed_matcher_controls=backgrounds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=("generate", "measured", "commands", "controls"))
    args = parser.parse_args()
    old, new = waves()
    if args.group == "generate":
        with CANDIDATE.open("xb") as f:
            sf.write(f, new, 16000, subtype="PCM_16", format="WAV")
        return
    np.testing.assert_array_equal(load_wav(dict(path=CANDIDATE, sha256=sha(CANDIDATE))), new)
    prior = read(REPORT)
    assert (sha(MODEL), sha(HEADER), sha(COEFF)) == (MODEL_SHA, prior["header_sha256"], prior["coefficients_sha256"])
    with tempfile.TemporaryDirectory() as tmp:
        detector = binding(Path(tmp)/"frozen.so", coded="v5")
        result = {"measured": measured, "commands": commands, "controls": controls}[args.group](old, new, detector, c161_binding())
    save(RESULTS/f"{PREFIX}-{args.group}.json", dict(v5_sha256=sha(WAV), doubled_sha256=sha(CANDIDATE),
         model_sha256=MODEL_SHA, header_sha256=sha(HEADER), coefficients_sha256=sha(COEFF), **result))


if __name__ == "__main__":
    main()
