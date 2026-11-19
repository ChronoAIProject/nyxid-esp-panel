"""Verify a two-code veto on the electrical playback reference in P4 QEMU."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

import replay as listener
import self_guard
from check_c161_v3_two_anchor_v1 import binding
from check_c161_v5_code_gain import PREFIX, RESULTS, waves
from code_gain_guard import missed_rooms, score
from reference_signature import qualify, transports, verify_backgrounds
from replay_c161_pcm_cues_v1 import detect

BUILD = listener.WORK/"build-reference-prefix"
REPORT = RESULTS/f"{PREFIX}-guard.json"
PRIOR = RESULTS/"4090-c161-mac-self-cue-guard.json"
OUT = RESULTS/"4090-c161-mac-reference-prefix"


def room_cases():
    for old, (wav, role, src, mic, ref) in zip(listener.read(REPORT)["records"], missed_rooms(), strict=True):
        assert (old["mic_sha256"], old["reference_sha256"], old["source"]) == (listener.digest(mic), listener.digest(ref), src)
        yield dict(wav=wav, **src, playback_role=role), mic, ref, old


def regressions():
    prior = {r["key"]:r for r in listener.read(PRIOR)["records"]}
    for key, mic, ref, src in self_guard.streams():
        old = prior.pop(key)
        assert (old["mic_sha256"], old["reference_sha256"]) == (listener.digest(mic), listener.digest(ref))
        yield dict(key=key, **src), mic, ref, dict(c161=old["c161"], fixed=old["v5"],
              reference_events=old["reference_v5"], opened=[old["opened"]] if old["opened"] else [])
    assert not prior


def collisions():
    # Use a real C161 positive and a real false pair match on the speaker bus.
    # Place the positive inside and after the veto, with no acoustic return.
    _, mic, _, meta = next(row for row in self_guard.streams() if row[3]["role"] == "reference_tone")
    at, = meta["baseline_c161"]
    group = "jenny"
    old = listener.read(RESULTS/f"4090-c161-coded-cue-aligned-v4-background-{group}.json")
    incident = next(r for r in old["records"] if r["events"])
    source = next(r["source"] for r in listener.read(RESULTS/f"4090-c161-history-gain-{group}-v1.json")["records"]
                  if r["phase"] == "background" and r["source"]["pcm_sha256"] == incident["pcm_sha256"])
    path = Path(source.get("wav_path") or source["path"])
    # Preserve the archived detector's 4-sample decimation phase.
    for delta in (480, 3200):
        stop = incident["events"][0]["t"] - at + delta
        assert stop >= 0 and stop % 4 == 0
        with sf.SoundFile(path) as f:
            assert len(f) == source["samples"] and f.samplerate == 16000 and f.channels == 1 and f.subtype == "PCM_16"
            f.seek(stop)
            ref = f.read(len(mic), dtype="int16")
        yield dict(role="collision", source_id=incident["source_id"], offset=stop, delta=delta, at=at,
                   original_pair_at=incident["events"][0]["t"], mic_index=meta["index"]), mic, ref, None


def measured_self(full, prefix, qualification):
    plan = RESULTS/f"{PREFIX}-measured.json"
    old, new = waves()
    rows = []
    for label, cue in (("v5", old), ("doubled", new)):
        ref = np.pad(cue, (14537, 16000))
        refs = [e["t"] for e in detect(full, [ref])]
        early = [e["t"] for e in detect(prefix, [ref])]
        if qualification:
            early = [t for t in early if qualification(ref, t)]
        for r in listener.read(plan)["records"]:
            raw = r[label]
            hits = (raw["c161"], [e["t"] for e in raw["fixed"]])
            def allowed(prefixes):
                return sorted(t for route in hits for t in self_guard.expected(route, refs, ref, prefixes))
            rows.append(dict(wav=label, rir=r["source"]["rir"], noisy=r["source"]["noisy"], mic_sha256=raw["pcm_sha256"],
                        reference_events=refs, reference_prefix=early, before=allowed([]), after=allowed(early)))
    return dict(source_sha256=listener.sha(plan), records=rows)


def main(group, output, signature):
    qualification = qualify if signature else None
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        full = binding(work/"full.so", coded="v5")
        pair = binding(work/"pair.so", coded="aligned")
        if group == "qualification":
            report = dict(background=verify_backgrounds(pair), transports=transports(pair, full))
        elif group == "host":
            report = measured_self(full, pair, qualification)
        else:
            firmware = BUILD/"nyx_mac_listener.bin"
            rows = []
            cases = {"rooms": room_cases, "regressions": regressions, "collisions": collisions}
            for i, (src, mic, ref, old) in enumerate(cases[group]()):
                row = score(work, src, mic, ref, full, i, build=BUILD, prefix=pair, qualify=qualification)
                if old is not None:
                    assert all(row[k] == old[k] for k in ("c161", "fixed", "reference_events"))
                    row["previous_opened"] = old["opened"]
                else:
                    assert row["c161"] == [src["at"]] and row["prefix_reference"] == [src["at"]-src["delta"]]
                rows.append(row)
                if (i+1)%10 == 0:
                    print(group, i+1, flush=True)
            report = dict(firmware_sha256=listener.sha(firmware), qemu_sha256=listener.sha(listener.QEMU), records=rows)
    listener.save(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=("rooms", "regressions", "collisions", "host", "qualification"))
    parser.add_argument("--signature", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    mode = "-signature" if args.signature else ""
    main(args.group, args.out or OUT.with_name(f"{OUT.name}{mode}-{args.group}.json"), args.signature)
