"""Play four synthesized Nyx cues at matched level through the P4 listener and Opus encoder."""
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import soundfile as sf

import coded_advance as advance
import paired_piper_willow as matched
from check_c161_command_playback_level import levels
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha
from score_paired_piper_willow import host, uplink
from voice_cues import source_bank

STEM = "4090-c161-piper-matched-coded"
CUES = RESULTS / (STEM + "-cues.zip")
UART = RESULTS / (STEM + "-uart.zip")
PREVIOUS = RESULTS / (matched.STEM + ".json")
digest = host.base.poll.digest


def write_cues(plan):
    bank = source_bank()
    cues = {}
    entries = []
    with ZipFile(CUES.with_suffix(".tmp"), "w", ZIP_DEFLATED) as archive:
        for key, source in plan["cues"].items():
            if not key.endswith("-ordinary"):
                continue
            voice = key.removesuffix("-ordinary")
            speech = bank[voice, "Hey Nyx."]
            assert digest(speech) == source["source_sha256"]
            lo, hi = source["source_active"]
            _, coded = advance.cue_pair(speech)
            assert hi == len(coded)
            active = coded[lo:hi]
            gain = 10 ** ((plan["matched_active_rms_dbfs"] - levels(active)["active_rms_dbfs"]) / 20)
            values = np.rint(active * gain)
            assert np.max(np.abs(values)) <= 32767
            cue = values.astype("<i2")
            measured = levels(cue)
            assert abs(measured["active_rms_dbfs"] - plan["matched_active_rms_dbfs"]) < .01
            cues[key] = cue
            entries.append(dict(voice=voice, source_sha256=source["source_sha256"], trim=[lo, hi],
                                gain=gain, levels=measured, **uplink.write_member(archive, voice + ".wav", cue)))
        manifest = dict(sample_rate=16000, target_active_rms_dbfs=plan["matched_active_rms_dbfs"],
                        plan_sha256=sha(matched.PLAN), code_shift_samples=advance.SHIFT, records=entries)
        archive.writestr(ZipInfo("manifest.json"), json.dumps(manifest, indent=2) + "\n")
    CUES.with_suffix(".tmp").replace(CUES)
    return cues


def main():
    plan, baseline = read(matched.PLAN), read(PREVIOUS)
    assert sha(matched.PLAN) == baseline["plan_sha256"] and sha(MODEL) == MODEL_SHA == baseline["nyx_model_sha256"]
    firmware = uplink.firmware()
    assert firmware["opus_sha256"] == baseline["opus_sha256"]
    reader, identity = host.piper.load_reader()
    assert identity == baseline["asr"]
    native, decoder = host.fixed.ch.c161_binding(), uplink.decoder_api()
    cues = write_cues(plan)
    records, summary = [], defaultdict(Counter)
    with TemporaryDirectory() as tmp, ZipFile(UART.with_suffix(".tmp"), "w", ZIP_DEFLATED) as uart:
        work = Path(tmp)
        uplink.WORK = work
        detector = advance.challenge.binding(work / "v5.so", coded="v5")
        prefix = advance.challenge.binding(work / "prefix.so", coded="aligned")
        for row, old in zip(plan["records"], baseline["records"], strict=True):
            if row["target_id"] != "ordinary":
                continue
            i, key = row["index"], row["cue"]
            original = host.piper.load_wav(row)
            assert i == old["index"] and digest(original) == row["pcm_sha256"] == old["mic_sha256"]
            assert not old["routes"]["nyx"]["events"]
            start, end = row["start"], row["end"]
            cue = cues[key]
            assert len(cue) == end - start and digest(original[start:end]) == row["cue_sha256"]
            mic = original.copy()
            mic[start:end] = cue
            assert digest(mic[end:]) == row["suffix_sha256"] == old["command_sha256"]
            meta = dict(index=i, cue_at=start, command_at=end, voice=key.removesuffix("-ordinary"), carrier=row["carrier"])
            routes = host.fixed.ch.routes(detector, native, mic)
            expected = dict(c161=routes["c161"], fixed=routes["fixed"], events=dict(nyx=routes["merged"]))
            if routes["merged"]:
                external = uplink.p4(meta, mic, detector, i, decoder, reader, expected, uart, require_timely=False)
                opened = external["delivered"]
                complete = external["uplink"]["selected"]["full"]["complete_suffix"]
            else:
                external = advance.self_score(work, meta, mic, np.zeros_like(mic), detector, i,
                                              build=advance.BUILD, prefix=prefix, qualify=advance.qualify)
                assert not external["opened"]
                opened, complete = None, False
            reference = np.zeros_like(mic)
            reference[start:end] = cue
            panel = advance.self_score(work, meta, mic, reference, detector, i,
                                       build=advance.BUILD, prefix=prefix, qualify=advance.qualify)
            assert panel["fixed"] == [r["t"] for r in routes["fixed"]] and panel["c161"] == routes["c161"]
            timely = opened is not None and start <= opened < end
            for group in ("all", meta["voice"]):
                summary[group].update(dict(trials=1, opened=opened is not None, timely=timely, complete=complete,
                                           timely_complete=timely and complete, self_open=bool(panel["opened"])))
            records.append(dict(index=i, pair=row["pair"], source=meta, original_pcm_sha256=digest(original),
                                pcm_sha256=digest(mic), cue_sha256=digest(cue), command_sha256=row["suffix_sha256"],
                                external=external, panel=panel))
            if len(records) % 12 == 0:
                print(len(records), flush=True)
        failures = sorted(work.glob("*-attempt*.gz"))
        for path in failures:
            uart.write(path, "retry/" + path.name)
    UART.with_suffix(".tmp").replace(UART)
    save(RESULTS / (STEM + ".json"), dict(records=records, summary=summary, **firmware, asr=identity,
         script_sha256=sha(Path(__file__)), plan_sha256=sha(matched.PLAN), comparison_sha256=sha(PREVIOUS),
         cue_archive_sha256=sha(CUES), uart_sha256=sha(UART), model_sha256=MODEL_SHA, retries=[p.name for p in failures],
         policy="Apply the existing three-part 80 ms advanced code to the four original Nyx voices. Preserve the "
                "original trimmed length, active RMS, command clock and samples outside the cue. Run two independent "
                "P4 QEMU listener boots per input with silent and synchronous cue-only references. Decode the "
                "external selected and raw Opus lanes. Inputs are the same 18 consumed command recordings."))


if __name__ == "__main__":
    main()
