"""Replay frozen Nyx paths and score the paired Willow returns on the same four streams."""
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from pocketsphinx import Decoder

from check_c161_piper_spellings import PLAN, ARCHIVE, RUNS, RESULTS, STEM
from check_c161_q8_keyword_map import recheck
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_v5_command_ns import digest
from check_c161_v5_far_command_noise import normalized
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, read, save, sha
from run_c161_first_command_v5 import transcripts
from score_c161_lexical_echo import infer
from select_c161_synthesized_cue_v2 import load_wav
from train_c161_q8_keyword_map import FILES, ROOT as MAP_ROOT
from train_c161_q8_name_contrast import ROOT as CONTRAST_ROOT


def official(index, source, archive):
    folder = RUNS / f"{index:02}"
    report = read(folder / "report.json")
    assert report["execution_sha256"] == sha(PLAN) and report["record_index"] == index
    assert report["model"] == "wn9_heywillow_tts" and report["stage"] == 104
    assert report["actual_float32_bits"] == report["post_prime_getter_bits"] == "3f24dd2f"
    assert report["source_pcm_sha256"] == source["pcm_sha256"] and report["total_prime_samples"] == 81600
    assert sha(folder / "raw-returns.bin") == report["raw_returns_sha256"]
    assert sha(folder / "serial.log") == report["serial_sha256"]
    returns = np.fromfile(folder / "raw-returns.bin", dtype=np.uint8)
    assert len(returns) == report["input_calls"] and np.all(returns <= 2)
    offset = report["additional_prime_samples"]
    events = (np.minimum((np.flatnonzero(returns == 2) + 1) * 512, report["input_samples"]) - offset).tolist()
    assert events == [e["sample_end"] - offset for e in report["events"]]
    for name in ("report.json", "serial.log", "stage-monitor.txt", "raw-returns.bin", "host.log"):
        archive.write(folder / name, f"{index:02}/{name}")
    return events


def decoders():
    manifest = read(RESULTS / "4090-c161-lexical-p4-q8-packing.json")["files"]
    models = {}
    for name, root in (("q8", FILES), ("keyword-map", MAP_ROOT), ("name-contrast", CONTRAST_ROOT)):
        weights = manifest["am/sendump"] if root == FILES else read(RESULTS / f"4090-c161-q8-{name}-fit.json")["adapted_sendump_sha256"]
        for path, expected in manifest.items():
            if path == "cases.bin":
                continue
            local = (root if path.startswith("am/") else FILES) / path
            assert sha(local) == (weights if path == "am/sendump" else expected)
        model = Decoder(hmm=str(root / "am"), dict=str(FILES / "two.dict"), keyphrase="hey nicks",
                        kws_threshold=1e-15, loglevel="ERROR", lm=None)
        models[name] = (model, model.get_cmn())
    return models


def annotate(source, pcm, routes, native, expanded, reader):
    rows, readings = [], {}
    outside = {k: set(v) for k, v in routes.items()}

    def asr(begin, end):
        capture = pcm[begin:end]
        key = digest(capture)
        if key not in readings:
            readings[key] = transcripts(reader, capture)
        return key

    for span in source["timeline"]:
        start, end = span["start"], span["end"]
        assert digest(pcm[start:end]) == span["pcm_sha256"]
        address = span["address"]
        want = normalized(source["carrier"])
        row = dict(address=address, start=start, end=end, pcm_sha256=span["pcm_sha256"], routes={})
        def inside(t):
            return start <= t < end + 6400
        row["native_proposals"] = list(filter(inside, native))
        row["expanded_proposals"] = list(filter(inside, expanded))
        row["full_clip_asr"] = asr(start, end)
        for name, events in routes.items():
            hits = list(filter(inside, events))
            outside[name].difference_update(hits)
            item = dict(hits=hits)
            if hits and address in ("Hye Nyx", "Hey Nyx", "Hey Nicks", "Hey Willow"):
                begin = max(0, (hits[0] + 159) // 160 * 160 - 32000)
                key = asr(begin, end + 6400)
                item.update(first=begin, asr=key, exact_command=normalized(readings[key]["text"])[-len(want):] == want)
            row["routes"][name] = item
        rows.append(row)
    return dict(rows=rows, outside={k: sorted(v) for k, v in outside.items()}, transcripts=readings)


def main():
    plan = read(PLAN)
    assert sha(ARCHIVE) == plan["archive_sha256"]
    mid_spec = read(RESULTS / "4090-c161-midlevel-training-v1.json")["reports"][0]
    mid = RESULTS / mid_spec["path"]
    assert sha(mid) == mid_spec["sha256"] and sha(MODEL) == MODEL_SHA
    models = decoders()
    reader, asr_id = load_reader()
    results = []
    summary = defaultdict(lambda: defaultdict(lambda: dict(trials=0, hits=0, exact_command=0)))
    uart = RESULTS / f"{STEM}-willow.zip"
    with ZipFile(uart.with_suffix(".tmp"), "w", ZIP_DEFLATED) as archive:
        for index, source in enumerate(plan["records"]):
            pcm = load_wav(source)
            assert digest(pcm) == source["pcm_sha256"]
            original = infer(pcm, mid, *models["q8"])
            base = original["events"]
            native, expanded = base["original"], base["expanded"]
            routes = dict(c161=base["c161"], willow=official(index, source, archive))
            checks = {}
            for name, (decoder, cmn) in models.items():
                if name == "q8":
                    checked = original["checked"]
                    groups = dict(native=base["native_proposals"], dual=base["both_proposals"])
                else:
                    scored = recheck(decoder, cmn, pcm, base["c161"], native, expanded)
                    groups, checked = scored["events"], scored["checked"]
                checks[name] = checked
                for speed, events in groups.items():
                    routes[f"{name}_{speed}"] = [e["sample"] for e in events]
            score = annotate(source, pcm, routes, native, expanded, reader)
            for row in score["rows"]:
                for name, item in row["routes"].items():
                    total = summary[name][row["address"]]
                    total["trials"] += 1
                    total["hits"] += bool(item["hits"])
                    total["exact_command"] += item.get("exact_command", False)
            results.append(dict(voice=source["voice"], carrier=source["carrier"], pcm_sha256=digest(pcm),
                                native=native, expanded=expanded, checks=checks, events=routes, **score))
            print("scored", index, flush=True)
    uart.with_suffix(".tmp").replace(uart)
    save(RESULTS / f"{STEM}-evaluation.json", dict(plan_sha256=sha(PLAN), willow_archive_sha256=sha(uart),
         c161_sha256=MODEL_SHA, mid_sha256=mid_spec["sha256"], whisper=asr_id, scorer_sha256=sha(Path(__file__)),
         weights_sha256={label: sha(root / "am/sendump") for label, root in
                         (("q8", FILES), ("keyword-map", MAP_ROOT), ("name-contrast", CONTRAST_ROOT))},
         records=results, summary=summary))


if __name__ == "__main__":
    main()
