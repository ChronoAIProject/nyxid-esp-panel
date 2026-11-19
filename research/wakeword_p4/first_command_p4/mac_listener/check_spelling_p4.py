"""Replay frozen character windows and the gap decoder on ESP32-P4 QEMU."""
import argparse
from hashlib import sha256
from pathlib import Path
import re
import struct
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

import replay_spelling_gap_dialogues as dialogue
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from run_c161_first_command_v5 import boot
from run_c161_coded_cue_p4_v5 import QEMU

gap, prior, base = dialogue.gap, dialogue.prior, dialogue.base
STEM = "4090-c161-spelling-p4"
WORDS = prior.TARGETS + prior.RIVALS
STRIDE = 144 * 40 + 71 * 32


def original_streams(report, gaps):
    for group in gap.study.GROUPS:
        sources = base.sources(group)
        letters = report(f"{prior.fit.STEM}-{group}")["records"]
        spelling = report(f"{prior.STEM}-{group}")["records"]
        for (_, pcm, c161, _, _), a, b, g in zip(sources, letters, spelling, gaps[group], strict=True):
            assert a["index"] == b["index"] == g["index"]
            assert base.poll.digest(pcm) == a["pcm_sha256"] == b["pcm_sha256"] == g["pcm_sha256"]
            assert c161 == a["c161"] == b["c161"]
            checks = []
            for x, y, z in zip(a["checks"], b["checks"], g["checks"], strict=True):
                assert x["end"] == y["end"] == z["end"] and y["score"] == z["score"]
                checks.append((x["end"], x["patch_sha256"], x["models"]["student"]["output_codes_sha256"],
                               y, z["gap"]))
            yield group, b["index"], pcm, c161, b["routes"]["paths"], g["events"], checks


def additional_streams(report):
    cohorts = [("dialogue", dialogue.corpus.STEM+"-plan", dialogue.corpus.STEM+"-replay"),
               ("joe", "4090-c161-history-gain-joe-v1", "4090-c161-name-final-s-joe")]
    for group, plan, replay in cohorts:
        for src, r in zip(report(plan)["records"], report(replay)["records"], strict=True):
            if group == "joe":
                assert src["index"] == r["index"] and src["c161"] == r["events"]["c161"]
                src = src["source"]
            pcm = dialogue.piper.load_wav(src)
            assert base.poll.digest(pcm) == src["pcm_sha256"] == r["pcm_sha256"]
            checks = [(c["end"], None, c["char_output_sha256"], c["spelling"], c["path"]["interword_blank_frames"])
                      for c in r["checks"] if "end" in c]
            e = r["events"]
            yield group, r["index"], pcm, e["c161"], e["paths"], e["gap"], checks


def collect(work):
    inputs = {}

    def report(name):
        path = RESULTS / (name + ".json")
        inputs[path.name] = sha(path)
        return read(path)

    vm, native = prior.model(), prior.fit.native_frontend()
    old, config = report(prior.CAL.stem), report(gap.FIT.stem)
    old_gap = report(gap.STEM+"-replay")
    assert old_gap["fit_sha256"] == sha(gap.FIT)
    vocab = read(prior.WAV2VEC / "vocab.json")
    assert sha(prior.WAV2VEC / "vocab.json") == old["vocab_sha256"]
    assert WORDS == old["target_spellings"] + old["competing_spellings"]
    streams, windows = [], []
    with (work/"blocks.bin").open("wb") as blocks:
        for sources in (original_streams(report, old_gap["records"]), additional_streams(report)):
            for group, index, pcm, c161, paths, gaps, checks in sources:
                start = len(windows)
                raw = base.native_inputs(native, pcm)[0] if checks else None
                ends = [end for end, *_ in checks]
                assert ends == sorted(set(ends))
                for end, patch_hash, code_hash, s, blank in checks:
                    patch = np.rint(prior.fit.room.window(raw, end).clip(0, 666)*(255./666)).astype(np.uint8)
                    codes = prior.outputs(vm, patch)
                    x = vm.table[patch]
                    assert x.shape == (144,40) and codes.shape == (71,32) and x.dtype == codes.dtype == np.int8
                    assert base.poll.digest(codes) == code_hash and (patch_hash is None or base.poll.digest(patch) == patch_hash)
                    old_pass = s["score"] >= old["threshold"]
                    gap_pass = s["score"] >= config["threshold"] and blank <= config["max_gap"]
                    if "accepted" in s:
                        assert s["accepted"] == old_pass
                    windows.append(dict(end=end, input_sha256=base.poll.digest(x), output_sha256=code_hash,
                                        target=s["target"], rival=s["rival"], target_value=s["target_value"],
                                        rival_value=s["rival_value"], gap=blank, paths=old_pass, capped=gap_pass))
                    blocks.write(x.tobytes() + codes.tobytes())
                stream = dict(group=group, index=index, pcm_sha256=base.poll.digest(pcm), samples=len(pcm),
                              c161=c161, windows=list(range(start,len(windows))), host=dict(paths=paths,capped=gaps))
                for route in stream["host"]:
                    predicted = [w["end"] for w in windows[start:] if w[route]]
                    assert base.scoring.merged(c161,predicted) == stream["host"][route]
                streams.append(stream)
    save(work/"fixtures.json", dict(records=windows, streams=streams, inputs=inputs, blocks_sha256=sha(work/"blocks.bin"),
         tokens=[[vocab[c] for c in word] for word in WORDS], separator=vocab["|"], max_gap=config["max_gap"],
         thresholds=dict(paths=old["threshold"], capped=config["threshold"])))


def packet(spec, count, data):
    h = struct.pack("<6I2d", 0x53504c31, count, len(prior.TARGETS), len(WORDS), spec["separator"],
                    spec["max_gap"], spec["thresholds"]["paths"], spec["thresholds"]["capped"])
    return h + b"".join(struct.pack("<B9s", len(chars), bytes(chars)) for chars in spec["tokens"]) + data


def run(work, batch_size, output=RESULTS, stem=None):
    output.mkdir(parents=True, exist_ok=True)
    stem = stem or STEM
    archive = output/(stem+"-uart.zip")
    report = output/(stem+".json")
    if archive.exists() or report.exists():
        raise FileExistsError(f"Choose an unused --output directory for replay reports: {output}")
    build = work/"build"
    spec = read(work/"fixtures.json")
    images = read(build/"flasher_args.json")["flash_files"]
    cfg = read(build/"config/sdkconfig.json")
    assert cfg["NN_ANSI_C"] and cfg["IDF_TARGET"] == "esp32p4"
    model_spec = read(prior.fit.EXPORT)["models"]["student"]
    model = RESULTS/model_spec["file"]
    assert sha(model) == model_spec["sha256"]
    assert sum((build/f).read_bytes().count(model.read_bytes()) for f in images.values()) == 1
    for name, digest in spec["inputs"].items():
        assert sha(RESULTS/name) == digest
    blocks = work/"blocks.bin"
    rows, streams = spec["records"], spec["streams"]
    assert sha(blocks) == spec["blocks_sha256"] and blocks.stat().st_size == STRIDE * len(rows)
    batches = []
    with blocks.open("rb") as src, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as z:
        for start in range(0,len(rows),batch_size):
            count = min(batch_size,len(rows)-start)
            batch = packet(spec, count, src.read(STRIDE*count))
            uart = work/f"spelling-{start:04}.log"
            boot(build,uart,work/"flash.bin",b"SPELL_DONE",(2<<20,batch))
            log = uart.read_text()
            assert "SPELL_FAIL" not in log, log[-1800:]
            arena, init = map(int,re.search(r"^SPELL_READY (\d+) (\d+)$",log,re.M).groups())
            n, heap = map(int,re.search(r"^SPELL_END (\d+) (\d+)$",log,re.M).groups())
            values = re.findall(r"^SPELL_ROW ([0-9 -]+)$",log,re.M)
            assert count == n == len(values)
            for i, line in enumerate(values):
                idx, diff, t, tv, r, rv, blank, old, capped, invoke, decode = map(int,line.split())
                assert idx == i and 0 <= t < len(prior.TARGETS) <= r < len(WORDS)
                assert old in (0,1) and capped in (0,1) and invoke > 0 and decode > 0
                w = rows[start+i]
                device = dict(target=WORDS[t], rival=WORDS[r], target_value=float(np.float32(tv/len(WORDS[t]))),
                              rival_value=float(np.float32(rv/len(WORDS[r]))), gap=blank, paths=bool(old), capped=bool(capped))
                w.update(mismatched_bytes=diff, differing_fields=[k for k,v in device.items() if w[k]!=v],
                         device=device, invoke_us=invoke, decode_us=decode)
            batches.append(dict(start=start, count=n, arena=arena, init_us=init, min_internal_free=heap,
                                fixture_sha256=sha256(batch).hexdigest(), uart_sha256=sha(uart)))
            z.write(uart,uart.name)
            print(start, n, "windows", flush=True)
    archive.with_suffix(".tmp").replace(archive)
    for s in streams:
        complete = 0
        hits = {route:[] for route in s["host"]}
        for i in s["windows"]:
            w = rows[i]
            complete = max(complete,w["end"]) + ((w["invoke_us"]+w["decode_us"])*16+999)//1000
            w["complete_sample"] = complete
            for route in hits:
                if w["device"][route]:
                    hits[route].append(complete)
        s["delayed"] = {route:base.scoring.merged(s["c161"],ts) for route,ts in hits.items()}
    save(report,dict(records=rows, streams=streams, batches=batches, inputs=spec["inputs"],
         thresholds=spec["thresholds"], max_gap=spec["max_gap"], words=WORDS, model_sha256=sha(model),
         model_bytes=model.stat().st_size, used_frames=71, alphabet=32, qemu_sha256=sha(QEMU),
         firmware_sha256=sha(Path(__file__).resolve().parents[2]/"midlevel_p4_selftest/main/proposal_spellings.cpp"),
         images={f:sha(build/f) for f in images.values()}, sdkconfig_sha256=sha(work/"sdkconfig"),
         archive_sha256=sha(archive), script_sha256=sha(Path(__file__)),
         policy="Serial character worker starts at the later of window end and previous worker completion. "
                "Charge emulated Invoke and 31-word decoder guest timer durations for every complete proposal. "
                "C161 retains its archive clock and the shared cooldown. Continuous frontend, concurrent tasks, "
                "audio I/O and hardware timing are outside this component measurement."))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=["collect","run"])
    p.add_argument("work",type=Path)
    p.add_argument("--batch-size",type=int,choices=range(1,201),default=100,metavar="1..200")
    p.add_argument("--output",type=Path,default=RESULTS,help="directory for new UART and timing reports")
    args = p.parse_args()
    if args.action == "collect":
        collect(args.work)
    else:
        run(args.work,args.batch_size,args.output)
