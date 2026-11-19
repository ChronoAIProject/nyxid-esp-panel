"""Score fixed-v5 and official Willow on common ASR, including trials without a wake."""
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from faster_whisper import WhisperModel
import attribute_first_command_audio as audit
import paired_piper_willow as paired

ch, uplink = audit.ch, audit.uplink


def willow_cases(group, report, plan):
    for row in report["records"]:
        if group == "kokoro":
            if row["role"] != "Hey Willow":
                continue
            src = plan["records"][row["index"]]
            assert row["pcm_sha256"] == src["pcm_sha256"] and row["role"] == src["target_id"]
            route = row["routes"]["willow"]
            opened = route.get("opened")
            capture = row["captures"].get(str(opened))
            key = [row["voice"], row["carrier"]]
        else:
            if not row["cue"].endswith("-willow"):
                continue
            src = plan["records"][row["index"]]
            assert src["pcm_sha256"] == row["mic_sha256"] and row["cue"] == src["cue"]
            route = row["routes"]["willow"]
            opened = route.get("open")
            capture = route.get("capture")
            key = [row["pair"], row["cue"]]
        assert bool(opened) == bool(capture)
        yield row["index"], key, src, opened, capture


def main():
    report = ch.RESULTS/(audit.STEM+".json")
    previous = ch.read(report)
    reader, small_files = ch.load_reader()
    assert small_files == previous["small_en"]
    for key, value in previous["turbo"]["files"].items():
        assert ch.sha(audit.TURBO/key) == value
    turbo = WhisperModel(str(audit.TURBO), device="cuda", compute_type="float16", num_workers=1, cpu_threads=1)
    decoder = uplink.decoder_api()
    rows, sources = [], {}
    groups = (("kokoro",audit.connected.STEM, audit.connected.PLAN), ("piper",paired.STEM,paired.PLAN))
    for group, stem, plan_path in groups:
        path = ch.RESULTS/(stem+".json")
        uart = ch.RESULTS/(stem+"-uart.zip")
        document = ch.read(path)
        plan = ch.read(plan_path)
        assert ch.sha(plan_path) == document["plan_sha256"] and document["asr"] == small_files
        assert ch.sha(uart) == document.get("uart_sha256", document.get("archive_sha256"))
        sources[group] = dict(report_sha256=ch.sha(path), uart_sha256=ch.sha(uart), plan_sha256=ch.sha(plan_path))
        with ZipFile(uart) as logs:
            for i, key, src, opened, capture in willow_cases(group, document, plan):
                row = dict(group=group, detector="willow", key=key, index=i, source_sha256=src["pcm_sha256"],
                           command=src["carrier"], opened=opened, timely=bool(opened and src["start"]<=opened<src["end"]))
                if opened:
                    log = logs.read(f"opus/{i:03}-willow.log").decode().replace("\r\n","\n")
                    first = capture["first"]
                    assert first == max(0,opened//160*160-32000)
                    count = capture["emitted"]-first
                    for lane, name in enumerate(("selected","raw")):
                        packets = uplink.packets(log,i,count,lane)
                        pcm = uplink.decode(decoder,packets)
                        assert len(packets) == capture["lanes"][name]["packets"]
                        assert ch.digest(pcm) == capture["lanes"][name]["decoded_sha256"]
                        if lane == 0:
                            selected = pcm
                    cut = max(0,opened-1600-first)
                    row.update(decoded_sha256=ch.digest(selected), crop_samples=cut, capture_start=first)
                    for mode, signal in (("full",selected), ("open_minus_100ms",selected[cut:])):
                        scores = {}
                        for model_name, model in (("small.en",reader), ("large-v3-turbo",turbo)):
                            v = uplink.recognition(model,signal,src["carrier"])
                            scores[model_name] = {k:v[k] for k in ("text","exact","complete_suffix")}
                        if mode == "full":
                            assert scores["small.en"]["text"] == capture["lanes"]["selected"]["asr"]["text"]
                        row[mode] = scores
                rows.append(row)
                if len(rows)%10 == 0:
                    print(len(rows), group, flush=True)
    for r in previous["records"]:
        group = r["group"]
        if group not in ("kokoro","piper"):
            continue
        s = r["source"]
        key = [s["voice"],s["carrier"]] if group=="kokoro" else [r["index"]]
        rows.append(dict(group=group, detector="fixed", index=r["index"], source_sha256=r["mic_sha256"],
                         key=key, command=s["carrier"], opened=r["opened"], timely=s["cue_at"]<=r["opened"]<s["command_at"],
                         full=r["stages"]["selected_full"]["texts"], open_minus_100ms=r["stages"]["selected_open_minus_100ms"]["texts"]))
    totals = defaultdict(lambda:defaultdict(Counter))
    for r in rows:
        for mode in ("full","open_minus_100ms"):
            for model in ("small.en","large-v3-turbo"):
                ctr = totals[r["group"]+"."+r["detector"]][mode+"."+model]
                ctr["trials"] += 1
                ctr["timely"] += r["timely"]
                if r["opened"] is not None:
                    score = r[mode][model]
                    ctr.update(dict(opened=1,exact=score["exact"],complete_suffix=score["complete_suffix"],
                                    timely_complete=r["timely"] and score["complete_suffix"]))
    assert len(rows)==114 and sum(r["opened"] is not None for r in rows if r["detector"]=="willow") == 80
    ch.save(ch.RESULTS/"4090-c161-willow-first-command-readers.json",dict(records=rows,summary=totals,
        sources=sources, fixed_report_sha256=ch.sha(report), script_sha256=ch.sha(Path(__file__)),
        policy="Official DEFAULT Hey Willow returns, existing handoff fixed-v5 clocks, same two unprompted GPU "
        "readers and same two decoded Opus time windows. Unopened cases count against end-to-end completeness. "
        "Piper uses four Willow cue voices on each of the same eighteen command clips and one handoff Nyx "
        "cue on each. Kokoro uses twelve name-specific whole-sentence syntheses per route at matched voice, "
        "text and speed. The two address syntheses can have different command waveforms."))


if __name__ == "__main__":
    main()
