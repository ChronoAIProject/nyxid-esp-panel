"""Compare the archived full capture with the existing OPEN-minus-100ms server view."""
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from faster_whisper import WhisperModel
import compare_selected_room_transcripts as prior

ch, gap = prior.ch, prior.gap
OUT = ch.RESULTS / "4090-c161-room-open-context.json"
READERS = ("small", "turbo")


def cases(fixed, spoken):
    for index, src in enumerate(fixed["records"]):
        route = src["routes"]["fixed_v5"]
        i = route["capture_id"]
        yield "fixed", index, i, route["opened"], route["capture"], f"opus/{i:03}-rooms.log"
    for cap in spoken["captures"]:
        i = cap["id"]
        yield "spoken", cap["scene"], i, cap["opened"], cap, f"opus/{i:03}-select.log"


def main():
    baseline = ch.read(prior.OUT)
    fixed_path, spoken_path = prior.replay.room.parent.OUT, prior.replay.OUT
    fit_path = prior.replay.selection.FIT
    fixed, spoken, fit = (ch.read(p) for p in (fixed_path, spoken_path, fit_path))
    assert spoken["sources"][fit_path.name] == ch.sha(fit_path)
    paths = (fixed_path, spoken_path, fixed_path.with_suffix(".zip"), spoken_path.with_suffix(".zip"))
    assert all(ch.sha(p) == baseline["sources"][p.name] for p in paths)
    small, identity = ch.load_reader()
    info = baseline["turbo"]
    assert identity == baseline["small_files"] and all(ch.sha(prior.audit.TURBO/k)==h for k,h in info["files"].items())
    turbo = WhisperModel(str(prior.audit.TURBO), device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
    decoder = gap.uplink.decoder_api()
    rows = {"fixed": {}, "spoken": {}}
    with ZipFile(paths[2]) as fixed_uart, ZipFile(paths[3]) as spoken_uart:
        for group, idx, i, opened, cap, key in cases(fixed, spoken):
            s = fixed["records"][idx]
            assert s["command_at"] == spoken["records"][idx]["command_at"]
            uart = fixed_uart if group == "fixed" else spoken_uart
            log = uart.read(key).decode().replace("\r\n", "\n")
            packets = gap.uplink.packets(log, i, cap["emitted"] - cap["first"], 0)
            pcm = gap.uplink.decode(decoder, packets)
            stored = baseline["fixed_v5" if group == "fixed" else "natural"][str(i)]
            assert ch.digest(pcm) == stored["decoded_sha256"] == cap["lanes"]["selected"]["decoded_sha256"]
            assert len(packets) == cap["lanes"]["selected"]["packets"] and s["command"] == stored["command"]
            assert cap["first"] == max(0, opened//160*160 - 32000)
            cut = max(0, opened - 1600 - cap["first"])
            cropped = pcm[cut:]
            row = dict(id=i, scene=idx, command=s["command"], room=s["room"], command_at=s["command_at"],
                       opened=opened, first=cap["first"], packets=len(packets), cropped_start=cut,
                       cropped_sha256=ch.digest(cropped), full_sha256=stored["decoded_sha256"], full={}, cropped={})
            for name, reader in zip(READERS, (small, turbo)):
                score = gap.uplink.recognition(reader, cropped, s["command"])
                row["cropped"][name] = {k:score[k] for k in ("text","complete_suffix","exact")}
                row["full"][name] = {k:stored[name][k] for k in row["cropped"][name]}
            rows[group][i] = row
    stats = defaultdict(lambda: defaultdict(Counter))
    def count(src, key, row):
        room = src["room"]
        counts = stats[room][key]
        counts["trials"] += 1
        if row is None:
            return
        timely = row["opened"] < row["command_at"]
        counts.update(opens=1, timely=int(timely))
        for model in READERS:
            for view in ("full","cropped"):
                result = row[view][model]
                for metric in ("complete_suffix", "exact"):
                    counts[f"{model}_{view}_{metric}"] += result[metric]
                counts[f"{model}_{view}_timely_complete"] += timely and result["complete_suffix"]
            before, after = [row[v][model]["complete_suffix"] for v in ("full", "cropped")]
            counts[f"{model}_gained"] += after and not before
            counts[f"{model}_lost"] += before and not after
    for i, (a,b) in enumerate(zip(fixed["records"], spoken["records"], strict=True)):
        assert a["room"]==b["room"] and a["silence_sha256"]==b["silence_sha256"]
        count(a, "fixed_v5", rows["fixed"][a["routes"]["fixed_v5"]["capture_id"]])
        for choice in fit["choices"].values():
            for mode in ("default","selected"):
                cue = b["cues"][str(choice[mode]["target"])]
                for name, r in cue["routes"].items():
                    row = rows["spoken"][r["capture"]] if r["hits"] else None
                    if row:
                        assert row["scene"] == i and row["opened"] == r["hits"][0]
                    count(a, mode+"_"+name, row)
    assert (len(rows["fixed"]), len(rows["spoken"])) == (16, 70)
    ch.save(OUT, dict(records=rows, summary=stats, source_sha256={p.name:ch.sha(p) for p in (*paths, prior.OUT, fit_path, Path(__file__))},
        small=identity, turbo=info, policy="Reuse the existing OPEN-minus-100ms policy. Decode complete archived "
        "selected-lane P4 Opus packets first. Discard samples before that clock only in the ASR input. "
        "No command labels, optimal offsets, or ASR outputs enter boundary selection. Count misses among all "
        "target trials. Same existing rooms, recordings and command voices; repeated pairs share recordings."))


if __name__ == "__main__":
    main()
