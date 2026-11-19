"""Compare the patched board receiver and its live read path to archived sample clocks."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import wave
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results"
PACK = RESULTS / "c161-panel-level-reference-mac.zip"
PACK_SHA = "af3b27ed470a154a520ef3f004890aa195a4ae603239307b0de98fa53b3731ad"
MIDDLE = "3074872c617a93b36c327690f86605f23d11e22f3299925e22af5eda5a1cabdb"
BELLA = RESULTS / "c161-bella-command-playback.zip"
BELLA_SHA = "67b54cc6493d1925914d0628e1785ca693e1b41dc5ee61e18c49f48858fe2e94"


def write_feeders(service, tmp):
    code = (service / "panel_voice.c").read_text()
    start = code.index("static inline int feed_wake_model(")
    stop = code.index("#ifdef NYX_PANEL_RESEARCH_WAKE_CLOCK_FIXTURE", start)
    (tmp / "panel_feeders.inc").write_text(code[start:stop])


def build_replay(board, tmp):
    service = board / "components/panel_services"
    write_feeders(service, tmp)
    subprocess.run(["cc", "-O2", "-std=c11", "-DNYX_PANEL_RESEARCH_FIXED_MARKER=1",
                    "-I", service, "-I", board / "boards/lcd_x8", "-I", tmp,
                    "-c", HERE / "replay.c", "-o", tmp / "replay.o"], check=True)
    exe = tmp / "replay"
    subprocess.run(["c++", "-O2", "-std=c++17", service / "fixed_marker/board_marker.cpp",
                    tmp / "replay.o", "-o", exe], check=True)
    return exe


def mono(archive, name):
    with wave.open(io.BytesIO(archive.read(name))) as wav:
        assert wav.getparams()[:3] == (1, 2, 16000)
        return wav.readframes(wav.getnframes())


def run(exe, mic, ref, hits, read_size=0, tx=0, slots=2):
    assert sys.byteorder == "little" and len(mic) == len(ref)
    count = len(mic) // 2
    mask = bytearray(count)
    for sample in hits:
        assert sample % 480 == 0 and 0 < sample <= count
        mask[sample-1] = 1
    out = subprocess.run([exe, str(count), str(read_size), str(tx), str(slots)],
                         input=mic + ref + mask, stdout=subprocess.PIPE, check=True)
    events = {"M": [], "C": [], "O": []}
    for line in out.stdout.decode().splitlines():
        lane, t = line.split()
        events[lane].append(int(t))
    return events


def expected_open(events, block, count):
    clocks = events["M"] + events["C"]
    return [min(count, (min(clocks) + block-1) // block * block)] if clocks else []


def capture(exe, archive, case, firmware):
    mic, ref = (mono(archive, case[k]) for k in ("m", "r"))
    # The original P4 listener zero-pads its final partial 256-frame I2S read.
    pad = bytes((-(len(mic) // 2) % 256) * 2)
    mic, ref = mic + pad, ref + pad
    count = len(mic) // 2
    uart = [line.split() for line in archive.read(case["uart"]).decode().splitlines()]
    assert uart[-1] == ["LISTENER", "PASS"]
    hits = [int(row[2]) for row in uart if row[0] == "HITS" and int(row[2]) > 0]
    oracle = {k: [int(row[1]) for row in uart if row[0] == tag]
              for k, tag in (("M", "C161_ALLOW"), ("C", "V5_ALLOW"), ("O", "OPEN"))}
    assert oracle["O"] == case["opened"]
    tx = case.get("tx_capture_sample", 0)
    archived_tx = run(exe, mic, ref, hits, tx=tx)
    archived_tx["O"] = expected_open(archived_tx, 256, count)
    if firmware == MIDDLE:
        assert archived_tx == oracle, (case["label"], archived_tx, oracle)
    physical = run(exe, mic, ref, hits)
    for read_size, slots in ((256, 2), (256, 4), (257, 2), (997, 4)):
        actual = run(exe, mic, ref, hits, read_size=read_size, slots=slots)
        expected = dict(physical, O=expected_open(physical, read_size, count))
        assert actual == expected, (case["label"], read_size, slots, actual, expected)
    physical["O"] = expected_open(physical, 256, count)
    return dict(label=case["label"], firmware=firmware, tx_sample=tx, has_nonzero_r=any(ref),
                raw_c161=hits, uart=oracle, with_archived_tx=archived_tx, live_no_tx=physical)


def restored_reference(exe, archive):
    # Geometry 0 has the same panel feed and origin for its simultaneous and
    # panel-only traces. Hold each M and its recorded C161 decision fixed.
    ref = mono(archive, "digital/g0-balanced-real-R-R.wav")
    demo = mono(archive, "speakers/panel-demo.wav")
    assert ref == bytes(32000*2) + demo + bytes(len(ref) - 32000*2 - len(demo))
    records = []
    for name, mfile in (("self-zero-R", "self-zero-R"), ("g0-balanced-real-R", "g0-balanced-real-R")):
        mic = mono(archive, f"digital/{mfile}-M.wav")
        uart = archive.read(f"digital/{name}.txt").decode().splitlines()
        clocks = [int(row.split()[2]) for row in uart if row.startswith("HITS ")]
        hits = [t for t in clocks if t > 0]
        assert hits and len(mic) == len(ref)
        zero = run(exe, mic, bytes(len(ref)), hits, read_size=256, slots=4)
        live = run(exe, mic, ref, hits, read_size=256, slots=4)
        assert zero["M"] and not live["M"]
        if name == "self-zero-R":
            assert not live["O"]
        else:
            assert live["C"] and live["O"][0] < 49600
        records.append(dict(mic=name, raw_c161=hits, zero_reference=zero, restored_reference=live))
    return records


def dry_playback(exe):
    assert hashlib.sha256(BELLA.read_bytes()).hexdigest() == BELLA_SHA
    rows = []
    with ZipFile(BELLA) as archive:
        for case in json.loads(archive.read("manifest.json"))["records"]:
            mic = mono(archive, case["file"])
            assert hashlib.sha256(mic).hexdigest() == case["pcm_sha256"]
            ref = bytes(len(mic))
            t = case["command_at"]
            fixed = run(exe, mic, ref, [], read_size=256)
            erased = run(exe, bytes(t*2) + mic[t*2:], ref, [], read_size=256)
            assert len(fixed["O"]) == 1 and case["cue_file_start"] < fixed["O"][0] < t
            assert not erased["O"]
            rows.append(dict(file=case["file"], command_at=t, cue_end=fixed["C"],
                             read_end=fixed["O"], erased_read_end=erased["O"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patched_board", type=Path)
    args = parser.parse_args()
    assert hashlib.sha256(PACK.read_bytes()).hexdigest() == PACK_SHA
    with TemporaryDirectory() as temp, ZipFile(PACK) as archive:
        exe = build_replay(args.patched_board.resolve(), Path(temp))
        manifest = json.loads(archive.read("manifest.json"))
        cases = [capture(exe, archive, c, manifest["firmware"][c["firmware"]])
                 for c in manifest["cases"]]
        restored = restored_reference(exe, archive)
        dry = dry_playback(exe)
    print(json.dumps(dict(mr_pack=PACK_SHA, middle_firmware=MIDDLE,
                          read_tests=[[256, 2], [256, 4], [257, 2], [997, 4]],
                          c161_input="Archived raw UART HITS at 480-sample boundaries",
                          events="M/C are allowed C161/cue sample ends. O is first read end.",
                          captures=cases, geometry0_reference_pair=restored, dry_pack=BELLA_SHA,
                          dry_conditions="Digital playback with zero R, TX absent, C161 hits disabled",
                          dry_commands=dry), indent=2))


if __name__ == "__main__":
    main()
