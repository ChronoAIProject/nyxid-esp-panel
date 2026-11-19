"""Replay the two published timed-marker packs against their P4 simulator UART clocks."""
from array import array
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import wave
from zipfile import ZipFile

import read_mr_capture as reader

RESULTS = reader.APP.parents[1] / "results"
PACKS = {
    "c161-panel-level-reference-mac.zip": "af3b27ed470a154a520ef3f004890aa195a4ae603239307b0de98fa53b3731ad",
    "c161-timed-panel-mac-ab.zip": "40a6943c2f2261b9b7beba1f027aa86ae4cb621cb9f62282c51e0153cc885da5",
}
MIDDLE = "3074872c617a93b36c327690f86605f23d11e22f3299925e22af5eda5a1cabdb"


def mono(archive, name):
    with wave.open(io.BytesIO(archive.read(name))) as wav:
        assert wav.getparams()[:3] == (1, 2, 16000)
        pcm = array("h")
        pcm.frombytes(wav.readframes(wav.getnframes()))
        assert len(pcm) == wav.getnframes()
        return pcm


def run_case(archive, case, known_middle, tmp):
    mic = mono(archive, case.get("mic") or case["m"])
    ref = mono(archive, case.get("ref") or case["r"])
    assert len(mic) == len(ref)
    lines = archive.read(case["uart"]).decode().splitlines()
    assert lines[-1] == "LISTENER PASS"
    timed = [int(line.split()[3]) for line in lines if line.startswith("HITS ")]
    timed = [sample for sample in timed if sample > 0]
    opened = [int(line.split()[1]) for line in lines if line.startswith("OPEN ")]
    assert opened == case["opened"]
    tx = case.get("tx_capture_sample", 0)
    layouts = {}
    for channels in (2, 4):
        samples = array("h", [0]) * (len(mic) * channels)
        samples[0::channels], samples[1::channels] = mic, ref
        if channels == 4:
            samples[2::channels], samples[3::channels] = ref[::-1], mic[::-1]
        wav_path = tmp / "capture.wav"
        with wave.open(str(wav_path), "wb") as out:
            out.setparams((channels, 2, 16000, 0, "NONE", "not compressed"))
            out.writeframes(samples.tobytes())
        report = reader.analyze(wav_path, middle_receiver=True, panel_tx_samples=[tx] if tx else None)
        assert report["events"]["mic_timed_three"] == timed, (case["uart"], report["events"], timed)
        if known_middle is not None:
            assert known_middle["transmit_event"] == tx
            if tx:
                near, = report["panel_playout"]["mic_panel_middle_near_tx"]
                assert near == dict(tx_sample=tx, mic_samples=known_middle["c_hits"])
            else:
                assert known_middle["c_hits"] == []
        layouts[channels] = report
    for field in ("events", "levels", "reference_activity", "panel_playout"):
        assert layouts[2].get(field) == layouts[4].get(field)
    events = layouts[2]["events"]
    row = dict(uart=case["uart"], mic_three=events["mic_three"], mic_timed_three=timed,
               uart_open=opened, capture_pcm_sha256={r["layout"]: r["pcm_sha256"] for r in layouts.values()})
    if known_middle is not None:
        row.update(tx_sample=tx, mic_panel_middle=events["mic_panel_middle"],
                   mic_near_tx=known_middle["c_hits"], mic_panel_pair=events["mic_panel"])
    return row


def main():
    rows = []
    with TemporaryDirectory() as folder:
        for name, digest in PACKS.items():
            path = RESULTS / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            with ZipFile(path) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                reports = {}
                for source, sha in manifest["sources"].items():
                    if manifest["firmware"].get(source) == MIDDLE:
                        data = (RESULTS / source).read_bytes()
                        assert hashlib.sha256(data).hexdigest() == sha
                        reports[source] = {r["index"]: r for r in json.loads(data)["records"]}
                if reports:
                    for header in ("c161_panel_middle_marker.h", "c161_timed_final_marker.h",
                                   "c161_coded_cue_coefficients_v5.h"):
                        assert (RESULTS.parent / "portable" / header).read_bytes() == archive.read("receiver/" + header)
                for case in manifest["cases"]:
                    source = case.get("firmware")
                    known = reports[source][case["index"]] if source in reports else None
                    row = run_case(archive, case, known, Path(folder))
                    rows.append(dict(pack=name, **row))
    report = dict(packs=PACKS, middle_receiver_sha256=MIDDLE, sample_rate=16000, layouts_per_case=2,
                  uart_cases=len(rows), cases_with_uart_cue=sum(bool(r["mic_timed_three"]) for r in rows),
                  legacy_clock_disagreements=sum(r["mic_three"] != r["mic_timed_three"] for r in rows),
                  legacy_misses=sum(not r["mic_three"] and bool(r["mic_timed_three"]) for r in rows),
                  records=rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
