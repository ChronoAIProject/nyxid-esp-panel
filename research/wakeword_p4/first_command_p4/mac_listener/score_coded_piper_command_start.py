"""Compare the full Opus capture with a command crop timed from the synthesized cue."""
from collections import Counter, defaultdict
import io
import json
from pathlib import Path
from zipfile import ZipFile

import soundfile as sf

import fixed_cue_dialogue_p4 as fixed
import paired_piper_coded as study
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

CODE_SPEC = RESULTS / "4090-c161-single-cue-advance-source.json"
GUARD = 320  # One Opus packet before the expected end of the synthetic call.


def score_capture(board, source, uart, reader, decoder, tail):
    i = board["index"]
    text = uart.read(f"{i:02}-opus.log").decode().replace("\r\n", "\n")
    assert "UPLINK PASS" in text
    # The first confirmed code is the only source of command timing. The labelled
    # command clock is used below only to measure the error of that estimate.
    codes = [t for t in board["allowed_v5"] if t <= board["delivered"]]
    estimated_end = codes[0] + tail if codes else None
    lanes = {}
    for lane, (name, delay) in enumerate((("selected", 1440), ("raw", 0))):
        old = board["uplink"][name]
        packets = study.uplink.packets(text, i, board["count"], lane)
        assert len(packets) == old["packets"]
        pcm = study.uplink.decode(decoder, packets)
        assert study.digest(pcm) == old["decoded_sha256"]
        cut = max(0, estimated_end - GUARD + delay - board["first"]) if codes else 0
        cropped = study.uplink.recognition(reader, pcm[cut:], source["carrier"]) if cut else old["full"]
        lanes[name] = dict(cut=cut, decoded_sha256=old["decoded_sha256"], packets=len(packets),
                           full=old["full"], from_wake=old["from_wake"], from_code=cropped)
    error = estimated_end - source["command_at"] if codes else None
    return dict(index=i, estimated_end=estimated_end, error_samples=error, lanes=lanes)


def main():
    spec = read(CODE_SPEC)
    tail = spec["trim_end"] - (spec["code_positions"][-1] + spec["code_length"])
    assert tail == study.advance.SHIFT
    reader, identity = fixed.host.piper.load_reader()
    decoder = study.uplink.decoder_api()
    reports = {}
    summary = defaultdict(lambda: defaultdict(Counter))
    trials = []
    groups = (("fixed", fixed.STEM, fixed.STEM), ("coded", study.STEM, study.STEM),
              ("fixed_rooms", "4090-c161-single-cue-advance-p4", "4090-c161-single-cue-advance"),
              ("coded_rooms", study.advance.STEM, study.advance.STEM))
    for group, report_stem, archive_stem in groups:
        path = RESULTS / (report_stem + ".json")
        archive = RESULTS / (archive_stem + "-uart.zip")
        report = read(path)
        assert sha(archive) == report["uart_sha256"] and identity == report.get("asr", report.get("asr_files"))
        reports[group] = dict(report_sha256=sha(path), uart_sha256=sha(archive))
        with ZipFile(archive) as uart:
            assert uart.testzip() is None
            for row in report["records"]:
                if group == "coded_rooms" and row["mode"] != "advance":
                    continue
                board = row["board"] if group == "fixed" else row["external"]
                source = row.get("source", row)
                if "delivered" not in board:
                    trials.append(dict(group=group, index=source["index"], missed=True))
                    continue
                result = score_capture(board, source, uart, reader, decoder, tail)
                trials.append(dict(group=group, voice=source["voice"], **result))
                for lane, data in result["lanes"].items():
                    for method in ("full", "from_wake", "from_code"):
                        stats = summary[group][lane + "." + method]
                        stats.update(dict(trials=1, exact=data[method]["exact"], complete=data[method]["complete_suffix"]))
    cues = {}
    ordinary_pack = RESULTS / (study.matched.STEM + "-cues.zip")
    with ZipFile(ordinary_pack) as ordinary, ZipFile(study.CUES) as coded:
        manifest = json.loads(coded.read("manifest.json"))
        assert sha(study.CUES) == read(RESULTS / (study.STEM + ".json"))["cue_archive_sha256"]
        for row in manifest["records"]:
            voice = row["voice"]
            for tag, archive, name in (("coded", coded, row["file"]), ("ordinary", ordinary, voice + "-ordinary.wav")):
                pcm, rate = sf.read(io.BytesIO(archive.read(name)), dtype="int16")
                assert rate == 16000
                cues.setdefault(voice, {})[tag] = dict(pcm_sha256=study.digest(pcm),
                    asr=study.uplink.recognition(reader, pcm, "Hey Nyx."))
    save(RESULTS / (study.STEM + "-command-start.json"), dict(records=trials, summary=summary,
         isolated_cues=cues, sources=reports, source_spec_sha256=sha(CODE_SPEC), asr=identity,
         ordinary_pack_sha256=sha(ordinary_pack), tail_samples=tail, guard_samples=GUARD,
         script_sha256=sha(Path(__file__)), policy="At a confirmed v5 open, predict the cue end from the "
         "third-code timestamp and the frozen 80 ms source tail. Leave one 20 ms packet before that end, "
         "accounting for the selected lane's known 90 ms delay at zero reference. Keep the original "
         "full capture on a C161-only open. Report unprompted Whisper including both exact utterance "
         "and suffix scores. Use dry direct recordings and the previous kitchen and living-room captures."))


if __name__ == "__main__":
    main()
