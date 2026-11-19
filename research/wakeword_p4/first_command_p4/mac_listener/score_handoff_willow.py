"""Check official return clocks and the dual-channel Nyx listener on paired panel speech."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf
import coded_close_join as joined
import handoff_barge_in as handoff
from handoff_willow import BASE, PLAN, WORK


def willow_events(index, record, plan_digest, archive, runs=WORK / "runs"):
    folder = runs / f"{index:02}"
    report = joined.read(folder / "report.json")
    assert report["model"] == "wn9_heywillow_tts" and report["stage"] == 104
    assert report["execution_sha256"] == plan_digest and report["record"] == record
    assert report["actual_float32_bits"] == report["post_prime_getter_bits"] == "3f24dd2f"
    assert report["source_pcm_sha256"] == record["pcm_sha256"] and report["total_prime_samples"] == 81600
    assert not report["additional_priming_events"] and all(v <= 0 for _, v in report["prime_returns"])
    for name, field in (("raw-returns.bin", "raw_returns_sha256"), ("serial.log", "serial_sha256")):
        assert joined.sha(folder/name) == report[field]
        archive.write(folder/name, f"{index:02}/willow-{name}")
    archive.write(folder/"report.json", f"{index:02}/willow-report.json")
    raw = np.fromfile(folder/"raw-returns.bin", np.uint8)
    ends = np.minimum((np.flatnonzero(raw == 2) + 1) * 512, report["input_samples"])
    events = (ends - report["additional_prime_samples"]).tolist()
    assert events == [e["relative_sample"] for e in report["real_events"]]
    return events


def main():
    plan = joined.read(PLAN)
    plan_digest = joined.sha(PLAN)
    history = joined.read(handoff.BASE.with_suffix(".json"))
    baseline = joined.read(joined.RESULTS / f"{joined.STEM}.json")
    assert baseline["plan_sha256"] == joined.sha(joined.PLAN)
    quiet = {(r["room"], r["voice"], r["carrier"], r["role"]): r for r in baseline["records"]}
    records = []
    zipped = BASE.with_suffix(".zip")
    tmp_zip = zipped.with_suffix(".tmp")
    with TemporaryDirectory() as tmpdir, ZipFile(tmp_zip, "w", ZIP_DEFLATED) as archive:
        tmp = Path(tmpdir)
        full = joined.binding(tmp/"full.so", coded="v5")
        pair = joined.binding(tmp/"pair.so", coded="aligned")
        for scene_index, ((source, _, player, _), old) in enumerate(zip(handoff.scenes(), history["records"], strict=True)):
            noise = np.random.default_rng(18221).standard_normal(len(player))
            ref = handoff.bench.channel(player, "noisy_line", noise)
            for j, role in enumerate(("fixed", "ordinary", "willow", "silence")):
                i = 4*scene_index + j
                record = plan["records"][i]
                key = tuple(source[k] for k in ("room", "voice", "carrier"))
                assert key == tuple(record[k] for k in ("room", "voice", "carrier"))
                wav = Path(record["path"])
                mic, rate = sf.read(wav, dtype="int16")
                assert rate == 16000 and joined.sha(wav) == record["sha256"]
                assert joined.digest(mic) == record["pcm_sha256"] and joined.digest(ref) == record["reference_sha256"]
                assert record["target_id"] == role
                willow = willow_events(i, record, plan_digest, archive)
                silence_reference = None
                if role == "fixed":
                    nyx = old["continue"]
                    assert nyx["mic_sha256"] == record["pcm_sha256"] and nyx["reference_sha256"] == record["reference_sha256"]
                else:
                    prior = quiet[(*key, role)]
                    assert prior["pcm_sha256"] == record["before_echo_sha256"]
                    silence_reference = dict(c161=prior["c161"], willow=prior["arms"]["willow"]["events"])
                    nyx = handoff.probe.score(tmp, source, mic, ref, full, i, build=handoff.test.BUILD, prefix=pair,
                                qualify=handoff.probe.qualify, wait_for_third=True, frame_activity=True, qualify_capture_frame=True)
                    assert joined.sha(tmp/"serial.log") == nyx["uart_sha256"]
                    archive.write(tmp/"serial.log", f"{i:02}/nyx-serial.log")
                records.append(dict(index=i, role=role, room=source["room"], voice=source["voice"],
                        carrier=source["carrier"], cue_at=record["start"], command_at=record["end"],
                        mic_sha256=record["pcm_sha256"], willow=willow, c161=nyx["c161"], fixed=nyx["fixed"],
                        nyx_open=nyx["opened"], nyx_uart_sha256=nyx["uart_sha256"], panel_silent=silence_reference))
                print(i, role, willow, nyx["opened"], flush=True)
    assert len(records) == len(plan["records"])
    tmp_zip.replace(zipped)
    joined.save(BASE.with_suffix(".json"), dict(records=records, plan_sha256=plan_digest,
            previous_sha256=joined.sha(joined.RESULTS / f"{joined.STEM}.json"),
            handoff_sha256=joined.sha(handoff.BASE.with_suffix(".json")), archive_sha256=joined.sha(zipped),
            firmware_sha256=joined.sha(handoff.test.BUILD/"nyx_mac_listener.bin"),
            model_sha256=handoff.listener.C161, code_sha256=joined.sha(Path(__file__)),
            policy="Official classic ESP32 QEMU DEFAULT on raw mono mic, versus C161 detections and the P4 QEMU "
                   "dual-input listener OPEN clock. Source commands start at 57600. Panel continues throughout. "
                   "For ordinary, Willow and silence, historic raw detection clocks without panel playback are paired by dry mic hash."))


if __name__ == "__main__":
    main()
