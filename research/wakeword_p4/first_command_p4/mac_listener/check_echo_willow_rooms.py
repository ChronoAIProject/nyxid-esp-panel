"""Compare the frozen Nyx routes and official Willow on identical P4 echo-cancelled room recordings."""
from pathlib import Path
import sys

ROOT = Path("research/wakeword_p4")
OUT = ROOT / "results/4090-c161-echo-willow-rooms.json"
PLAN = OUT.with_name(OUT.stem + "-plan.json")
WORK = Path(".local/wakeword/c161-echo-willow-rooms")
RUNNER = Path(__file__).resolve().relative_to(Path.cwd())


def prepare():
    import struct
    from tempfile import TemporaryDirectory
    import numpy as np
    import soundfile as sf
    import handoff_willow as raw
    import handoff_barge_in as rooms
    import replay_c161_speex_p4 as speex

    ch = rooms.fixed.ch
    old = ch.read(raw.PLAN)
    results = ch.read(raw.BASE.with_suffix(".json"))
    assert results["plan_sha256"] == ch.sha(raw.PLAN)
    elf = speex.OUT / "build-float/nyx_speex_echo.elf"
    pin = ch.read(ch.RESULTS / (speex.STEM + "-float.json"))
    assert ch.sha(elf) == pin["elf_sha256"] and ch.sha(speex.QEMU) == pin["qemu_sha256"]
    assert ch.sha(elf.with_suffix(".bin")) == pin["app_sha256"]
    files = {path: old["files"][path] for path in old["runtime"].values()}
    sources = [RUNNER, raw.PLAN, raw.BASE.with_suffix(".json"), ROOT/"qemu_hey_reference_v2/run_development.py",
               ROOT/"qemu_hey_reference_v2/run_willow_mac_matched_prime_v1.py", elf, elf.with_suffix(".bin")]
    files.update({str(p): ch.sha(p) for p in sources})
    WORK.mkdir(parents=True, exist_ok=True)
    records = []
    with TemporaryDirectory(dir=WORK) as tmp:
        temp = Path(tmp)
        for j, (meta, _, player, _) in enumerate(rooms.scenes()):
            noise = np.random.default_rng(18221).standard_normal(len(player))
            ref = rooms.bench.channel(player, "noisy_line", noise)
            for k in range(4):
                i = 4*j+k
                src, original = old["records"][i], results["records"][i]
                assert src["room"] == meta["room"] and src["carrier"] == meta["carrier"]
                mic, rate = sf.read(src["path"], dtype="int16")
                assert rate == 16000 and len(mic) == len(ref) <= 160000
                assert ch.digest(mic) == src["pcm_sha256"] == original["mic_sha256"]
                assert ch.digest(ref) == src["reference_sha256"] and src["target_id"] == original["role"]
                # The existing firmware takes exactly ten seconds. Only zero padding follows the source.
                pad = 160000-len(mic)
                payload = struct.pack("<II", 0x35434541, 160000)
                payload += np.pad(mic, (0,pad)).tobytes() + np.pad(ref, (0,pad)).tobytes()
                log = temp/"speex.log"
                speex.boot(elf.parent, log, temp/"flash.bin", b"SPEEX PASS", (2<<20, payload))
                pcm, _, _ = speex.parse(log, 160000)
                pcm = pcm[:len(mic)]
                wav = WORK/f"{i:02}.wav"
                sf.write(wav, pcm, rate, subtype="PCM_16")
                files[str(wav)] = ch.sha(wav)
                records.append(dict(index=i, target_id=src["target_id"], room=src["room"], voice=src["voice"],
                                    carrier=src["carrier"], start=src["start"], end=src["end"], samples=len(mic),
                                    path=str(wav), sha256=files[str(wav)], pcm_sha256=ch.digest(pcm),
                                    mic_sha256=ch.digest(mic), ref_sha256=ch.digest(ref)))
                print("prepared", i, flush=True)
    assert len(records) == 64
    ch.save(PLAN, dict(records=records, files=files, runtime=old["runtime"], threshold_requests=["DEFAULT"],
                       policy="Same archived 64 room mic/reference pairs. Replay each through the pinned standalone "
                       "P4 SpeexDSP float image; discard all trailing pad samples. Continuous panel playback."))


def score():
    from tempfile import TemporaryDirectory
    from zipfile import ZipFile, ZIP_DEFLATED
    import soundfile as sf
    import check_spoken_playback_echo as spoken
    import handoff_willow as raw
    from score_handoff_willow import willow_events
    ch = spoken.ch
    plan = ch.read(PLAN)
    plan_sha = ch.sha(PLAN)
    raw_plan = ch.read(raw.PLAN)
    raw_report = ch.read(raw.BASE.with_suffix(".json"))
    net = spoken.natural.Detector()
    assert net.identity == ch.read(spoken.OUT)["models"]
    rows = []
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        from replay_c161_pcm_cues_v1 import detect
        v5 = ch.binding(Path(tmp)/"v5.so", coded="v5")
        for i, (src, mic_src, old) in enumerate(zip(plan["records"],raw_plan["records"],raw_report["records"],strict=True)):
            assert src["index"] == i == old["index"]
            willow = willow_events(i, src, plan_sha, logs, runs=WORK/"runs")
            stages = {}
            for name, source, official in (("raw",mic_src,old["willow"]), ("speex",src,willow)):
                pcm, rate = sf.read(source["path"], dtype="int16")
                assert rate == 16000 and ch.digest(pcm) == source["pcm_sha256"]
                score = net.score(pcm)
                code = [r["t"] for r in detect(v5,[pcm])]
                if name == "raw":
                    assert score["c161"] == old["c161"] and code == old["fixed"]
                events = dict(c161=score["c161"], v5=code, natural=score["events"], willow=official)
                # Speex requires a complete 160-sample block. Both use a common 256-sample capture clock.
                delivered = {key:[(t+255)//256*256 for t in ts] for key,ts in events.items()}
                if name == "speex":
                    delivered = {key:[((t+159)//160*160+255)//256*256 for t in ts] for key,ts in events.items()}
                stages[name] = dict(events=events, delivered=delivered, proposals=score["proposals"],
                                    verified=[t for t in score["live_ends"] if score["checks"][t]["pass"]])
            rows.append(dict(index=i, source=src, stages=stages))
    archive.with_suffix(".tmp").replace(archive)
    ch.save(OUT, dict(records=rows, archive_sha256=ch.sha(archive), models=net.identity, plan_sha256=plan_sha,
                   script_sha256=ch.sha(RUNNER), actual_willow_threshold_bits="3f24dd2f",
                   policy="Use continuous 81600-sample priming for both wake families. Same source clock, raw baseline "
                   "Willow returns from the earlier official run and new DEFAULT returns on actual standalone P4 "
                   "Speex output. Report notification, same-window pre-command delivery and off-target events. "
                   "Deliveries model frame/chunk availability, excluding compute and verifier queue costs."))


if __name__ == "__main__":
    action = sys.argv.pop(1)
    if action == "prepare":
        prepare()
    elif action == "case":
        sys.path.insert(0, str((ROOT/"qemu_hey_reference_v2").resolve()))
        import run_willow_mac_matched_prime_v1 as official
        official.PLAN, official.__file__ = PLAN, str(RUNNER)
        official.case()
    elif action == "willow":
        import run_c161_willow_matched_scenes_v1 as official
        official.PLAN, official.OUT, official.__file__ = PLAN, WORK/"runs", str(RUNNER)
        official.willow()
    elif action == "score":
        score()
    else:
        raise SystemExit("prepare | willow | score | case")
