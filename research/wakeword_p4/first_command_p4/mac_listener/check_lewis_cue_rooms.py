"""Compare a single reusable literal cue and v5 on the archived Willow room command clocks."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import check_lewis_cue_gap as gap
import voice_cues_willow as rooms

ch, study, uplink = gap.ch, gap.study, gap.uplink
OUT = ch.RESULTS / "4090-c161-lewis-cue-gap-rooms.json"


def main():
    bank = rooms.source_bank()
    plan = ch.read(rooms.PLAN)
    reference = ch.read(rooms.RESULTS/(rooms.STEM+"-willow.json"))
    assert reference["plan_sha256"] == ch.sha(rooms.PLAN)
    nyx, rate = sf.read(gap.CUE, dtype="int16")
    assert rate == 16000
    nick, = [pcm for src, pcm in study.prefixes() if (
        src["voice"], src["style"], src["role"]) == ("bm_lewis", "connected", "Hey Nick")]
    fixed_path = gap.device.host.fixed.CUE
    v5, rate = sf.read(fixed_path, dtype="int16")
    assert rate == 16000
    sources = {"lewis_nyx": nyx, "lewis_nick": np.pad(nick, (0, gap.GAP)), "fixed_v5": v5}
    reader, asr = ch.load_reader()
    assert asr == reference["asr_files"]
    decoder = uplink.decoder_api()
    fw = uplink.firmware()
    native = gap.device.host.fixed.ch.c161_binding()
    stats, records = defaultdict(Counter), []
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory() as folder, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        work = Path(folder)
        uplink.WORK = work
        detector = gap.device.host.fixed.ch.binding(work/"v5.so", coded="v5")
        for rir, noise_name in (("h012_", "air_conditioner"), ("h004_", "music")):
            h, room, rir_hash = rooms.read_room(rir)
            noise, _, noise_hash = rooms.read_noise(noise_name)
            for s in plan["records"]:
                if s["room"] != room or s["target_id"] != "silence":
                    continue
                end = s["end"]
                command = s["carrier"]
                cmd = fftconvolve(bank[s["voice"], command].astype(float), h)
                common = noise[(s["noise_offset"]+np.arange(s["samples"])) % len(noise)]*s["noise_gain"]
                common[end:end+len(cmd)] += cmd
                def pcm(signal):
                    return np.rint(s["gain"] * signal).clip(-32768, 32767).astype("<i2")
                assert ch.digest(pcm(common)) == s["pcm_sha256"]
                official, = [r for r in reference["records"] if (r["room"], r["voice"], r["carrier"], r["role"]) ==
                             (room, s["voice"], command, "willow")]
                assert official["command_at"] == end
                w = official["events"]["willow"]
                rows = dict(room=room, command_voice=s["voice"], command=command, command_at=end,
                            noise_sha256=noise_hash, rir_sha256=rir_hash, silence_sha256=s["pcm_sha256"],
                            official_willow_source=official["index"], official_willow_events=w, routes={})
                for name, cue in sources.items():
                    start = end-len(cue)
                    assert start >= 0
                    scene = common.copy()
                    wet = fftconvolve(cue.astype(float), h)
                    scene[start:start+len(wet)] += wet
                    mic = pcm(scene)
                    a = gap.device.host.fixed.ch.routes(detector, native, mic)
                    assert a["pcm_sha256"] == ch.digest(mic)
                    item = dict(c161=a["c161"], fixed=a["fixed"], events=a["merged"], cue_at=start,
                                pcm_sha256=a["pcm_sha256"])
                    hit = next((t for t in a["merged"] if start <= t < len(mic)), None)
                    c = stats[name]
                    c.update(trials=1, opened=hit is not None, before_cue=sum(t < start for t in a["merged"]))
                    if hit is not None:
                        opened = study.voices.timing.block(hit)
                        item["opened"] = opened
                        c["timely"] += opened < end
                        if name != "lewis_nick":
                            i = len(records)*3 + list(sources).index(name)
                            cap = gap.replay.capture(reader, decoder, work, logs, i, "rooms", mic, opened, command)
                            complete = cap["lanes"]["selected"]["asr"]["complete_suffix"]
                            c.update(complete=complete, timely_complete=opened < end and complete)
                            item.update(capture=cap, capture_id=i)
                    rows["routes"][name] = item
                records.append(rows)
            print(room, len(records), flush=True)
        retries = sorted(work.glob("*-attempt*.gz"))
        for path in retries:
            logs.write(path, "retry/"+path.name)
    assert len(records) == 16
    with ZipFile(archive.with_suffix(".tmp")) as logs:
        assert logs.testzip() is None
        for r in records:
            for item in r["routes"].values():
                if "capture" not in item:
                    continue
                i, cap = item["capture_id"], item["capture"]
                text = logs.read(f"opus/{i:03}-rooms.log").decode().replace("\r\n", "\n")
                for lane, name in enumerate(("selected", "raw")):
                    packets = uplink.packets(text, i, cap["emitted"]-cap["first"], lane)
                    assert ch.digest(uplink.decode(decoder, packets)) == cap["lanes"][name]["decoded_sha256"]
    archive.with_suffix(".tmp").replace(archive)
    ch.save(OUT, dict(records=records, summary=stats, firmware=fw, asr=asr, archive_sha256=ch.sha(archive),
                      sources={p.name: ch.sha(p) for p in (study.PACK, gap.CUE, fixed_path, rooms.PLAN,
                               rooms.RESULTS/(rooms.STEM+"-willow.json"), Path(__file__))},
                      retries=[p.name for p in retries], policy="All 16 previously exposed air conditioner and "
                      "music room cells. Rebuild and byte-check the silent-cue control, then insert each candidate "
                      "at its file gain ending at the archived command clock before convolution. Reuse actual "
                      "official Willow events on its own cue, native C161 and v5 time on new cues, next 256-sample "
                      "delivery and P4 Opus. These are simulated rooms and repeat one literal synthetic cue."))


if __name__ == "__main__":
    main()
