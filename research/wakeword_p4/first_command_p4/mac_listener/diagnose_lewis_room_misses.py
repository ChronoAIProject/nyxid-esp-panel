"""Separate direct pickup, room response, background history and playback level at fixed command clocks."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import check_lewis_cue_rooms as parent

ch, rooms, gap = parent.ch, parent.rooms, parent.gap
OUT = ch.RESULTS / "4090-c161-lewis-room-factors.json"


def room_inputs():
    bank = rooms.source_bank()
    source = ch.read(rooms.PLAN)
    original = ch.read(parent.OUT)["records"]
    for prefix, kind in (("h012_", "air_conditioner"), ("h004_", "music")):
        h, room, rh = rooms.read_room(prefix)
        noise, _, nh = rooms.read_noise(kind)
        for s in source["records"]:
            if (s["room"], s["target_id"]) != (room, "silence"):
                continue
            old, = [r for r in original if (r["room"], r["command_voice"], r["command"]) ==
                    (room, s["voice"], s["carrier"])]
            assert old["rir_sha256"] == rh and old["noise_sha256"] == nh
            bg = noise[(s["noise_offset"] + np.arange(s["samples"])) % len(noise)] * s["noise_gain"]
            command = fftconvolve(bank[s["voice"], s["carrier"]].astype(float), h)
            cmd = np.zeros_like(bg)
            end = s["end"]
            cmd[end:end + len(command)] = command
            assert ch.digest(quantize((bg + cmd) * s["gain"])) == s["pcm_sha256"]
            yield s, old, kind, h, bg, cmd


def quantize(x):
    return np.rint(x).clip(-32768, 32767).astype("<i2")


def measure(native, pcm, start, end):
    codes, times, _, _ = rooms.run_stream(native, pcm)
    clock = np.asarray(times)
    codes = np.asarray(codes[0], np.uint8)
    events = rooms.times(rooms.decode(codes, clock))
    sums = np.convolve(codes.astype(int), np.ones(5, int), "valid")
    delivered = (clock[4:] + 255) // 256 * 256
    before = (start <= delivered) & (delivered < end)
    at = int(np.argmax(np.where(before, sums, -1)))
    assert before[at]
    return dict(events=events, timely=any(start <= ((t+255)//256)*256 < end for t in events),
                peak5=int(sums[at]), peak_at=int(clock[at+4]), codes_sha256=ch.digest(codes))


def main():
    cue, rate = sf.read(gap.CUE, dtype="int16")
    assert rate == 16000 and ch.sha(gap.CUE) == ch.read(gap.OUT)["cue_sha256"]
    native = rooms.c161_binding()
    records, summary = [], defaultdict(lambda: defaultdict(Counter))
    for s, old, kind, h, bg, cmd in room_inputs():
        end = s["end"]
        start = end - len(cue)
        spoken = slice(start, end-gap.GAP)
        out = dict(room=s["room"], cue_at=start, command_at=end, voice=s["voice"], command=s["carrier"], tests={})
        for channel, stimulus in (("direct", cue), ("room", fftconvolve(cue.astype(float), h))):
            sig = np.zeros_like(bg)
            sig[start:start+len(stimulus)] = stimulus
            power = np.mean(sig[spoken]**2)
            out[channel+"_snr_db"] = float(10*np.log10(power/np.mean(bg[spoken]**2)))
            for condition in ("quiet", "background", "history_only", "from_cue"):
                noise = bg.copy()
                if condition == "quiet":
                    noise.fill(0)
                elif condition == "history_only":
                    noise[start:] = 0
                elif condition == "from_cue":
                    noise[:start] = 0
                gains = (-6, 0, 6, 12) if condition == "background" else (0,)
                for db in gains:
                    name = f"{channel}.{condition}.{db}"
                    mix = (cmd + noise + sig * 10**(db/20)) * s["gain"]
                    pcm = quantize(mix)
                    result = measure(native, pcm, start, end)
                    result.update(pcm_sha256=ch.digest(pcm),
                                  clipped=int(np.count_nonzero((mix < -32768.5) | (mix >= 32767.5))))
                    if name == "room.background.0":
                        assert (result["pcm_sha256"], result["events"]) == (
                            old["routes"]["lewis_nyx"]["pcm_sha256"], old["routes"]["lewis_nyx"]["c161"])
                    out["tests"][name] = result
                    summary[kind][name].update(trials=1, timely=result["timely"], clipped=result["clipped"])
        records.append(out)
    assert len(records) == 16
    ch.save(OUT, dict(records=records, summary=summary, sources={p.name: ch.sha(p) for p in (
                  gap.CUE, rooms.PLAN, parent.OUT, Path(__file__))}, model_sha256=ch.sha(rooms.MODEL),
                  policy="C161 fixed 235/256 threshold and 5-step smoothing. Diagonal room+background reproduces "
                  "all archived scene hashes and events. Change only the cue acoustic path, external speaker "
                  "gain, or recorded noise before and/or after cue start. Keep the command and room timing fixed. "
                  "Abrupt history boundaries are diagnostic. Peak sum uses five native UINT8 scores at read-block "
                  "delivery times strictly before the command. All 16 scene cells already used."))


if __name__ == "__main__":
    main()
