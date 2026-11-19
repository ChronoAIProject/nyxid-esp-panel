"""Locate C161 misses on ordinary connected synthesis, including a fixed silent interval at the sender."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

import check_marked_connected_panel as mixed

ch, tx, study = mixed.ch, mixed.tx, mixed.study
OUT = ch.RESULTS / "4090-c161-unmarked-connected-scores.json"
GAP = 4800


def insert_gap(pcm, at):
    assert not np.any(pcm[-GAP:])
    return np.r_[pcm[:at], np.zeros(GAP, dtype="<i2"), pcm[at:-GAP]]


def score(native, pcm, start, at, planned_command):
    rooms = tx.panel.grid.room.rooms
    codes, clock, _, _ = rooms.run_stream(native, pcm)
    c = np.asarray(codes[0], np.uint8)
    t = np.asarray(clock)
    events = rooms.times(rooms.decode(c, t))
    sums = np.convolve(c.astype(int), np.ones(5, int), "valid")
    delivered = (t[4:] + 255)//256*256
    during = (delivered >= start) & (delivered < at)
    timely = [int((e+255)//256*256) for e in events if start <= (e+255)//256*256 < planned_command]
    late = [int((e+255)//256*256) for e in events if (e+255)//256*256 >= planned_command]
    i = int(np.argmax(np.where(during, sums, -1)))
    assert during[i]
    return dict(mic_sha256=ch.digest(pcm), events=events, timely_model_blocks=timely, late_model_blocks=late,
                pre_boundary_peak5=int(sums[i]), pre_boundary_peak_block=int(delivered[i]),
                max5_until_command=int(sums[(delivered >=start) & (delivered < planned_command)].max())), c, t


def main():
    history = ch.read(study.OUT)
    groups = study.sources()
    previous = {(r["source"]["index"], r["source"]["external_db"], r["source"]["view"]):r
                for r in history["records"]}
    quiet, sr = sf.read(mixed.panel.demo.WAVE, dtype="int16")
    assert sr == 16000 and ch.sha(mixed.panel.demo.WAVE) == history["sources"][mixed.panel.demo.WAVE.name]
    v5, _ = tx.study.panel_wave()
    bed, _, bed_sha = tx.study.grid.room.rooms.read_noise("music")
    native = tx.panel.grid.room.rooms.c161_binding()
    rows, stats, arrays = [], defaultdict(Counter), {}

    def record(meta, pcm, lane, gap, old=None):
        start, at = meta["start"], meta["command_at"]
        result, c, t = score(native, pcm, start, at, at+gap)
        archive_key = f"input{len(rows):04}"
        arrays[archive_key] = c
        arrays[archive_key+"_clock"] = t
        if old is not None and not gap:
            assert result["mic_sha256"] == old["result"]["pcm_sha256"] and result["events"] == old["result"]["c161"]
        row = dict(source=meta, lane=lane, gap=gap, scores_key=archive_key, **result)
        if old is not None and not gap:
            row.update(allowed_model=old["result"]["allowed_model"], opened=old["result"]["opened"])
        rows.append(row)
        summary = stats[f'{meta["target"]}.{lane}.{gap}']
        summary.update(trials=1, timely=int(bool(result["timely_model_blocks"])),
                       late_only=int(bool(result["late_model_blocks"]) and not result["timely_model_blocks"]),
                       never=int(not result["events"]))
        return c, t

    for (voice, text), group in groups.items():
        for name in study.connected.NAMES:
            src, wave = group[name]
            at = src["end"]
            gap_wave = insert_gap(wave, at)
            key = src["index"]
            meta = dict(index=key, voice=voice, command=text, target=name, start=src["start"], command_at=at)
            record(meta, wave, "dry", 0)
            record(meta, gap_wave, "dry", GAP)
            for db in study.LEVELS:
                old = previous[key, db, "both"]
                s = old["source"]
                scene = mixed.room_scene(s, group, quiet, v5, bed, bed_sha)
                original, _ = scene(wave)
                padded, _ = scene(gap_wave)
                # FFT convolutions can differ at the level of floating-point roundoff.
                np.testing.assert_array_equal(original[:at], padded[:at])
                c, t = record(s, original, "both", 0, old)
                cg, tg = record(s, padded, "both", GAP)
                np.testing.assert_array_equal(t, tg)
                np.testing.assert_array_equal(c[t<=at], cg[tg<=at])
                if name == "Hey Nyx":
                    (hp, _, _), (hx, _, _) = tx.diag.room(s["geometry"])
                    n = len(original)
                    bg = bed[(s["music_offset"] + np.arange(n)) % len(bed)]
                    ref = fftconvolve(v5.astype(float), hp)[:len(v5)]
                    bg = bg * (.35*np.linalg.norm(ref)/np.sqrt(len(ref))/np.sqrt(np.mean(bg**2)))
                    for gap, dry in ((0, wave), (GAP, gap_wave)):
                        external = fftconvolve(dry.astype(float), hx)
                        scaled = 10**(db/20)
                        direct = scaled*np.pad(dry.astype(float), (0, n-len(dry)))
                        wet = scaled*np.pad(external, (0, n-len(external)))
                        for lane, signal in (("direct_quiet", direct), ("room_quiet", wet)):
                            pcm, _ = mixed.digitize(signal, s["preamp_gain"])
                            record(s, pcm, lane, gap)
                        pcm, _ = mixed.digitize(bg+wet, s["preamp_gain"])
                        if gap:
                            record(s, pcm, "external", gap)
                        else:
                            assert ch.digest(pcm) == previous[key, db, "external"]["result"]["pcm_sha256"]
                    local_only, _ = scene(np.zeros_like(wave))
                    record(s, local_only, "panel_music_only", 0)
    # Recover the exact archived mics for external speech with music and for the panel/command controls.
    for s, pcm in study.scenes():
        if s["target"] in study.connected.NAMES and s["view"] in ("external","no_address"):
            record(s, pcm, s["view"], 0, previous[s["index"],s["external_db"],s["view"]])
    assert len(rows) == 888
    with TemporaryDirectory(dir=OUT.parent) as folder:
        tmp = Path(folder)
        archive = OUT.with_suffix(".npz")
        np.savez_compressed(tmp/archive.name, **arrays)
        with np.load(tmp/archive.name,allow_pickle=False) as z:
            assert set(z.files) == set(arrays)
            for k in z.files:
                np.testing.assert_array_equal(z[k], arrays[k])
        ch.save(tmp/OUT.name, dict(records=rows, summary=stats, gap_samples=GAP, threshold_sum5=1175,
                model=ch.read(mixed.OUT)["model"], score_archive_sha256=ch.sha(tmp/archive.name),
                sources={p.name:ch.sha(p) for p in (study.OUT, study.connected.PLAN, study.connected.PACK,
                mixed.panel.demo.WAVE, Path(mixed.__file__), Path(__file__))},
                policy="Fixed C161 literal whole-sentence Nyx/Nicks, Nick and Willow; no marker. Prior primed "
                       "stream and unchanged five-code decoder, 256-sample delivery. Test 300 ms of silence "
                       "inserted in the external track only at the annotated command start; same panel TX, music, "
                       "gains and source samples before the boundary. Gap trials measure model delivery, before "
                       "panel arbitration. This delays command availability; used development scenes."))
        (tmp/archive.name).replace(archive)
        (tmp/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
