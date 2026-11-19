"""Distinguish the panel's own demonstration from an external original v5 using marker order."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import check_final_marker_confirmation as host
from code_gain_guard import activity_pulses
from self_guard import expected

ch, grid = host.ch, host.grid
OUT = ch.RESULTS / "4090-c161-panel-marker-order.json"
WAV = ch.RESULTS / "hye-nyx-panel-bca.wav"
PAIR_HEADER = host.HEADER.with_name("c161_coded_cue_aligned_v4.h")
COEFF = host.HEADER.with_name("c161_coded_cue_coefficients_v5.h")
OFFSETS = (0, 4000, 8000, 16000, 40000)
GEOMETRIES = ((1, 3), (7, 9), (18, 20), (24, 26))
# Time until the third marker plus the original full-reference echo guard.
VETO = 1944 + 3488


def panel_wave():
    meta, original, speech, codes = grid.fit.original()
    alternate = speech.copy()
    for pos, code in zip(meta["code_positions"], codes[[1, 2, 0]], strict=True):
        alternate[pos:pos+len(code)] += code
    assert np.abs(alternate).max() <= 32767
    return original, alternate.astype("<i2")


def reference_header():
    text = PAIR_HEADER.read_text()
    for before, after in (
        ("c161_coded_cue_band_coefficients_v4.h", COEFF.name),
        ("class AlignedCodedCueV4", "class PanelMarkerPair"),
        ("kCueBandFir[", "kCueBandFirV5["),
        ("kCueBandPatterns[pattern]", "kCueBandPatternsV5[pattern + 1]"),
        ("kCueBandEnergy[pattern]", "kCueBandEnergyV5[pattern + 1]"),
    ):
        assert text.count(before) == 1
        text = text.replace(before, after)
    return text


def reference_binding(work):
    portable = work / "portable"
    source = work / "frontend_probe"
    portable.mkdir(parents=True)
    source.mkdir()
    (portable / COEFF.name).symlink_to(COEFF.resolve())
    (portable / "panel_marker_pair.h").write_text(reference_header())
    wrapper = host.SOURCE.read_text().replace(PAIR_HEADER.name, "panel_marker_pair.h").replace("nyx::AlignedCodedCueV4", "nyx::PanelMarkerPair")
    (source / host.SOURCE.name).write_text(wrapper)
    with patch.object(host.probe, "ROOT", work):
        return host.probe.binding(work / "panel.so", coded="aligned")


def simultaneous():
    v5, panel = panel_wave()
    bank = grid.room.rooms.source_bank()
    parents = sorted(k for k in bank if k[1].startswith(("Raise ", "Read ")))
    bg, _, noise_hash = grid.room.rooms.read_noise("music")
    local_at = 32000
    for i, (a, b) in enumerate(GEOMETRIES):
        hp, panel_room, ph = grid.room.rooms.read_room(f"h{a:03}_")
        hx, external_room, xh = grid.room.rooms.read_room(f"h{b:03}_")
        voice, command = parents[i]
        dry = np.r_[v5, bank[voice, command]]
        ext = fftconvolve(dry.astype(float), hx)
        panel_waves = {"abc":v5, "bca":panel}
        responses = {n:fftconvolve(w.astype(float), hp) for n,w in panel_waves.items()}
        for offset in OFFSETS:
            at = local_at + offset
            length = at + len(ext) + 4800
            noise = bg[(39413*i + np.arange(length)) % len(bg)]
            noise = noise * (.35*np.linalg.norm(responses["abc"][:len(v5)]) / np.sqrt(len(v5)) / np.sqrt(np.mean(noise**2)))
            for db in (-6, 0, 6):
                external = np.zeros(length)
                external[at:at+len(ext)] = ext * 10**(db/20)
                signals, refs = {}, {}
                for name, local in responses.items():
                    mic = noise.copy()
                    mic[local_at:local_at+len(local)] += local
                    ref = np.zeros(length, dtype="<i2")
                    ref[local_at:local_at+len(v5)] = panel_waves[name]
                    signals[name] = {"panel":mic, "external":noise+external, "both":mic+external}
                    refs[name] = ref
                peak = max(np.abs(x).max() for v in signals.values() for x in v.values())
                gain = min(1., 28000/peak)
                meta = dict(geometry=i, panel_room=panel_room, external_room=external_room,
                            panel_rir_sha256=ph, external_rir_sha256=xh, noise_sha256=noise_hash,
                            offset=offset, external_db=db, local_at=local_at, start=at, command_at=at+len(v5),
                            command=command, voice=voice, command_pcm_sha256=ch.digest(bank[voice,command]), gain=gain)
                for name, roles in signals.items():
                    for role, signal in roles.items():
                        ref = np.zeros_like(refs[name]) if role == "external" else refs[name]
                        yield meta, name, role, grid.room.quantize(signal*gain), ref


def score(mic, ref, detectors, cache):
    strict, weak, prefix, panel, native = detectors
    key = ch.digest(mic)
    if key not in cache:
        route = grid.fit.fixed.ch.routes(weak, native, mic)
        cache[key] = (route["fixed"], route["c161"])
    fixed, c161 = cache[key]
    raw = dict(pcm_sha256=key, fixed=fixed, c161=c161)
    guard = grid.drift.guarded(raw, strict, prefix, ref)
    panel_hits = [r["t"] for r in host.detect(panel, [ref])]
    pulse = activity_pulses(ref)
    pending = sorted(set(guard["prefix_reference"]) - set(guard["qualified_prefix"]))
    def accepted(hits):
        return expected(hits, guard["reference_events"], pulse, guard["qualified_prefix"], pending)
    old_model = accepted(c161)
    accepted_fixed = accepted([r["t"] for r in fixed])
    assert sorted(old_model+accepted_fixed) == guard["allowed"]
    allowed_model = []
    for t in c161:
        for decision in accepted([t]):
            if not any(p <= decision and p + VETO >= t for p in panel_hits):
                allowed_model.append(decision)
    allowed = sorted(allowed_model+accepted_fixed)
    return dict(**raw, reference_sha256=guard["reference_sha256"], original_opened=guard["opened"],
                panel_pairs=panel_hits, allowed=allowed, allowed_fixed=accepted_fixed, allowed_model=allowed_model,
                opened=[(allowed[0]+255)//256*256] if allowed else [])


def main(timed=False):
    from check_timed_final_marker import HEADER, OUT as timed_grid, binding as timed_binding
    make_matcher = timed_binding if timed else host.binding
    grid_report = timed_grid if timed else host.OUT
    output = OUT.with_name(OUT.stem + "-timed.json") if timed else OUT
    v5, panel = panel_wave()
    if not timed:
        sf.write(WAV, panel, 16000, subtype="PCM_16")
    assert ch.digest(sf.read(WAV,dtype="int16")[0]) == ch.digest(panel)
    reader, identity = ch.load_reader()
    readings = host.study.prior.p4.reading({"v5":v5,"panel":panel}, reader)
    singles, pairs, totals = [], [], defaultdict(Counter)
    prior = ch.read(grid_report)["records"]
    with TemporaryDirectory() as folder:
        work = Path(folder)
        detectors = (ch.binding(work/"strict.so",coded="v5"), make_matcher(work),
                     ch.binding(work/"prefix.so",coded="aligned"), reference_binding(work/"ref"), grid.room.rooms.c161_binding())
        cache = {}
        for meta, name, role, mic, ref in simultaneous():
            result = score(mic, ref, detectors, cache)
            pairs.append(dict(source=meta, name=name, role=role, result=result))
            for mode in ("original_opened", "opened"):
                clocks = result[mode]
                totals[f"overlap.{name}.{role}.{mode}"].update(trials=1, opens=bool(clocks),
                    timely=bool(role != "panel" and clocks and meta["start"]<=clocks[0]<meta["command_at"]),
                    before_external=bool(clocks and clocks[0]<meta["start"]))
        cache.clear()
        for i, (s, signals, _) in enumerate(host.study.established({"v5":v5,"bca":panel})):
            orig = prior[i]
            assert s == orig["source"] and ch.digest(signals["v5"]) == orig["pcm_sha256"]
            ref = host.study.prior.level.guard.refs(s, "bca", {"bca":panel})
            result = score(signals["bca"], ref, detectors, cache)
            singles.append(dict(index=i, source=s, result=result))
            totals["single_panel"].update(trials=1, unguarded=bool(result["original_opened"]),
                guarded=bool(result["opened"]), pairs=bool(result["panel_pairs"]))
    assert len(pairs)==360 and len(singles)==672
    ch.save(output, dict(records=pairs, single_panel=singles, summary=totals, readings=readings, reader=identity,
        wav_sha256=ch.sha(WAV), original_wav_sha256=ch.sha(grid.fit.fixed.CUE),
        dry_rms_db_vs_v5=float(20*np.log10(np.linalg.norm(panel.astype(float))/np.linalg.norm(v5.astype(float)))),
        sources={p.name:ch.sha(p) for p in (grid_report, HEADER if timed else host.HEADER, PAIR_HEADER, COEFF, Path(__file__))},
        policy="Only the panel-owned demonstration cyclically permutes the three additive markers. External playback "
        "uses original v5 with the microphone matcher identified by the input grid report. Both reference checks operate on actual PCM. "
        "An electrical B,C pair suppresses only C161 over the next 5432 samples. Compare both orders, both guards, "
        "single sources and overlap on shared gains. Offset-zero source identity is checked with source-only controls. "
        "All 28 single-panel rooms retain the external command, matching the earlier 672 trials."))
    print(dict(totals), readings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timed", action="store_true")
    main(parser.parse_args().timed)
