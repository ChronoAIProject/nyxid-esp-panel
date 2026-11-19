"""Test a weaker final confirmation after the original two strong, time-aligned markers."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import soundfile as sf

import check_marker_gain_music as study
import check_spoken_wake_eq as spoken
import check_c161_v3_two_anchor_v1 as probe
from replay_c161_pcm_cues_v1 import detect, joined

ch, grid = study.ch, study.grid
OUT = ch.RESULTS / "4090-c161-final-marker-confirmation.json"
ROOT = ch.RESULTS.parent
HEADER = ROOT / "portable/c161_coded_cue_v5.h"
SOURCE = ROOT / "frontend_probe/c161_fixed_playback_probe.cpp"
HEADER_SHA = "9bf9bd02846576c5c103ba507870ebfafd6b7b0657017ce878137cad67eef797"


def candidate_header():
    assert ch.sha(HEADER) == HEADER_SHA
    text = HEADER.read_text()
    old = "double(dot) * dot < 0.09 * kCueBandEnergyV5[pattern] * energy_"
    new = "double(dot) * dot < (pattern == 2 ? 0.04 : 0.09) * kCueBandEnergyV5[pattern] * energy_"
    assert text.count(old) == 1
    return text.replace(old, new).replace("class CodedCueV5", "class FinalMarkerConfirmation")


def binding(folder):
    portable = folder / "portable"
    sources = folder / "frontend_probe"
    portable.mkdir()
    sources.mkdir()
    (portable / "final_marker.h").write_text(candidate_header())
    coeff = HEADER.with_name("c161_coded_cue_coefficients_v5.h")
    (portable / coeff.name).symlink_to(coeff.resolve())
    content = SOURCE.read_text().replace("c161_coded_cue_v5.h", "final_marker.h").replace("nyx::CodedCueV5", "nyx::FinalMarkerConfirmation")
    (sources / SOURCE.name).write_text(content)
    with patch.object(probe, "ROOT", folder):
        return probe.binding(folder / "candidate.so", coded="v5")


def interference(matcher):
    music, conversations = [], []
    for s in ch.read(study.fit.DEV)["records"]:
        assert ch.sha(Path(s["path"])) == s["sha256"]
        x, sr = sf.read(s["path"], dtype="int16")
        assert sr == 16000 and ch.digest(x) == s["pcm_sha256"]
        music.append(dict(track=s["source_path"], samples=len(x), pcm_sha256=s["pcm_sha256"], fixed=detect(matcher, [x])))
    controls = ch.read(spoken.diagnostic.INTERFERENCE)
    for group in ("daily", "bot"):
        for (meta, pcm, c161, _, _), old in zip(spoken.diagnostic.everyday.sources(group), controls["records"][group], strict=True):
            digest = ch.digest(pcm)
            assert digest == old["pcm_sha256"] and c161 == old["independent"]
            spans = [{k: s[k] for k in ("address", "text", "start", "end")} for s in meta["source"]["spans"]]
            conversations.append(dict(group=group, index=old["index"], pcm_sha256=digest, c161=c161,
                                      fixed=detect(matcher, [pcm]), spans=spans))
    return music, conversations


def main():
    history = ch.read(study.OUT)
    dry = study.tracks()
    rows, controls, counts = [], {}, defaultdict(Counter)
    with TemporaryDirectory() as folder:
        work = Path(folder)
        candidate = binding(work)
        full = ch.binding(work / "v5.so", coded="v5")
        pair = ch.binding(work / "pair.so", coded="aligned")
        for group, inputs in (("established", study.established(dry)), ("other_music", study.more_music(dry))):
            for i, (s, waves, control) in enumerate(inputs):
                prior = history["records"][len(rows)]
                base = prior["routes"]["v5"]
                pcm = waves["v5"]
                assert (s, group, i) == (prior["source"], prior["group"], prior["index"])
                assert ch.digest(pcm) == base["pcm_sha256"] and ch.digest(control) == prior["control_sha256"]
                original = detect(full, [pcm])
                assert original == base["fixed"]
                ref = study.prior.level.guard.refs(s, "v5", dry)
                fixed = detect(candidate, [pcm])
                merged = joined(base["c161"], fixed)
                opened = grid.gap.study.voices.timing.block(merged[0]) if merged else None
                timely = opened is not None and s["start"] <= opened < s["command_at"]
                route = dict(c161=base["c161"], fixed=fixed, pcm_sha256=base["pcm_sha256"])
                guarded = grid.drift.guarded(route, full, pair, ref)
                assert guarded["reference_sha256"] == base["reference_sha256"] and guarded["reference_events"] == base["reference_events"]
                key = prior["control_sha256"]
                if key not in controls:
                    ctrl = history["controls"][key]
                    controls[key] = dict(group=group, c161=ctrl["c161"], original_fixed=ctrl["fixed"], fixed=detect(candidate, [control]))
                counts[group].update(trials=1, timely=timely, baseline=base["timely"], gained=timely and not base["timely"],
                                     lost=base["timely"] and not timely, self_open=bool(guarded["opened"]),
                                     early=opened is not None and opened < s["start"])
                rows.append(dict(group=group, index=i, source=s, pcm_sha256=base["pcm_sha256"], control_sha256=key,
                                 previous_opened=base["opened"], opened=opened, fixed=fixed, c161=base["c161"],
                                 reference_sha256=guarded["reference_sha256"], self_opened=guarded["opened"], timely=timely))
            print(group, counts[group], flush=True)
        tracks, conversations = interference(candidate)
        size = candidate.pcm_cue_bytes()
        assert size == full.pcm_cue_bytes()
    assert len(rows) == 1288 and len(controls) == 364 and len(tracks) == 12
    ch.save(OUT, dict(records=rows, summary=counts, controls=controls, music=tracks, conversations=conversations,
                      state_bytes=size, original_wav_sha256=ch.sha(grid.fit.fixed.CUE), c161_sha256=history["model_sha256"],
                      sources={p.name:ch.sha(p) for p in (HEADER, HEADER.with_name("c161_coded_cue_coefficients_v5.h"),
                              SOURCE, study.OUT, study.fit.DEV, spoken.diagnostic.INTERFERENCE, Path(__file__))},
                      policy="One microphone matcher change on original v5 PCM. Require absolute correlation 0.30 "
                      "on the first two and 0.20 on the third with the original polarity, timing and cooldown. "
                      "Use the original reference detector, original frozen C161 per-input notifications and original "
                      "two-second pre-roll. Previously consumed rooms and tracks, no threshold sweep or new exposure."))


if __name__ == "__main__":
    main()
