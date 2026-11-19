"""Compare the previously frozen equal-RMS cue on the clocked single-C receiver."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import check_panel_middle_p4 as middle

ch, tx = middle.ch, middle.tx
WAVE = ch.RESULTS / "hye-nyx-v5-marker-gains-rms.wav"
OUT = ch.RESULTS / "4090-c161-panel-rms-cue-p4.json"


def inputs(baseline, original, candidate):
    scene = baseline["source"]
    _, noise, panel, cmd, room, gain, _ = tx.diag.scene(baseline)
    command, _ = tx.balanced.match_level(original, cmd)
    at = scene["start"]
    mix = noise.copy()
    local_at = scene["local_at"]
    mix[local_at:local_at + len(panel)] += panel
    for label, cue in (("original", original), ("rms", candidate), ("no_cue", np.zeros_like(candidate))):
        pcm = mix.copy()
        external = fftconvolve(np.r_[cue, command].astype(float), room) * gain
        pcm[at:at + len(external)] += external
        yield label, tx.study.grid.room.quantize(pcm * scene["gain"])


def main():
    previous = ch.read(middle.OUT)
    prior = {r["host_index"]: r for r in previous["records"] if r["group"].startswith("balanced.")}
    originals = ch.read(tx.diag.BOARD)
    versions = originals["versions"]
    build = middle.WORK / "build"
    assert ch.sha(build / "nyx_mac_listener.bin") == previous["firmware"]
    assert tx.panel.grid.gap.uplink.firmware() == versions["codec"]
    cue, _ = tx.study.panel_wave()
    candidate, rate = sf.read(WAVE, dtype="int16")
    assert rate == 16000 and candidate.shape == cue.shape
    rms_ratio = np.linalg.norm(candidate.astype(float)) / np.linalg.norm(cue.astype(float))
    assert abs(20 * np.log10(rms_ratio)) < 0.001
    small, identity = ch.load_reader()
    assert identity == versions["small"]
    turbo = tx.diag.reader(versions["turbo"])
    decoder = tx.panel.grid.gap.uplink.decoder_api()
    rows, summaries = [], defaultdict(Counter)
    archive = OUT.with_suffix(".zip")
    scratch = archive.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(scratch, "w", ZIP_DEFLATED) as z:
        work = Path(folder)
        detector = tx.single.timed.binding(work)
        native = tx.panel.grid.room.rooms.c161_binding()
        for baseline in originals["records"]:
            if (baseline["name"], baseline["role"]) != ("bca", "both"):
                continue
            i = baseline["host_index"]
            old = prior[i]
            scene = old["source"]
            assert scene == baseline["source"]
            clock = old["transmit_event"]
            for label, mic in inputs(baseline, cue, candidate):
                if label == "original":
                    assert ch.digest(mic) == old["mic_sha256"]
                    continue
                ref = np.zeros_like(mic)
                assert ch.digest(ref) == old["reference_sha256"]
                raw = tx.panel.grid.fit.fixed.ch.routes(detector, native, mic)
                expected = middle.pair.admit(raw, clock, middle.single.hits(mic, clock), ref)
                case = (i, scene, "bca", "both", mic, ref, expected)
                row = tx.panel.capture(work, z, len(rows), case, small, turbo, decoder,
                                       dict(bca=build), tx_pair=clock)
                row.update(cue=label, transmit_event=clock, baseline_index=old["index"])
                rows.append(row)
                def timely(r):
                    return bool(r["opened"] and scene["start"] <= r["opened"][0] < scene["command_at"])
                stats = summaries[f'{label}.{scene["external_db"]}']
                stats.update(trials=1, opens=int(bool(row["opened"])), timely=int(timely(row)),
                             small=int(row["small"]), turbo=int(row["turbo"]))
                if label == "rms":
                    for key, before, after in (("timely", timely(old), timely(row)),
                                               ("small", old["small"], row["small"]),
                                               ("turbo", old["turbo"], row["turbo"])):
                        stats.update({key + "_before": int(before), key + "_gained": int(after and not before),
                                      key + "_lost": int(before and not after)})
            print(i, scene["external_db"], rows[-2]["opened"], rows[-1]["opened"], flush=True)
    assert len(rows) == 120 and len(prior) == 60
    with ZipFile(scratch) as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist()))
    scratch.replace(archive)
    ch.save(OUT, dict(records=rows, summary=summaries, firmware=previous["firmware"], model=previous["model"],
        small=identity, turbo=versions["turbo"], codec=versions["codec"], rms_delta_db=float(20*np.log10(rms_ratio)),
        archive_sha256=ch.sha(archive), sources={p.name: ch.sha(p) for p in (middle.OUT, tx.diag.BOARD, WAVE, Path(__file__))},
        policy="All 60 existing paired BCA-panel and v5-external scenes. Reproduce every original balanced microphone "
               "hash. Compare the established equal-RMS external cue, then silence the external cue. Preserve the "
               "same balanced external command, music, gains, geometry, timing, panel demo, TX event and zero R. "
               "Count full packet recognition on all trials, including unopened inputs. Reused development exposure."))


if __name__ == "__main__":
    main()
