"""Trace the two known 240° misses through isolated sources and the native P4 listener."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.io import loadmat
from scipy.signal import fftconvolve, resample_poly
import torch

import bella_headroom_new_angles as angles
import diagnose_bella_player_marker as trace
import package_bella_headroom_comparison as pack

fit, P = pack.fit, pack.P
receiver = fit.receiver
fixed = receiver.test.fixed
WORK = Path(".local/wakeword/bella-240-masking")
OUT = angles.OUT.with_name("4090-bella-240-masking.json")
PROTO = angles.PROTO.with_name("bella_240_masking_protocol.md")
LEVELS = {"original": 1., "minus6": 10**(-6/20), "minus12": 10**(-12/20), "target-only": 0.}


def stems_for(orientations, roles):
    previous = P.read(angles.OUT)
    assert previous["plan_sha256"] == P.sha(angles.PLAN)
    records = {r["label"]: r for r in previous["records"]}
    paths = {(p["angle"], p["direction"]): p for p in P.read(angles.PLAN)["paths"]}
    scale = P.read(fit.study.PLAN)["paths"][0]["common_scale"]
    rng = np.random.Generator(np.random.PCG64(927463))
    assert P.sha(pack.PACK) == P.read(pack.OUT)["archive_sha256"]
    with ZipFile(pack.PACK) as z:
        manifest = json.loads(z.read("manifest.json"))
        for angle in (120, 240):
            mat = angles.WORK / f"angle_{angle}.0.mat"
            assert P.sha(mat) == angles.ANGLES[angle]
            brir = loadmat(mat)["brir"]
            responses = {name: resample_poly(brir[:, 1, ch], 1, 3)/scale for name, ch in (("front", 2), ("back", 8))}
            for side, h in responses.items():
                path = paths[angle, side]
                assert P.digest(h) == path["rir_sha256"]
                other = responses["back" if side == "front" else "front"]
                for scene in (77, 79, 90):
                    prefix = f"{scene:03}/"
                    dry = fit.speakers.unpack(z.read(prefix + "interruption.wav"))
                    noise = rng.normal(0, 12, len(dry) + len(other) - 1)
                    if angle not in orientations:
                        continue
                    talk = .25 * fftconvolve(dry.astype(float), other)
                    for role in roles:
                        name = prefix + role + ".wav"
                        source = fit.speakers.unpack(z.read(name))
                        target = .25 * fftconvolve(source.astype(float), h)
                        row = records[f"{angle}-{side}-{scene:03}-{role}"]
                        end = manifest["files"][name]["command_at"] + path["arrival"]
                        assert end == row["end"]
                        yield row, source, target, talk, noise


def scenes():
    for row, _, target, talk, noise in stems_for({240}, ("early-nyx", "headroom")):
        for mode, gain in LEVELS.items():
            yield row, row["role"], mode, target, gain*talk, noise
        if row["role"] == "headroom":
            yield row, "interruption", "interruption-only", np.zeros_like(talk), talk, noise


def measure_triplet(mic, target, interruption, starts, fir, patterns):
    ends = [s+1280 for s in starts]
    parts = trace.correlations(mic, [round(s/4) for s in ends], fir, patterns)
    peaks = []
    for (ticks, corr), start in zip(parts, starts, strict=True):
        i = int(np.argmax(np.abs(corr)))
        window = slice(start, start+1280)
        peaks.append(dict(sample=int(ticks[i])*4, signed_correlation=float(corr[i]),
                          min_signed=float(corr.min()), max_signed=float(corr.max()),
                          target_rms=float(np.sqrt(np.mean(target[window]**2))),
                          interruption_rms=float(np.sqrt(np.mean(interruption[window]**2)))))
    (ta, a), (tb, b), _ = parts
    pairs = {}
    for sign in (-1, 1):
        first = ta[a*sign >= .3]
        middle = tb[b*sign >= .3]
        paired = np.any(np.abs(middle[:, None] - first - 640) <= 6, axis=1)
        pairs[str(sign)] = (4*middle[paired]).tolist()
    return dict(expected_end_samples=ends, peaks=peaks, paired_middle_samples=pairs,
                chain_sample=trace.chain(parts, (640, 480)))


def main():
    torch.set_num_threads(1)
    device = torch.cuda.get_device_name()
    assert "4090" in device
    fir = trace.coefficients("kCueBandFirV5")
    coeffs = trace.coefficients("kCueBandPatternsV5").reshape(3, 320)
    np.testing.assert_array_equal((coeffs**2).sum(1), trace.coefficients("kCueBandEnergyV5"))
    patterns = torch.tensor(coeffs, device="cuda", dtype=torch.float64)
    stems = [s["start"] for s in P.read(fit.early.fitted.FIT)["stems"]]
    advanced = P.read(fit.early.FIT)["advanced_samples"]
    shift = P.read(fit.OUT)["fit"]["winner"]["shift"]
    build = P.read(receiver.WORK / "build.json")
    assert build == P.read(angles.OUT)["build"] and P.sha(fixed.QEMU) == build["qemu_sha256"]
    flash = bytearray((receiver.WORK / "guest.flash").read_bytes())
    assert fixed.digest(flash) == build["flash_sha256"]
    WORK.mkdir(exist_ok=True, parents=True)
    cache = WORK / build["flash_sha256"]
    cache.mkdir(exist_ok=True)
    rows, files = [], {}
    with TemporaryDirectory(dir=WORK) as folder:
        work = Path(folder)
        archive = work / OUT.with_suffix(".zip").name
        with ZipFile(archive, "w", ZIP_DEFLATED) as logs:
            for r, role, mode, target, talk, noise in scenes():
                label = f'{r["direction"]}/{r["scene"]:03}/{role}/{mode}'
                total = np.rint(target + (talk + noise))
                assert total.max() <= 32767 and total.min() >= -32768
                mic = total.astype("<i2")
                if mode == "original":
                    assert P.digest(mic) == r["pcm_sha256"]
                ref = np.zeros_like(mic)
                assert P.digest(ref) == r["ref_sha256"]
                block = fixed.run_guest.payload(mic.tobytes(), ref.tobytes())
                active_cache = angles.WORK/build["flash_sha256"] if mode == "original" else cache
                native, log, failures = fixed.boot(work, flash, active_cache, block, 1)
                events = native[0]["events"]
                if mode == "original":
                    assert events == r["nyx"]
                for p in [log, *failures]:
                    member = label + ("/wake.log" if p == log else "/" + p.name)
                    logs.write(p, member)
                    files[member] = P.sha(p)
                origin = r["end"] - 23200
                # The shared interruption-only control uses A's nominal windows.
                offsets = {"early": -advanced + (shift if role == "headroom" else 0), "late": 0}
                groups = {k: measure_triplet(mic, target, talk, [origin+s+v for s in stems], fir, patterns)
                          for k, v in offsets.items()}
                chains = [g["chain_sample"] for g in groups.values() if g["chain_sample"] is not None]
                assert events["C"] == ([min(chains)] if chains else []), (label, events, chains)
                opened = next(iter(events["O"]), None)
                rows.append(dict(label=label, scene=r["scene"], direction=r["direction"], role=role, mode=mode,
                                 sample_rate=16000, start=r["start"] if role!="interruption" else None, end=r["end"],
                                 mic_sha256=P.digest(mic), ref_sha256=r["ref_sha256"], input_sha256=fixed.digest(block),
                                 events=events, timely=opened is not None and r["start"] <= opened < r["end"],
                                 triplets=groups))
                print(label, events, flush=True)
        assert len(rows) == len({r["label"] for r in rows}) == 54
        sources = (Path(__file__).resolve().relative_to(Path.cwd()), PROTO, angles.OUT, angles.PLAN, pack.OUT,
                   fit.early.fitted.FIT, fit.early.FIT, fit.OUT, fit.study.PLAN, trace.COEFF,
                   Path(trace.__file__).resolve().relative_to(Path.cwd()))
        P.save(work/OUT.name, dict(records=rows, sources={str(p): P.sha(p) for p in sources}, build=build,
                                   gpu=device, torch=str(torch.__version__), files=files, archive_sha256=P.sha(archive)))
        archive.replace(OUT.with_suffix(".zip"))
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
