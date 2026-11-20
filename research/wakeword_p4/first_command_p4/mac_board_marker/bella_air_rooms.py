"""Replay the released stereo source at two levels in the Aachen hall and stairway."""
from io import BytesIO
from pathlib import Path
import struct
import sys
from urllib.request import Request, urlopen
from zipfile import ZipFile
import zlib

import bella_surrey_orientations as study

WORK = Path(".local/wakeword/bella-air-rooms")
OUT = study.OUT.with_name("4090-bella-air-rooms.json")
PLAN = OUT.with_name(OUT.stem+"-plan.json")
PROTO = study.PROTOCOL.with_name("bella_air_rooms_protocol.md")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())
SLR = "https://www.openslr.org/resources/28/rirs_noises.zip"
# Local-file header offset, compressed size, SHA256 of decompressed member.
RESPONSES = {
    "aula": (("aula_carolina_1_3_0_3", 569059277, 171191, "66f39c808c06de7023509aba8debc7608ae09970e6803a434f7b837b48a79de7"),
             ("aula_carolina_1_3_180_3", 569413313, 181779, "57266778b040fabb4c4505826621db87226963fc2541192396f592ff56e4332c")),
    "stairway": (("stairway_1_1_0", 572298515, 48273, "c936c2c84f79d2549ad40a1ce2b23573efa650f9bb373d86a4f842bae13cf43f"),
                 ("stairway_1_1_180", 572641518, 50816, "0bf9e9efdb234d401af7325d8b0ee609fb6ee72c44a445c52342fb401489d8f4")),
}


def response_file(entry):
    stem, offset, size, sha = entry
    path = WORK / ("air_type1_air_binaural_" + stem + ".wav")
    member = "RIRS_NOISES/real_rirs_isotropic_noises/" + path.name
    if not path.exists():
        end = offset+size+255
        with urlopen(Request(SLR, headers={"Range": f"bytes={offset}-{end}"}), timeout=40) as r:
            assert r.status == 206 and r.headers["Content-Range"].startswith(f"bytes {offset}-{end}/")
            block = r.read(end-offset+2)
        assert block[:4] == b"PK\x03\x04"
        n, extra = struct.unpack_from("<HH", block, 26)
        assert block[30:30+n].decode() == member
        start = 30+n+extra
        raw = zlib.decompress(block[start:start+size], -15)
        assert study.sha256(raw).hexdigest() == sha
        path.write_bytes(raw)
    assert study.sha256(path.read_bytes()).hexdigest() == sha
    return path


def receive(wave, h, other, noise):
    import numpy as np
    import torch
    from scipy.signal import fftconvolve

    count = len(noise)
    nfft = 1 << (len(wave)+len(h)-2).bit_length()
    x = torch.tensor(wave.T.copy(), dtype=torch.float64, device="cuda")
    paths = torch.tensor(np.stack((h, other)), dtype=torch.float64, device="cuda")
    lanes = torch.fft.irfft(torch.fft.rfft(x, n=nfft)*torch.fft.rfft(paths, n=nfft), n=nfft)
    summed = lanes[:, :count].sum(0).cpu().numpy() + noise
    digitized = np.rint(summed).clip(-32768, 32767).astype("<i2")
    cpu = fftconvolve(wave[:, 0], h)[:count] + fftconvolve(wave[:, 1], other)[:count] + noise
    assert np.array_equal(digitized, np.rint(cpu).clip(-32768, 32767).astype("<i2"))
    return digitized, np.rint(summed) != digitized


def prepare():
    import json
    import numpy as np
    import soundfile as sf
    import torch
    import package_bella_6db_stereo as stereo

    torch.set_num_threads(1)
    gpu = torch.cuda.get_device_name()
    assert "4090" in gpu
    WORK.mkdir(exist_ok=True, parents=True)
    P = stereo.P
    previous = P.read(stereo.player.PLAN)
    files = {v: previous["files"][v] for v in previous["runtime"].values()}
    for p in (SCRIPT, PROTO, stereo.PACK, stereo.OUT, stereo.SOURCE.PACK):
        files[str(p)] = P.sha(p)
    assert P.sha(stereo.PACK) == P.read(stereo.OUT)["archive_sha256"]
    assert P.sha(stereo.SOURCE.PACK) == P.read(stereo.SOURCE.OUT)["archive_sha256"]
    rng = np.random.Generator(np.random.PCG64(209473))
    rows, geometry = [], []
    with ZipFile(stereo.PACK) as z, ZipFile(stereo.SOURCE.PACK) as base:
        dry_meta = json.loads(base.read("manifest.json"))["files"]
        for room, entries in RESPONSES.items():
            paths = []
            for entry in entries:
                p = response_file(entry)
                files[str(p)] = entry[-1]
                ir, rate = sf.read(p)
                assert rate == 16000 and ir.ndim == 2 and ir.shape[1] == 2
                paths.append(ir[:, 1])
            scale = float(np.linalg.norm(paths[0]))
            for index, label in enumerate(("az0", "az180")):
                h, other = paths[index]/scale, paths[1-index]/scale
                direct = int(np.abs(h[:800]).argmax())
                geometry.append(dict(room=room, direction=label, norm_az0=scale, arrival=direct,
                                     target_ir_sha256=P.digest(h), interruption_ir_sha256=P.digest(other)))
                for scene in (77, 79, 90):
                    info = dry_meta[f"{scene:03}/early-nyx.wav"]
                    count = info["samples"]+20000
                    noise = rng.normal(0, 12, count)
                    ref = np.zeros(count, dtype="<i2")
                    ref_path = WORK/f"ref-{count}.wav"
                    sf.write(ref_path, ref, 16000, subtype="PCM_16")
                    files[str(ref_path)] = P.sha(ref_path)
                    for role in ("early-nyx", "headroom", "willow", "absent"):
                        for level in ("baseline", "plus6"):
                            source = f"{scene:03}/{role}-{level}.wav"
                            wave, rate = sf.read(BytesIO(z.read(source)))
                            assert rate == 16000 and wave.shape == (info["samples"], 2)
                            mic, clipped = receive(32768*wave, h, other, noise)
                            end = info["command_at"]+direct
                            start = None if role == "absent" else int(np.flatnonzero(wave[:info["command_at"], 0])[0])+direct
                            key = f"{room}-{label}-{scene:03}-{role}-{level}"
                            path = WORK/(key+".wav")
                            sf.write(path, mic, 16000, subtype="PCM_16")
                            files[str(path)] = P.sha(path)
                            rows.append(dict(label=key, room=room, direction=label, scene=scene, source=source,
                                             mode=level, role=role, command=info["command"], start=start, end=end,
                                             samples=count, clipped=int(clipped.sum()), clipped_before_command=int(clipped[:end].sum()),
                                             path=str(path), sha256=files[str(path)], pcm_sha256=P.digest(mic),
                                             reference_path=str(ref_path), ref_sha256=P.digest(ref)))
    assert len(rows) == len({r["pcm_sha256"] for r in rows}) == 96
    P.save(PLAN, dict(records=rows, paths=geometry, files=files, runtime=previous["runtime"], gpu=gpu,
                      torch=str(torch.__version__), ir_distribution=SLR, threshold_requests=["DEFAULT"]))


if __name__ == "__main__":
    stage = sys.argv.pop(1)
    if stage == "prepare":
        prepare()
    elif stage == "case":
        sys.path.insert(0, str(study.ROOT/"qemu_hey_reference_v2"))
        import run_willow_mac_matched_prime_v1 as official
        official.PLAN, official.__file__ = PLAN, str(SCRIPT)
        official.case()
    elif stage == "willow":
        from check_address_gain_wakes import willow_cases
        willow_cases(PLAN, WORK, SCRIPT)
    else:
        study.WORK, study.PLAN, study.OUT = WORK, PLAN, OUT
        {"nyx": study.nyx, "report": study.report}[stage]()
        if stage == "report":
            (WORK/"4090-bella-surrey-orientations-wake-uart.zip").replace(OUT.with_suffix(".zip"))
