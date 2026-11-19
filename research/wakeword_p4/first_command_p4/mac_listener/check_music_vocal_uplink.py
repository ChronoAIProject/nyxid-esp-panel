"""Use the published MUSDB-HQ-only Hybrid Demucs vocals estimate on complete decoded P4 calls."""
from collections import Counter, defaultdict
import importlib.util
import io
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile, ZIP_STORED

from faster_whisper import WhisperModel
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch

import check_marker_music_commands as music
import compare_room_willow_uplink as room_willow
import score_willow_music_cases as willow

ch, prior, grid = music.ch, music.prior, music.grid
OUT = ch.RESULTS / "4090-c161-hdemucs-command-uplink.json"
WORK = Path(".local/wakeword/c161-hdemucs")
ASSETS = {
    "_hdemucs.py": ("https://raw.githubusercontent.com/pytorch/audio/v2.8.0/src/torchaudio/models/_hdemucs.py",
                    "54f9d0ef7940f657c0c518d9f3934218960fdfa98f3704e34be4d4e2a7a5bbf9"),
    "musdbhq.pt": ("https://download.pytorch.org/torchaudio/models/hdemucs_high_musdbhq_only.pt",
                   "85ce4420e0852344b816487b4c4bb821bd406376b93bf684b2b76aa56404c46a"),
}


def load_separator():
    for name, (_, checksum) in ASSETS.items():
        assert ch.sha(WORK / name) == checksum
    spec = importlib.util.spec_from_file_location("pinned_hdemucs", WORK / "_hdemucs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.hdemucs_high(sources=["drums", "bass", "other", "vocals"])
    model.load_state_dict(torch.load(WORK / "musdbhq.pt", weights_only=True, map_location="cpu"))
    return model.eval().cuda()


@torch.inference_mode()
def isolate(model, pcm):
    # Each call fits in one segment. Use the tutorial's stereo normalization and undo it on output.
    mono = resample_poly(pcm.astype(np.float32) / 32768, 441, 160)
    x = torch.from_numpy(mono).cuda()
    mean, std = x.mean(), x.std().clamp_min(1e-8)
    pair = ((x - mean) / std).repeat(1, 2, 1)
    vocals = model(pair)[0, model.sources.index("vocals")].mean(0) * std + mean
    y = resample_poly(vocals.cpu().numpy(), 160, 441)[:len(pcm)]
    assert len(y) == len(pcm) and np.isfinite(y).all()
    return grid.room.quantize(y * 32768)


def nyx_cases(path, doc, logs, decoder):
    for idx, row in enumerate(doc["records"]):
        src = row["board"]["source"]
        name = row.get("cue", row.get("name"))
        if name not in ("v5", "rms") or row.get("local"):
            continue
        if path == music.prior.OUT and row["group"] != "common":
            continue
        if path == music.server.strong.OUT and row["group"] != "music":
            continue
        if path == music.OUT and "assigned" not in row["groups"]:
            continue
        group = "common" if "scene" in src else "assigned_music"
        scene = src["scene"] if group == "common" else src.get("grid_index", src.get("index"))
        opens = row["board"]["opened"]
        cap = row.get("uplink")
        case = dict(cue=name, group=group, scene=scene, report_row=idx, command=src["carrier"],
                    opened=opens, command_at=src["command_at"], timely=bool(opens and src["cue_at"] <= opens[0] < src["command_at"]))
        wave = None
        if cap:
            wave, count = prior.audio.decoded(logs, f"{idx:02}-opus.log", decoder, idx, cap["samples"], 0, cap["pcm_sha256"])
            assert count == cap["packets"]
            turbo = cap["turbo"] if "turbo" in cap else row["views"]["turbo"]["original"]
            case.update(original=dict(small=cap["recognition"], turbo=turbo), packets=count)
        assert bool(opens) == (wave is not None)
        yield case, wave


def willow_cases(path, doc, logs, decoder):
    common = path == room_willow.OUT
    for idx, row in enumerate(doc["records"]):
        if not common and row["role"] != "matched_rms":
            continue
        route = row["routes"]["willow"] if common else row
        opened = route.get("opened")
        case = dict(cue="willow", group="common" if common else "assigned_music",
                    scene=row["scene"] if common else row["grid_index"], report_row=idx,
                    command=row["command"] if common else row["carrier"], opened=[] if opened is None else [opened],
                    command_at=row["command_at"], timely=opened is not None and opened < row["command_at"])
        wave = None
        if opened is not None:
            cap = route["capture"]
            lane = cap["lanes"]["selected"]
            boot_id = route["id"] if common else idx
            wave, count = prior.audio.decoded(logs, f"opus/{boot_id:03}-willow.log", decoder, boot_id,
                                              cap["emitted"] - cap["first"], 0, lane["decoded_sha256"])
            assert count == lane["packets"]
            case.update(original=route["full"] if common else dict(small=lane["asr"], turbo=row["turbo"]), packets=count)
        yield case, wave


def run():
    torch.set_num_threads(1)
    separator = load_separator()
    small, identity = ch.load_reader()
    info = ch.read(music.OUT)["turbo"]
    model_dir = prior.audio.prior.context.prior.audit.TURBO
    assert all(ch.sha(model_dir / n) == h for n, h in info["files"].items())
    turbo = WhisperModel(str(model_dir), device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
    assert small.model.device == turbo.model.device == "cuda"
    decoder = grid.gap.uplink.decoder_api()
    sources, rows = {}, []
    stats = defaultdict(lambda: defaultdict(Counter))
    archive = OUT.with_suffix(".zip")
    tmp = archive.with_suffix(".tmp")
    with ZipFile(tmp, "w", ZIP_STORED) as outputs:
        for path in (music.prior.OUT, music.normal.OUT, music.server.strong.OUT, music.OUT, room_willow.OUT, willow.OUT):
            doc = ch.read(path)
            uart = path.with_suffix(".zip")
            assert doc["small"] == identity and doc["turbo"] == info and ch.sha(uart) == doc["archive_sha256"]
            sources.update({path.name: ch.sha(path), uart.name: doc["archive_sha256"]})
            walk = willow_cases if path in (room_willow.OUT, willow.OUT) else nyx_cases
            with ZipFile(uart) as logs:
                for case, wave in walk(path, doc, logs, decoder):
                    case["source"] = path.name
                    group, name = case["group"], case["cue"]
                    tally = stats[group][name]
                    tally.update(trials=1, opened=wave is not None, timely=case["timely"])
                    if wave is not None:
                        t = perf_counter()
                        y = isolate(separator, wave)
                        elapsed = perf_counter() - t
                        case.update(input_pcm_sha256=ch.digest(wave), output_pcm_sha256=ch.digest(y), samples=len(y),
                                    seconds=elapsed, separated={}, member=f"{group}/{case['scene']:03}-{name}.flac")
                        clip = io.BytesIO()
                        sf.write(clip, y, 16000, format="FLAC", subtype="PCM_16")
                        outputs.writestr(case["member"], clip.getvalue())
                        for size, model in (("small", small), ("turbo", turbo)):
                            score = grid.gap.uplink.recognition(model, y, case["command"])
                            case["separated"][size] = score
                            old = case["original"][size]["complete_suffix"]
                            new = score["complete_suffix"]
                            tally.update({size + "_raw": old, size + "_vocals": new,
                                          size + "_gained": new and not old, size + "_lost": old and not new})
                    rows.append(case)
                    print(len(rows), group, name, case["scene"], flush=True)
    assert all(len({r["scene"] for r in rows if r["cue"] == n and r["group"] == g}) ==
               stats[g][n]["trials"] == count for n in ("rms", "v5", "willow") for g, count in (("common", 16), ("assigned_music", 11)))
    with ZipFile(tmp) as z:
        assert z.testzip() is None and len(z.namelist()) == sum(bool(r["opened"]) for r in rows)
        for r in rows:
            if r["opened"]:
                x, rate = sf.read(io.BytesIO(z.read(r["member"])), dtype="int16")
                assert rate == 16000 and ch.digest(x) == r["output_pcm_sha256"]
    tmp.replace(archive)
    ch.save(OUT, dict(records=rows, summary=stats, reads=dict(small=identity, turbo=info), assets=ASSETS,
                      sources={**sources, Path(__file__).name: ch.sha(Path(__file__))}, archive_sha256=ch.sha(archive),
                      gpu=torch.cuda.get_device_name(), torch=torch.__version__, policy="Full-call offline separation "
                      "after P4 Opus decode. Same frozen MUSDB-HQ-only Hybrid Demucs applied to every opened call. "
                      "Duplicate mono at 44.1 kHz, standardize by mixture mean and standard deviation, take the "
                      "published vocal stem, average channels, restore scale and 16 kHz. Original PCM and scores "
                      "come from content-verified archives. Count every non-open against command retention. Willow "
                      "uses native level on the 16 common calls and RMS-matched level on the eleven assigned tracks. "
                      "Consumed repeated commands. Offline GPU inference duration excludes ASR and model load."))


if __name__ == "__main__":
    run()
