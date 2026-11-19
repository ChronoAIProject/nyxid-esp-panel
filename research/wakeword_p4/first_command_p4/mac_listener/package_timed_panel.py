"""Package verified microphone/reference pairs, speaker tracks and research headers for Mac."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import soundfile as sf

import check_panel_marker_p4 as panel
import check_timed_final_p4 as single
from compare_room_willow_uplink import put_audio
from package_context_playback import add, formatted

study, ch = panel.study, single.ch
PACK = ch.RESULTS / "c161-timed-panel-mac-ab.zip"
HOST = study.OUT.with_name(study.OUT.stem + "-timed.json")
BOARD = HOST.with_name(HOST.stem + "-p4.json")


def verify(archive):
    with ZipFile(archive) as z:
        assert z.testzip() is None
        manifest = json.loads(z.read("manifest.json"))
        assert len(z.namelist()) == len(manifest["files"]) + 1
        assert set(z.namelist()) == set(manifest["files"]) | {"manifest.json"}
        for name, info in manifest["files"].items():
            data = z.read(name)
            assert len(data) == info["bytes"] and sha256(data).hexdigest() == info["sha256"]
            if name.endswith(".wav"):
                x, sr = sf.read(BytesIO(data), dtype="int16")
                assert sr == 16000 and x.ndim == 1 and ch.digest(x) == info["pcm_sha256"]


def main():
    board = ch.read(BOARD)
    versions = board["versions"]
    single_board = ch.read(single.OUT)
    assert versions["original"] == single_board["versions"]["candidate"]
    assert versions["model"] == single_board["versions"]["model"] == panel.phase.listener.C161
    archive = BOARD.with_suffix(".zip")
    assert ch.sha(archive) == board["archive_sha256"]
    indices = {r["host_index"]:r for r in board["records"]}
    root = panel.WORK.with_name(panel.WORK.name + "-timed")
    compiled = root / "src/research/wakeword_p4/first_command_p4/mac_listener/main"
    assert (compiled/"panel_marker_pair.h").read_text() == study.reference_header()
    assert (compiled/"final_marker.h").read_bytes() == single.timed.HEADER.read_bytes()
    tmp = PACK.with_suffix(".tmp")
    cases = []
    v5, bca = study.panel_wave()
    commands = study.grid.room.rooms.source_bank()
    pcm_files = {}
    with ZipFile(tmp, "w") as z, ZipFile(archive) as logs:
        def audio(path, samples):
            digest = ch.digest(samples)
            if digest not in pcm_files:
                pcm_files[digest] = put_audio(z, path, samples)["file"]
            return pcm_files[digest]

        speaker_v5 = audio("speaker/original-v5.wav", v5)
        speaker_bca = audio("speaker/panel-bca.wav", bca)
        assert ch.sha(study.WAV) == sha256(z.read(speaker_bca)).hexdigest()
        assert ch.sha(study.grid.fit.fixed.CUE) == sha256(z.read(speaker_v5)).hexdigest()
        for file in (single.timed.HEADER, study.COEFF, compiled/"panel_marker_pair.h"):
            add(z, "receiver/" + file.name, file.read_bytes())
        for i,(meta, name, role, mic, ref) in enumerate(study.simultaneous()):
            # All four nominal-level simultaneous geometries. Include one documented weak simultaneous miss.
            if meta["offset"] or (meta["external_db"] != 0 and (meta["geometry"],meta["external_db"]) != (2,-6)):
                continue
            if role == "external" and name == "bca":
                continue
            r = indices[i]
            assert (r["source"],r["name"],r["role"]) == (meta,name,role)
            assert (ch.digest(mic),ch.digest(ref)) == (r["mic_sha256"],r["reference_sha256"])
            scene = f'g{meta["geometry"]}_{meta["external_db"]}dB'
            command = commands[meta["voice"],meta["command"]]
            assert ch.digest(command) == meta["command_pcm_sha256"]
            speaker = audio(f'speaker/external-g{meta["geometry"]}.wav', np.r_[v5,command])
            mic_file = audio(f"digital/{scene}/{name}-{role}-M.wav", mic)
            ref_file = audio(f"digital/{scene}/{name}-{role}-R.wav", ref)
            uart = f'uart/{r["index"]:03}.txt'
            add(z, uart, logs.read(f'events/{r["index"]:03}.txt'))
            row = dict(source=meta, order=name, role=role, mic=mic_file, ref=ref_file, uart=uart,
                       panel_speaker=speaker_v5 if name=="abc" else speaker_bca,
                       external_speaker=speaker, opened=r["opened"], small=r["small"], turbo=r["turbo"])
            if "uplink" in r:
                row["uplink"] = {k:r["uplink"][k] for k in ("samples","fifo_sha256","packets","pcm_sha256")}
            cases.append(row)
        assert len(cases) == 25
        neg_meta, _, _, mic, ref, _, _ = single.false_alert()
        negative = dict(source=neg_meta, mic=audio("speech/across-M.wav",mic), ref=audio("speech/silent-R.wav",ref))
        reservation = ch.read(ch.RESULTS/"4090-jenny-earnings-development-reservation-v1.json")
        negative["original_url"], = (r["source_url"] for r in reservation["records"] if r["source_id"]=="4483680")
        negative["attribution"] = "Eight-second excerpt of Earnings-22 recording 4483680 (Rev.com speech-datasets), decoded to mono PCM16 at 16 kHz."
        with TemporaryDirectory() as folder:
            work = Path(folder)
            matcher = single.timed.binding(work)
            prefix = ch.binding(work/"prefix.so",coded="aligned")
            row = panel.phase.handoff.probe.score(work, neg_meta, mic, ref, matcher, 0, build=root/"build", prefix=prefix,
                       qualify=panel.phase.handoff.probe.qualify, wait_for_third=True, frame_activity=True, qualify_capture_frame=True)
            assert row["opened"] == row["c161"] == row["fixed"] == []
            negative.update(opened=row["opened"], uart="uart/across-combined.txt")
            add(z,negative["uart"],("\n".join(row["uart_events"])+"\n").encode())
        files = {n:dict(bytes=len(z.read(n)),sha256=sha256(z.read(n)).hexdigest()) for n in z.namelist()}
        for digest, name in pcm_files.items():
            files[name]["pcm_sha256"] = digest
        add(z,"manifest.json",formatted(dict(model_sha256=versions["model"], threshold=0.91796875, smoothing_steps=5,
            sample_rate=16000, encoding="mono PCM16 WAV", priming_samples=81600, block_samples=256,
            firmware={k:versions[k] for k in ("original","panel_order","codec")},
            sources={p.name:ch.sha(p) for p in (HOST, BOARD, single.OUT, archive, Path(__file__))},
            integration="Both modes use nyx::TimedFinalMarker on M and the original v5 self-play guard on R. "
                 "Order abc plays original-v5 on the panel. Order bca plays panel-bca on the panel and updates "
                 "nyx::PanelMarkerPair on every R sample. From an electrical B,C match through the following 5432 "
                 "samples block the C161 route, including pending C161 hits. Run the ordinary microphone cue route "
                 "and the original playback reference guard in both modes. See build_panel_marker_order.sh timed.",
            replay="For digital replay inject the listed M and R from their first sample with the same clock. "
                   "For physical playback, role panel uses the indicated panel WAV only, external uses the external "
                   "speaker WAV only, both starts both together. In each room hold positions and gains fixed between "
                   "orders. External-only recordings are shared by both orders. All OPENs are in samples at 16 kHz. "
                   "Record the actual M/R, playback and OPEN clocks, first Opus packet, decoded command, slowest "
                   "capture block and minimum internal heap on each board.", cases=cases, ordinary_speech=negative, files=files)))
    verify(tmp)
    tmp.replace(PACK)
    print(PACK.stat().st_size,ch.sha(PACK))


if __name__ == "__main__":
    main()
