"""Exercise the board read path on matched room microphones with and without the address."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import replay_bella_command_prefix as bella
import check_bella_weak_playback as weak

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mac_board_marker"))
import run_guest

ch, study, tx = bella.ch, bella.study, bella.tx
OUT = ch.RESULTS / "4090-c161-bella-board-reference.json"
BASE = ch.RESULTS / "4090-c161-mac-board-marker-p4.json"
WORK = Path(".local/wakeword/c161-bella-board-reference")


def pairs():
    baseline = ch.read(bella.OUT)
    panel, sr = sf.read(study.mixed.panel.demo.WAVE,dtype="int16")
    assert sr == 16000 and ch.sha(weak.PACK) == run_guest.BELLA_SHA
    music, _, music_id = tx.study.grid.room.rooms.read_noise("music")
    v5, _ = tx.study.panel_wave()
    filters, _ = bella.gains.direction.filters()
    with ZipFile(weak.PACK) as pack:
        files = {r["file"]:r for r in json.loads(pack.read("manifest.json"))["records"]}
        for row in baseline["captures"]:
            s = row["source"]
            file = f'{row["batch"]}/{s["index"]:03}.wav'
            positive = study.playback.read_wav(pack,files[file])
            negative = positive.copy()
            negative[:s["command_at"]] = 0
            known, = [r for r in baseline["records"] if r["source"] == s and r["batch"] == row["batch"]]
            # Recover the pre-convolution extent from the independently recorded I2S read count.
            # READY records the unpadded input length. Verify both reconstructed SHA256s.
            logname = f'events/{row["index"]:03}.txt'
            with ZipFile(bella.OUT.with_suffix(".zip")) as logs:
                ready, = (l.split() for l in logs.read(logname).decode().splitlines() if l.startswith("READY "))
            count = int(ready[2])
            _, (hx, _, _) = tx.diag.room(s["geometry"])
            length = count-4800-len(hx)+1
            group = {"length":(files[file],range(length))}
            scene = study.mixed.room_scene(s,group,panel,v5,music,music_id)
            ref = np.zeros(count,dtype="<i2")
            ref[s["panel_at"]:s["panel_at"]+len(panel)] = panel
            for role, wave, digest in (("fixed",positive,row["mic_sha256"]),
                                       ("no_address",negative,known["no_address_pcm_sha256"])):
                mic, _ = scene(bella.rear_filter(wave,filters["back"]))
                assert len(mic) == count and ch.digest(mic) == digest
                yield role+"/"+file, mic.tobytes(), ref.tobytes(), None, s["command_at"]


def main():
    expected = ch.read(BASE)
    guest = Path(".local/wakeword/c161-mac-board-marker/guest/board.bin")
    sim = Path(".local/wakeword/c161-panel-middle-uart/build")
    qemu = Path(".local/wakeword/qemu-p4-release/qemu/bin/qemu-system-riscv32")
    assert ch.sha(guest) == expected["guest_sha256"] and ch.sha(qemu) == expected["qemu_sha256"]
    WORK.mkdir(exist_ok=True,parents=True)
    with TemporaryDirectory(dir=WORK) as tmp:
        work = Path(tmp)
        data = run_guest.replay_cases(guest,sim,qemu,work,pairs())
        archive = OUT.with_suffix(".zip")
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as output:
            for p in sorted((work/"uart").glob("*.log")):
                output.write(p,"uart/"+p.name)
        assert len(data["records"]) == len({r["label"] for r in data["records"]}) == 48
        data.update(model_sha256=expected["model_sha256"],sources={p.name:ch.sha(p) for p in
                    (BASE,bella.OUT,weak.PACK,study.mixed.panel.demo.WAVE,Path(__file__))},
                    archive_sha256=ch.sha(work/archive.name), policy="24 existing matched rear mixes and their "
                    "24 verified address-erased controls. Restore the panel playout waveform in R at panel_at, "
                    "with no independent TX. Real board feeder, C161 and marker in digital P4 QEMU.")
        ch.save(work/OUT.name,data)
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
