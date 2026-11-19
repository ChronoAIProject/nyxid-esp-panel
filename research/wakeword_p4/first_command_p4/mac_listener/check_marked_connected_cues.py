"""Export one marked wake file per synthetic voice and test the cropped files in the P4 simulator."""
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import check_marked_connected_panel as mixed
from run_c161_first_command_v5 import transcripts

study, tx, ch, panel = mixed.study, mixed.tx, mixed.ch, mixed.panel
OUT = ch.RESULTS / "4090-c161-marked-connected-cues.json"
PACK = ch.RESULTS / "c161-four-voice-wake-cues.zip"


def main():
    spec, v5, _, markers = tx.study.grid.fit.original()
    history = ch.read(panel.OUT)
    builds = dict(bca=panel.middle.WORK / "build")
    assert ch.sha(builds["bca"] / "nyx_mac_listener.bin") == history["firmware"]
    assert tx.panel.grid.gap.uplink.firmware() == history["codec"]
    small, identity = ch.load_reader()
    assert history["small"] == identity
    turbo = tx.diag.reader(history["turbo"])
    decoder = tx.panel.grid.gap.uplink.decoder_api()
    rows, files = [], []
    tmp, pack_tmp = OUT.with_suffix(".tmp"), PACK.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(tmp, "w", ZIP_DEFLATED) as z, ZipFile(pack_tmp, "w", ZIP_DEFLATED) as pack:
        work = Path(folder)
        matcher = tx.single.timed.binding(work)
        native = tx.panel.grid.room.rooms.c161_binding()
        for (voice, command), group in study.sources().items():
            if command != study.connected.CARRIERS[0]:
                continue
            src, plain = group["Hey Nyx"]
            rendered = mixed.render(group,v5,spec,markers)
            for name, full in (("plain",plain),("marked",rendered["marked"]),("reversed_final",rendered["reversed_final"])):
                # Drop only the artificial leading 2 s from the original synthesis plan.
                clip = full[32000:src["end"]]
                buf = BytesIO()
                sf.write(buf, clip, 16000, format="WAV", subtype="PCM_16")
                data = buf.getvalue()
                path = f"{voice}/{name}.wav"
                pack.writestr(path,data)
                wav = dict(file=path, bytes=len(data), sha256=study.sha256(data).hexdigest(), pcm_sha256=ch.digest(clip),
                           samples=len(clip), source_index=src["index"], source_sha256=src["sha256"],
                           small_dry=transcripts(small,clip)["text"], turbo_dry=transcripts(turbo,clip)["text"])
                mic = np.pad(clip,(32000,32000))
                zero = np.zeros_like(mic)
                raw = tx.panel.grid.fit.fixed.ch.routes(matcher,native,mic)
                expected = panel.middle.pair.admit(raw,0,[],zero)
                meta = dict(start=src["start"],command_at=src["end"],command="Hey Nyx",voice=voice,variant=name)
                case = (src["index"],meta,"bca","external",mic,zero,expected)
                row = tx.panel.capture(work,z,len(rows),case,small,turbo,decoder,builds,tx_pair=0)
                timely = bool(row["opened"] and src["start"] <= row["opened"][0] < src["end"])
                files.append(dict(wav, open_sample=row["opened"],timely=timely))
                rows.append(dict(row, timely=timely, c161=expected["c161"], fixed=expected["fixed"], **wav))
        assert len(rows) == 12
        manifest = dict(rate=16000,format="mono signed PCM16 WAV",files=files, model=history["model"],
                        firmware=history["firmware"],threshold=0.91796875,smoothing_steps=5,
                        note="Each voice uses the first carrier. All files contain just the address. For digital "
                        "replay, prepend and append 32000 zero samples and feed zero reference, no panel event. "
                        "OPEN is relative to the resulting microphone input. Use the same player gain and "
                        "position for acoustic A/B. Marked and reversed files add synthesized v5 code to speech.")
        pack.writestr("manifest.json",json.dumps(manifest,indent=2)+"\n")
    for f,out in ((pack_tmp,PACK),(tmp,OUT.with_suffix(".zip"))):
        with ZipFile(f) as z:
            assert z.testzip() is None
        f.replace(out)
    ch.save(OUT,dict(records=rows,model=history["model"],firmware=history["firmware"],codec=history["codec"],
                     small=identity,turbo=history["turbo"], pack_sha256=ch.sha(PACK),archive_sha256=ch.sha(OUT.with_suffix(".zip")),
                     sources={p.name:ch.sha(p) for p in (study.connected.PLAN,study.connected.PACK,Path(mixed.__file__),Path(__file__))}))


if __name__ == "__main__":
    main()
