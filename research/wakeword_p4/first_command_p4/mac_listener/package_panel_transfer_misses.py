"""Pack matched digital inputs, firmware clocks and two physical playback tracks for the four residuals."""
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import soundfile as sf

import check_quiet_panel_rms_transfer as study
from compare_room_willow_uplink import put_audio
from package_context_playback import add, formatted
from package_timed_panel import verify

ch, tx = study.ch, study.tx
PACK = ch.RESULTS / "c161-music-transfer-mac-ab.zip"


def main():
    native = ch.read(study.transfer.OUT)
    alternate = ch.read(study.OUT)
    misses = {r["scene"] for r in native["records"] if r["panel"]=="quiet" and r["role"]=="both" and not r["timely"]}
    assert len(misses)==4
    prior = {(r["scene"],r["role"]):r for r in native["records"] if r["panel"]=="quiet" and r["scene"] in misses}
    guest = {(r["host_index"],r["view"]):r for r in native["guest"] if r["panel"]=="quiet"}
    revised = {r["host_index"]:r for r in alternate["guest"] if r["group"]=="music"}
    clocks = {r["scene"]:r["transmit_event"] for r in alternate["records"] if r["group"]=="music"}
    cue, rate = sf.read(study.rms.WAVE, dtype="int16")
    assert rate==16000
    v5, _ = tx.study.panel_wave()
    builds = dict(bca=study.panel.middle.WORK/"build")
    assert ch.sha(builds["bca"]/"nyx_mac_listener.bin")==native["firmware"]==alternate["firmware"]
    scratch = PACK.with_suffix(".tmp")
    with ExitStack() as stack, TemporaryDirectory() as folder, ZipFile(scratch,"w") as z:
        work = Path(folder)
        archives = {}
        for doc, result in ((study.transfer.OUT,native),(study.OUT,alternate)):
            archive = doc.with_suffix(".zip")
            assert ch.sha(archive)==result["archive_sha256"]
            archives[doc] = stack.enter_context(ZipFile(archive))
        written, rows = {}, []
        def wave(name, pcm):
            digest = ch.digest(pcm)
            if digest not in written:
                written[digest] = put_audio(z,name,pcm)["file"]
            return written[digest]
        panel_demo, rate = sf.read(study.panel.demo.WAVE,dtype="int16")
        assert rate==16000
        panel_file = wave("speaker/panel-minus6.wav",panel_demo)
        speakers = None
        for i,(s,views) in enumerate(study.transfer.inputs(cue)):
            if i not in misses:
                continue
            if speakers is None:
                cmd = tx.study.grid.room.rooms.source_bank()[s["voice"],s["command"]]
                command, _ = tx.balanced.match_level(v5,cmd)
                assert ch.digest(command)==s["command_pcm_sha256"]
                speakers = dict(panel=panel_file, original=wave("speaker/external-v5.wav",np.r_[v5,command]),
                                rms=wave("speaker/external-rms.wav",np.r_[cue,command]),
                                voice=s["voice"], command=s["command"], command_pcm_sha256=ch.digest(command))
            assert s["command_pcm_sha256"]==speakers["command_pcm_sha256"]
            for name,role,pcm in (("v5","both",views["quiet","both"]),("rms","both",views["quiet_rms","both"]),
                                  ("no-cue","no_address",views["quiet","no_address"])):
                clock = clocks[i]
                label = f"digital/{i:03}-{name}"
                zero = np.zeros_like(pcm)
                if name=="rms" or (i,role) in guest:
                    r = revised[i] if name=="rms" else guest[i,role]
                    archive = archives[study.OUT if name=="rms" else study.transfer.OUT]
                    events = archive.read(f'events/{r["index"]:03}.txt')
                else:
                    expected = prior[i,role]["result"]
                    assert expected["opened"]==[]
                    # These baseline negatives did not run in the assigned subset.
                    # Obtain their UART from the actual firmware on the byte-checked input.
                    with ZipFile(work/"extra.zip","w") as fresh:
                        case=(i,s,"bca","both",pcm,zero,expected)
                        r = tx.panel.capture(work,fresh,0,case,None,None,None,builds,tx_pair=clock)
                        events = fresh.read("events/000.txt")
                assert r["mic_sha256"]==ch.digest(pcm) and r["reference_sha256"]==ch.digest(zero)
                assert events.endswith(b"LISTENER PASS\n")
                add(z,label+".txt",events)
                rows.append(dict(scene=i,mode=name,m=wave(label+"-M.wav",pcm),r=wave(label+"-R.wav",zero),
                                 uart=label+".txt",tx_capture_sample=clock,opened=r["opened"],source=s))
        assert len(rows)==12 and speakers is not None
        files={n:dict(bytes=len(z.read(n)),sha256=sha256(z.read(n)).hexdigest()) for n in z.namelist()}
        for digest,n in written.items():
            files[n]["pcm_sha256"]=digest
        add(z,"manifest.json",formatted(dict(files=files,cases=rows,speakers=speakers,model_sha256=alternate["model"],
            threshold=0.91796875,smoothing_steps=5,firmware_sha256=native["firmware"],
            sources={p.name:ch.sha(p) for p in (study.transfer.OUT,study.OUT,study.rms.WAVE,study.panel.demo.WAVE,Path(__file__))},
            sample_rate=16000,prime_samples=81600,format="Mono PCM16. Digital M/R aligned at capture origin. "
            "Supply the TX marker from the manifest, 1-based. Physical speaker tracks contain only the dry "
            "sources. For each pair use the same hardware volumes and positions; start the external speaker "
            "source.lag samples after the panel. The digital M includes room and background music.")))
    verify(scratch)
    scratch.replace(PACK)
    print(PACK.stat().st_size,ch.sha(PACK))


if __name__ == "__main__":
    main()
