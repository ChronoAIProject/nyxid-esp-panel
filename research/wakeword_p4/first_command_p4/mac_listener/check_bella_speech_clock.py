"""Separate the Bella voice from the fixed markers on the existing 1.45 s playback clock."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import check_bella_final_s as final_s

cue, P = final_s.cue, final_s.P
OUT = P.RESULTS / "4090-c161-bella-speech-clock.json"
WAV = P.RESULTS / "hye-nyx-af-bella-separated-v1.wav"
WORK = Path(".local/wakeword/c161-bella-speech-clock")


def main():
    stems, info = final_s.make_tracks()
    speech = stems["speech"].astype(np.int32)
    original = stems["original"].astype(np.int32)
    codes = original - speech
    # Use only the first 200 ms of exact digital silence as timing slack.
    advance = 3200
    assert not speech[:advance].any()
    tail = info["phone_intervals"]["s"][1]
    trimmed = speech.copy()
    trimmed[tail-64:tail] = np.rint(trimmed[tail-64:tail] * (1+np.cos(np.linspace(0,np.pi,64)))/2)
    trimmed[tail:] = 0
    tracks = {"original": original}
    for label, stem in (("trim", trimmed), ("advance", speech), ("trim_advance", trimmed)):
        moved = np.pad(stem[advance:], (0,advance)) if "advance" in label else stem
        track = moved + codes
        assert np.abs(track).max() <= 32767 and np.array_equal(track - moved, codes)
        tracks[label] = track
    tracks = {k:v.astype("<i2") for k,v in tracks.items()}
    historical = P.read(final_s.OUT)
    assert historical["source"] == info
    proc, model, qid = cue.prior.load()
    assert qid == historical["qwen"] and qid["device"].endswith("RTX 4090")
    turbo = cue.prior.reader(historical["turbo"])
    dry = {}
    for label, wave in tracks.items():
        a = cue.prior.evaluate(proc,model,wave,"Hey Nix")
        b = cue.prior.recognition(turbo,wave,"Hey Nix")
        dry[label] = dict(pcm_sha256=P.digest(wave),qwen=a["text"],turbo=b["text"],words=b["words"])
        print(label, a["text"], b["text"],flush=True)
    previous, = [r for r in historical["records"] if r["label"] == "original"]
    assert all(dry["original"][k] == previous[k] for k in dry["original"])
    # The combined edit is the preselected room candidate regardless of the ASR output.
    candidate = tracks["trim_advance"]
    source = P.read(cue.overlap.OUT)
    assert P.sha(cue.GUEST) == source["guest_sha256"] and P.sha(P.QEMU) == source["qemu_sha256"]
    rows = []

    def cases():
        for label, wave in tracks.items():
            mic = np.pad(wave, (32000,32000))
            rows.append(dict(label=label,role="dry",mic_sha256=P.digest(mic)))
            yield label,mic.tobytes(),bytes(mic.nbytes),previous["events"] if label=="original" else None,None
        for old,mic,ref,clipped in cue.inputs(candidate):
            label = f"daily/{old['index']:03}"
            rows.append(dict(label=label,role="daily",index=old["index"],file=old["file"],command_at=old["command_at"],
                             original_open=old["events"]["O"],mic_sha256=P.digest(mic),reference_sha256=P.digest(ref),
                             clipped_samples=clipped))
            yield label,mic.tobytes(),ref.tobytes(),None,old["command_at"]

    WORK.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=WORK) as tmp:
        work = Path(tmp)
        guest = cue.overlap.board.run_guest.replay_cases(cue.GUEST,Path(".local/wakeword/c161-panel-middle-uart/build"),
                                                        P.QEMU,work,cases())
        archive = OUT.with_suffix(".zip")
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as z:
            for row, observed in zip(rows,guest["records"],strict=True):
                assert row["label"] == observed["label"]
                row.update(observed)
                log = work/"uart"/(row["input_sha256"]+".log")
                assert P.sha(log) == row["uart_sha256"]
                z.write(log,row["label"]+".log")
        daily = [r for r in rows if r["role"]=="daily"]
        assert [r["index"] for r in daily]==list(range(96))
        counts = {key:sum(bool(times) and times[0]<r["command_at"] for r in daily
                          for times in [r["events"]["O"] if key=="candidate" else r["original_open"]])
                  for key in ("original","candidate")}
        sf.write(work/WAV.name,candidate,16000,subtype="PCM_16")
        check,rate = sf.read(work/WAV.name,dtype="int16")
        assert rate==16000 and np.array_equal(check,candidate)
        P.save(work/OUT.name,dict(source=info,advance_samples=advance,trim_from=tail,trim_taper_samples=64,
               marker_residual_sha256=P.digest(codes.astype("<i2")),dry=dry,counts_before_command=counts,
               candidate_wav=WAV.name,candidate_sha256=P.sha(work/WAV.name),records=rows,
               guest_sha256=guest["guest_sha256"],qemu_sha256=guest["qemu_sha256"],model=source["model"],
               qwen=qid,turbo=historical["turbo"],archive_sha256=P.sha(work/archive.name),
               sources={p.name:P.sha(p) for p in (Path(__file__),Path(final_s.__file__),final_s.OUT,
                                                   Path(cue.__file__),cue.overlap.OUT)},
               policy="Factorial dry controls for dropping speech following the last nominal /s/ token and moving "
                      "the voice 200 ms earlier. Keep the original quantized marker residual at its original gain "
                      "and clock. Full dry CUDA ASR and P4 on all four. Preselected combined edit in all 96 "
                      "consumed daily mixtures with unchanged command clocks, panel ref, room and other voices."))
        for dest in (WAV,archive,OUT):
            (work/dest.name).replace(dest)
    print(counts,flush=True)


if __name__ == "__main__":
    main()
