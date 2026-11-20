"""Replay the three fixed calls as panel self-play on the same four documented panel room paths."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import check_bella_marker_only as cue

P = cue.P
OUT = P.RESULTS / "4090-c161-bella-marker-only-self.json"
HISTORY = P.RESULTS / "4090-c161-eight-voice-self-play.json"


def main():
    geometry = {r["source"]["geometry"]:r["source"]["panel_rir_sha256"]
                for r in P.read(cue.overlap.board.bella.OUT)["captures"]}
    previous = {r["lane"]:r for r in P.read(HISTORY)["records"] if r["voice"]=="af_bella" and r["version"]=="enhanced"}
    original,rate = sf.read(cue.overlap.board.bella.CUE,dtype="int16")
    loud,other = sf.read(cue.overlap.peak.CUE,dtype="int16")
    candidate,_ = cue.make_cue()
    gain_report = P.read(cue.overlap.peak.OUT)
    assert rate==other==16000 and P.digest(original)==previous["synchronous"]["dry_sha256"]
    assert P.sha(cue.overlap.peak.CUE)==gain_report["cue_sha256"]
    tracks = dict(original=original,enhanced=loud,marker_only=candidate)
    parent = P.read(cue.overlap.OUT)
    assert P.sha(cue.GUEST)==parent["guest_sha256"] and P.sha(P.QEMU)==parent["qemu_sha256"]
    rows = []
    def cases():
        for g,checked_hash in sorted(geometry.items()):
            (impulse,path,digest),_ = cue.overlap.tx.diag.room(g)
            assert digest==checked_hash
            waves = {name:fftconvolve(wav.astype(float),impulse) for name,wav in tracks.items()}
            # Reproduce the published Bella bedroom microphone. Each other path has one
            # common gain giving the original-level mic a 16,000-count peak before rounding.
            historical = digest==previous["synchronous"]["room_sha256"]
            scale = previous["synchronous"]["gain"] if historical else 16000/np.max(np.abs(waves["original"]))
            for name,wave in tracks.items():
                wet = np.rint(waves[name]*scale)
                clipped = int(np.count_nonzero((wet < -32768) | (wet > 32767)))
                mic=np.pad(np.clip(wet,-32768,32767).astype("<i2"),(32000,32000))
                for lane in ("silent","synchronous"):
                    reference=np.zeros_like(mic)
                    if lane=="synchronous":
                        reference[32000:32000+len(wave)] = wave
                    if historical and name=="original":
                        assert P.digest(mic)==previous[lane]["mic_sha256"]
                        assert P.digest(reference)==previous[lane]["reference_sha256"]
                    label=f"g{g}/{name}/{lane}"
                    rows.append(dict(label=label,geometry=g,room=path,room_sha256=digest,scale=float(scale),
                                     version=name,clipped_samples=clipped,reference=lane,mic_sha256=P.digest(mic),reference_sha256=P.digest(reference),
                                     dry_sha256=P.digest(wave),historical_control=historical and name=="original"))
                    yield label,mic.tobytes(),reference.tobytes(),None,None
    work_root=Path(".local/wakeword/c161-bella-marker-only-self")
    work_root.mkdir(parents=True,exist_ok=True)
    archive=OUT.with_suffix(".zip")
    with TemporaryDirectory(dir=work_root) as tmp:
        work=Path(tmp)
        guest=cue.overlap.board.run_guest.replay_cases(cue.GUEST,Path(".local/wakeword/c161-panel-middle-uart/build"),
                                                       P.QEMU,work,cases())
        counts=Counter()
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as z:
            for row,actual in zip(rows,guest["records"],strict=True):
                assert row["label"]==actual["label"]
                row.update(actual)
                if row["historical_control"]:
                    assert row["events"]["O"]==previous[row["reference"]]["opened"]
                counts[row["version"]+"."+row["reference"]]+=bool(row["events"]["O"])
                uart=work/"uart"/(row["input_sha256"]+".log")
                assert P.sha(uart)==row["uart_sha256"]
                z.write(uart,row["label"]+".log")
        assert len(geometry)==4 and len(rows)==24 and sum(r["historical_control"] for r in rows)==2
        P.save(work/OUT.name,dict(records=rows,opens=counts,geometry_count=len(geometry),
               archive_sha256=P.sha(work/archive.name),model=parent["model"],guest_sha256=guest["guest_sha256"],
               qemu_sha256=guest["qemu_sha256"],sources={p.name:P.sha(p) for p in (
                   Path(__file__),Path(cue.__file__),cue.overlap.OUT,cue.overlap.board.bella.OUT,HISTORY,
                   cue.overlap.board.bella.CUE,cue.overlap.peak.CUE)},
               policy="Only the panel plays a complete isolated generated address. No TX event or external source. "
                      "Matched mic and either the undelayed own-source electrical reference or a completely silent R. "
                      "Actual patched receiver on a fresh P4 QEMU boot for every distinct input. Four reused room "
                      "responses, same capture gain for all versions on each path, saturating PCM16 ADC. No physical board or FAR exposure."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)
    print(counts,flush=True)


if __name__=="__main__":
    main()
