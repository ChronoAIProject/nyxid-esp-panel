"""Replay the Mac C161-plus-marker listener on the exact prospective room microphones."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import soundfile as sf

import check_address_gain_wakes as comparison
import prospective_bella_rooms as study

fixed, room = comparison.fixed, comparison.room
OUT = study.OUT.with_name(study.STEM+"-c161.json")
TRACE = OUT.with_suffix(".zip")


def main():
    prior = room.read(study.OUT)
    reference = room.read(fixed.REPORT)
    assert room.digest(fixed.MODEL.read_bytes())==reference["model_sha256"] and fixed.MODEL.stat().st_size==60952
    assert reference["qemu_sha256"]==prior["qemu_sha256"]
    flash = comparison.fixed_image(reference,"dual")
    cache = study.WORK/"c161"/room.digest(bytes.fromhex(reference["qemu_sha256"])+flash[:2<<20])
    cache.mkdir(parents=True,exist_ok=True)
    rows,counts=[],defaultdict(Counter)
    with TemporaryDirectory(dir=study.WORK) as folder:
        temp = Path(folder)
        with ZipFile(temp/TRACE.name,"w",ZIP_DEFLATED) as logs:
            for r in prior["records"]:
                mic,rate=sf.read(r["path"],dtype="int16")
                assert rate==16000 and room.digest(mic.tobytes())==r["pcm_sha256"]
                block=fixed.run_guest.payload(mic.tobytes(),bytes(mic.nbytes))
                assert room.digest(block)==r["nyx"]["input_sha256"]
                for attempt in range(2):
                    try:
                        observed,log,failures=fixed.boot(cache,flash,cache,block,1)
                        break
                    except AssertionError:
                        if attempt:
                            raise
                item,=observed
                label=r["label"]
                logs.write(log,label+".log")
                failed={}
                for path in failures:
                    name="failed/"+label+"/"+path.name
                    logs.write(path,name)
                    failed[name]=room.digest(path.read_bytes())
                e=item["events"]
                c=counts[f"{r['mode']}/{r['role']}/{r['level']}"]
                c.update(cases=1,neural=int(bool(e["M"])),marker=int(bool(e["C"])),opened=int(bool(e["O"])),
                         first_open_changed=int(e["O"]!=r["nyx"]["events"]["O"]))
                rows.append(dict(label=label,pcm_sha256=r["pcm_sha256"],input_sha256=r["nyx"]["input_sha256"],
                                 events=e,standalone=r["nyx"]["events"],uart_sha256=room.digest(log.read_bytes()),failed=failed))
                if len(rows)%20==0:
                    print(len(rows),flush=True)
        assert len(rows)==120
        room.save(temp/OUT.name,dict(records=rows,counts=counts,guest_sha256=reference["guest_sha256"]["dual"],
                  qemu_sha256=reference["qemu_sha256"],model_sha256=reference["model_sha256"],
                  sources={p.name:room.digest(p.read_bytes()) for p in (Path(__file__),study.OUT,fixed.REPORT,
                           Path(comparison.__file__),Path(fixed.__file__))},archive_sha256=room.digest((temp/TRACE.name).read_bytes()),
                  scope="Board M denotes a C161 event admitted by the receiver; C is a fixed cue event. "
                        "O is the first service request. Both lanes have identical digital M/R to the standalone replay."))
        (temp/TRACE.name).replace(TRACE)
        (temp/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
