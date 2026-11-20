"""Replay the two frozen variable heads and causal 150 ms confirmation on digital P4."""
import argparse
from hashlib import sha256
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import check_ordered_parts_p4 as p4
import check_parts_confirmation as trial

fit, replay = trial.fit, trial.replay
OUT = fit.RESULTS / (trial.STEM + "-p4.json")


def main(python):
    work = (fit.WORK/"confirmation-p4").resolve()
    work.mkdir(exist_ok=True)
    speech = fit.RESULTS/(trial.STEM+"-speech.json")
    broad = fit.RESULTS/(trial.trial.STEM+"-speech.json")
    index = lambda path: {(r["group"],r["key"]):r for r in fit.read(path)["records"]}
    prior, extended = index(speech), index(broad)
    # The existing probe labels its two slots fixed and variable. Both weights here use variable gaps.
    profiles = trial.models()
    assert [p[2] for p in profiles] == [1638, -1560]
    models = {slot:(trial.evaluation.gpu_weights(path, digest)[0], trial.trial.GAPS)
              for slot, (path, digest, _) in zip(("fixed","variable"),profiles,strict=True)}
    flash = p4.build(work, models, python, confirmation=True)
    qemu = p4.fixture.QEMU
    assert fit.sha(qemu) == fit.read(p4.fixture.REPORT)["qemu_sha256"]
    cache = work/sha256(bytes.fromhex(fit.sha(qemu))+flash[:2<<20]).hexdigest()
    cache.mkdir(exist_ok=True)
    native = fit.frontend.native_frontend()
    rows = []
    with TemporaryDirectory(dir=work) as tmp:
        temp = Path(tmp)
        archive = temp / OUT.with_suffix(".zip").name
        with ZipFile(archive, "w", ZIP_DEFLATED) as logs:
            for r, src, c161, pcm in replay.corpus("speech",{}):
                group, key = r["group"],r["key"]
                before = extended[group,key]
                selected = ((group == "connected" and r["role"] in ("Hey Nyx","Hey Willow")) or
                            (group == "daily" and any(before["union"].values())) or group == "six_cups" or
                            (group == "joe" and r["role"] == "hey_nyx" and before["union"]["expanded"]))
                if not selected:
                    continue
                if pcm is None:
                    pcm,rate = sf.read(src.get("wav_path") or src["path"],dtype="int16")
                    assert rate == 16000
                assert sha256(pcm.tobytes()).hexdigest() == prior[group,key]["pcm_sha256"]
                block = p4.run_guest.payload(pcm.tobytes(),bytes(pcm.nbytes))
                clocked = ((len(pcm)+255)//256*256)//480*480
                mic = np.pad(pcm,(0,max(0,clocked-len(pcm))))[:clocked]
                bands,clock = replay.native_features(native,mic)
                ticks = np.arange(1,len(bands)+1)
                valid = (ticks >= 170) & ((ticks+1)*480 <= 81600+clocked)
                expected = np.column_stack([ticks[valid]] + [fit.integer_scores(bands,w,g)[valid] for w,g in models.values()])
                input_digest = sha256(block).hexdigest()
                log = cache/(input_digest+".log")
                if log.exists() and b"BOARD PASS\r\n" not in log.read_bytes():
                    log.rename(cache/(input_digest+".failed-"+fit.sha(log)+".log"))
                if not log.exists():
                    flash[2<<20:(2<<20)+len(block)] = block
                    image = work/"flash.bin"
                    image.write_bytes(flash)
                    p4.run_guest.boot(qemu,image,log)
                uart = log.read_text()
                actual = np.array([[int(v) for v in row] for row in re.findall(r"^PART (\d+) (-?\d+) (-?\d+)$",uart,re.M)])
                assert np.array_equal(actual, expected)
                confirms = list(map(int,re.findall(r"^CONFIRM (\d+)$",uart,re.M)))
                assert confirms == prior[group,key]["events"]["150ms"]
                board_c161 = [int(v) for v in re.findall(r"^BOARD decision (\d+) \d+ \d+$",uart,re.M) if int(v)]
                assert board_c161 == c161
                opened = [x["sample"] for x in replay.choose(board_c161,confirms)]
                assert opened == prior[group,key]["union"]["150ms"]
                assert "watchdog got triggered" not in uart and "Guru Meditation" not in uart
                label = group+"/"+key
                logs.write(log,label+".log")
                rows.append(dict(label=label,role=r["role"],input_sha256=input_digest,pcm_sha256=prior[group,key]["pcm_sha256"],
                                 frames=len(actual),c161=c161,confirmation=confirms,union=opened,uart_sha256=fit.sha(log)))
                print(label,len(actual),opened,flush=True)
        assert len(rows)==41
        out = temp/OUT.name
        fit.save(out, dict(records=rows, archive_sha256=fit.sha(archive), qemu_sha256=fit.sha(qemu), guest_sha256=fit.sha(work/"board.bin"),
                 sources={str(p):fit.sha(p) for p in (speech,broad,trial.CAL,fit.MODEL,Path(__file__),Path(p4.__file__),
                         p4.HERE/"ordered_parts_probe.h",p4.HERE/"parts_confirmation_probe.h",p4.HERE/"guest.c",
                         *(p for p,_,_ in profiles))}, scope="Every complete live 480-sample call on digital P4. "
                 "Both scores, native confirmation sample, and C161 events checked; union reconstructed from native returns."))
        archive.replace(OUT.with_suffix(".zip"))
        out.replace(OUT)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idf-python", required=True, type=Path)
    main(parser.parse_args().idf_python)
