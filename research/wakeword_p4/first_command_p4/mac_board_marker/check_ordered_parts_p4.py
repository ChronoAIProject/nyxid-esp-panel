"""Run both part heads from the P4 C161 frontend and compare every live integer score."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

from build_guest import link_guest
from check_replay import write_feeders
import check_fixed_sound_runtime as fixture
import run_guest
import replay_conversation_part_cutoffs as study

fit = study.fit
RESULT = fit.RESULTS / (fit.STEM + "-p4.json")
HERE = Path(__file__).resolve().parent


def build(work, models, python):
    board = fixture.PRIMARY
    runtime = board / "components/panel_wake_runtime"
    service = board / "components/panel_services"
    header = ["#include <cstdint>", "namespace ordered_parts {",
              "struct Weights {int8_t w1[1600]; int32_t b1[16]; int8_t w2[48]; int32_t b2[3]; "
              "unsigned minimum1,maximum1,minimum2,maximum2;};"]
    for n,(w,gaps) in models.items():
        arrays = ["{"+",".join(map(str,w[k].ravel()))+"}" for k in ("w1","b1","w2","b2")]
        header.append("constexpr Weights "+n+"={"+",".join(arrays+[str(i) for pair in gaps for i in pair])+"};")
    header.append("}")
    (work/"ordered_parts_weights.h").write_text("\n".join(header)+"\n")
    text = (runtime/"panel_wake_runtime.cpp").read_text()
    anchor = "    FrontendState frontend{};"
    check = "            if(features.size!=40 || feature_count>=3){failed=true;return -1;}"
    assert text.count(anchor)==text.count(check)==1
    text = text.replace(anchor,anchor+"\n    ordered_parts::Probe probe;")
    text = text.replace(check,check+"\n            probe.frame(features.values,feature_count);")
    instrumented = work/"part_runtime.cpp"
    instrumented.write_text('#include "ordered_parts_probe.h"\n'+text)
    write_feeders(service,work)
    guest = (HERE/"guest.c").read_text()
    anchor = "        if (result>0) printf("
    assert guest.count(anchor) == 1
    guest = guest.replace(anchor, "        report_ordered_parts();\n" + anchor)
    (work/"guest.c").write_text("extern void report_ordered_parts(void);\n" + guest)
    link_guest(fixture.SIM,work,"nyx_mac_listener.elf",[
        ("panel_voice_audio.c", work/"guest.c"), ("listener.cpp",service/"fixed_marker/board_marker.cpp"),
        ("panel_wake_runtime.cpp", instrumented)], [work,HERE,runtime,service],python.absolute())
    app = (work/"board.bin").read_bytes()
    assert app.count(fixture.MODEL.read_bytes())==1
    flash = bytearray([255])*(4<<20)
    cfg = fit.read(fixture.SIM/"flasher_args.json")
    for offset,member in cfg["flash_files"].items():
        data = app if member==cfg["app"]["file"] else (fixture.SIM/member).read_bytes()
        at = int(offset,16)
        assert at+len(data)<=2<<20
        flash[at:at+len(data)] = data
    return flash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idf-python",required=True,type=Path)
    args = parser.parse_args()
    work = (fit.WORK/"p4").resolve()
    work.mkdir(parents=True,exist_ok=True)
    native = fit.frontend.native_frontend()
    scores = fit.read(fit.FIT)
    speech = fit.RESULTS / (fit.STEM+"-conversation-speech.json")
    prior = {(r["group"],r["key"]):r for r in fit.read(speech)["records"]}
    models = {}
    for n in ("fixed","variable"):
        s = scores["models"][n]
        file = fit.RESULTS/s["file"]
        assert fit.sha(file)==s["sha256"]
        with np.load(file) as z:
            models[n] = (dict(z),s["gaps"])
    flash = build(work,models,args.idf_python)
    qemu_sha = fit.sha(fixture.QEMU)
    assert qemu_sha==fit.read(fixture.REPORT)["qemu_sha256"]
    cache = work/sha256(bytes.fromhex(qemu_sha)+flash[:2<<20]).hexdigest()
    cache.mkdir(exist_ok=True)
    rows = []
    with ZipFile(work/"uart.zip","w",ZIP_DEFLATED) as trace:
        for meta,src,c161,pcm in study.replay.corpus("speech",{}):
            group,key = meta["group"],meta["key"]
            h = prior[group,key]
            if group=="connected":
                if meta["role"]=="fixed-v5":
                    continue
            elif group!="six_cups" and not (group=="daily" and any(h["events"].values())):
                continue
            if pcm is None:
                pcm,rate = sf.read(src["path"],dtype="int16")
                assert rate==16000
            assert sha256(pcm.tobytes()).hexdigest()==h["pcm_sha256"]
            block=run_guest.payload(pcm.tobytes(),bytes(pcm.nbytes))
            clocked = ((len(pcm)+255)//256*256)//480*480
            mic = np.pad(pcm,(0,max(0,clocked-len(pcm))))[:clocked]
            bands,clock = study.replay.native_features(native,mic)
            # Complete feature groups only; the offline frontend also exports a padded last group.
            ticks = (np.arange(len(clock))+1)
            valid = (ticks>=170) & ((ticks+1)*480<=81600+clocked)
            expected = np.column_stack([ticks[valid]] + [fit.integer_scores(bands,w,g)[valid] for w,g in models.values()])
            raw_key = sha256(block).hexdigest()
            log = cache/(raw_key+".log")
            if log.exists() and b"BOARD PASS\r\n" not in log.read_bytes():
                log.rename(cache/(raw_key+".failed-"+fit.sha(log)+".log"))
            if not log.exists():
                flash[2<<20:(2<<20)+len(block)]=block
                (work/"flash.bin").write_bytes(flash)
                run_guest.boot(fixture.QEMU,work/"flash.bin",log)
            uart = log.read_text()
            actual = np.array([[int(a),int(b),int(c)] for a,b,c in re.findall(r"^PART (\d+) (-?\d+) (-?\d+)$",uart,re.M)])
            assert actual.shape==expected.shape and np.array_equal(actual[:,0],expected[:,0])
            assert "watchdog got triggered" not in uart and "Guru Meditation" not in uart
            model_hits = [int(v) for v in re.findall(r"^BOARD decision (\d+) \d+ \d+$",uart,re.M) if int(v)]
            label = group+"/"+key
            trace.write(log,label+".log")
            failed = {}
            for bad in cache.glob(raw_key+".failed-*.log"):
                member = "failed/"+label+"/"+bad.name
                trace.write(bad,member)
                failed[member]=fit.sha(bad)
            errors = np.abs(actual[:,1:]-expected[:,1:])
            compare = {}
            for j,n in enumerate(models):
                cutoff = fit.read(study.CAL)["cutoffs"][n]["value"]
                observed = study.replay.events(actual[:,j+1],clock[valid],cutoff)
                compare[n] = dict(mismatched_scores=int(np.count_nonzero(errors[:,j])),
                                  max_absolute_error=int(errors[:,j].max()),events=observed,host_events=h["events"][n])
            rows.append(dict(label=label,role=meta["role"],frames=len(actual),pcm_sha256=h["pcm_sha256"],input_sha256=raw_key,
                             uart_sha256=fit.sha(log),c161=model_hits,host_c161=c161,heads=compare,failed=failed))
            print(label,compare,flush=True)
    assert len(rows)==51
    archive = RESULT.with_suffix(".zip")
    with TemporaryDirectory(dir=work) as tmp:
        path = Path(tmp)/RESULT.name
        fit.save(path,dict(records=rows,guest_sha256=fit.sha(work/"board.bin"),qemu_sha256=qemu_sha,
              archive_sha256=fit.sha(work/"uart.zip"),sources={p.name:fit.sha(p) for p in (
                  fit.FIT,speech,study.CAL,HERE/"ordered_parts_probe.h",Path(__file__),HERE/"guest.c",fixture.MODEL)},
              scope="Both integer heads share the live raw C161 frontend on P4 QEMU. Scores from every complete "
                    "480-sample input group after the prime. P4 C161 detection still runs. Development digital replay."))
        (work/"uart.zip").replace(archive)
        path.replace(RESULT)


if __name__=="__main__":
    main()
