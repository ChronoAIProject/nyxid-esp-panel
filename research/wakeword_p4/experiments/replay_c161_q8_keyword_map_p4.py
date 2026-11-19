"""Replace only sendump in the existing P4 model volume and verify 62 actual device windows."""
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from pocketsphinx import Decoder
from check_c161_pocketsphinx import search
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, sha
from train_c161_q8_keyword_map import FILES, LEXICAL, ROOT, STEM, WEIGHTS

sys.path.insert(0,str(RESULTS.parent/"pocketsphinx_p4_selftest"))
import run_qemu


def replay(root, stem, learned):
    sim = root/"p4"
    build = sim/"build-q8"
    original = LEXICAL/"build-q8"
    volume = sim/"q8/volume"
    manifest = read(RESULTS/"4090-c161-lexical-p4-q8-inputs.json")
    assert sha(learned) == sha(root/"am/sendump")
    for name, expected in manifest["files"].items():
        if name == "cases.bin":
            continue
        src = LEXICAL/"q8/volume"/name
        assert sha(src) == expected
        dest = volume/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(learned if name == "am/sendump" else src,dest)
        manifest["files"][name] = sha(dest)
    build.mkdir(parents=True,exist_ok=True)
    files = read(original/"flasher_args.json")["flash_files"]
    prior = read(RESULTS/"4090-c161-lexical-p4-q8.json")
    for name in (*files.values(),"flasher_args.json","nyx_lexical_probe.elf"):
        if name == "storage.bin":
            continue
        if name in prior["images"]:
            assert sha(original/name) == prior["images"][name]
        (build/name).parent.mkdir(exist_ok=True,parents=True)
        shutil.copyfile(original/name,build/name)
    shutil.copyfile(LEXICAL/"q8-sdkconfig",sim/"q8-sdkconfig")
    assert sha(build/"nyx_lexical_probe.elf") == prior["elf_sha256"]
    subprocess.run([sys.executable,".local/wakeword/esp-idf/components/spiffs/spiffsgen.py",
                    "0x200000",str(volume),str(build/"storage.bin"),"--page-size=256","--obj-name-len=32",
                    "--meta-len=4","--use-magic","--use-magic-len"],check=True)
    decoder = Decoder(hmm=str(root/"am"),dict=str(FILES/"two.dict"),keyphrase="hey nicks",
                      kws_threshold=1e-15,loglevel="ERROR",lm=None)
    cmn = decoder.get_cmn()
    casefile = LEXICAL/"q8/cases/cases.bin"
    assert sha(casefile) == manifest["files"]["cases.bin"]
    with casefile.open("rb") as f:
        for row in manifest["records"]:
            start,end,t = struct.unpack("<III",f.read(12))
            assert (start,end,t) == (row["start"],row["end"],row["proposal"])
            pcm = np.frombuffer(f.read(2*(end-start)),"<i2")
            assert digest(pcm) == row["window_pcm_sha256"]
            matches = search(decoder,cmn,pcm,start)
            row.update(matches=matches,confirmed=min((max(t,m["returned"]) for m in matches
                       if t-7200 <= m["end"] <= end),default=None))
        assert not f.read(1)
    manifest["parent_inputs_sha256"] = sha(RESULTS/"4090-c161-lexical-p4-q8-inputs.json")
    manifest["adapted_sendump_sha256"] = sha(learned)
    inp = sim/"q8/inputs.json"
    inp.write_text(json.dumps(manifest,indent=2)+"\n")
    run_qemu.WORK = sim
    run_qemu.main("q8")
    for src,suffix in ((build/"result.json",".json"), (build/"uart.log","-uart.log"), (inp,"-inputs.json")):
        shutil.copyfile(src,RESULTS/f"{stem}-p4{suffix}")


if __name__ == "__main__":
    replay(ROOT,STEM,WEIGHTS)
