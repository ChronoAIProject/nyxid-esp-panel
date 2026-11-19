"""Compare fixed-point SpeexDSP PCM across P4 QEMU and host, then score the actual P4 output."""
import argparse
import re
import struct
import subprocess
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from pocketsphinx import Decoder

from check_c161_compact_lexical import WORK as LEXICAL
from check_c161_lexical_echo import OUT, QEMU, cases
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_speex_echo import binding, cancel
from check_c161_v5_command_ns import digest
from check_c161_v5_far_command_noise import normalized
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from run_c161_first_command_v5 import boot, transcripts
from score_c161_lexical_echo import infer

ARCHIVE = OUT/"speexdsp-1.2.1.tar.gz"
EXPECTED = "8c777343e4a6399569c72abc38a95b24db56882c83dbdb6c6424a5f4aeb54d3d"
BUILD = OUT/"build"
PROBE = RESULTS.parent/"speex_echo_p4/main"
STEM = "4090-c161-speex-p4"


def host_build(floating):
    assert sha(ARCHIVE) == EXPECTED
    src = OUT/"speexdsp-1.2.1"
    fft = ["smallft.c"] if floating else ["kiss_fft.c","kiss_fftr.c"]
    flags = ["-DFLOATING_POINT","-DUSE_SMALLFT"] if floating else ["-DFIXED_POINT","-DUSE_KISS_FFT"]
    units = [src/"libspeexdsp"/n for n in ["mdf.c","fftwrap.c",*fft]]
    dest = OUT/("speex-float.so" if floating else "speex-fixed.so")
    subprocess.run(["cc","-O2","-fwrapv",*flags,"-DVAR_ARRAYS","-DEXPORT=",
                    "-shared","-fPIC","-I",str(PROBE),"-I",str(src/"include"),*map(str,units),"-lm","-o",str(dest)],check=True)
    return binding(dest), sha(dest)


def parse(log, length):
    pcm, costs = bytearray(), []
    text = log.read_text()
    assert "SPEEX PASS" in text and "watchdog got triggered" not in text
    for at, us, data in re.findall(r"SPEEX_PCM (\d+) (\d+) ([0-9a-f]+)\n",text):
        block=bytes.fromhex(data)
        assert int(at)==len(pcm)//2 and len(block)==320
        pcm.extend(block)
        costs.append(int(us))
    assert len(pcm)==length*2
    match=re.search(r"SPEEX_HEAP before=(\d+) ready=(\d+) after=(\d+) stack=(\d+)",text)
    return np.frombuffer(pcm,"<i2"), costs, dict(zip(("before","ready","after","stack"),map(int,match.groups())))


def main(floating):
    api, so = host_build(floating)
    build = OUT/"build-float" if floating else BUILD
    stem = f"{STEM}-float" if floating else STEM
    reader, asr = load_reader()
    files = LEXICAL/"q8/files"
    manifest = read(RESULTS/"4090-c161-lexical-p4-q8-packing.json")
    assert all(sha(files/n)==h for n,h in manifest["files"].items())
    decoder=Decoder(hmm=str(files/"am"),dict=str(files/"two.dict"),keyphrase="hey nicks",kws_threshold=1e-15,
                    loglevel="ERROR",lm=None)
    cmn=decoder.get_cmn()
    model_spec=read(RESULTS/"4090-c161-midlevel-training-v1.json")["reports"][0]
    mid=RESULTS/model_spec["path"]
    assert sha(mid)==model_spec["sha256"]
    prior=RESULTS/"4090-c161-speex-echo.json"
    rows=[]
    archive=RESULTS/f"{stem}-uart.zip"
    local=OUT/("float-p4" if floating else "fixed-p4")
    local.mkdir(exist_ok=True)
    if archive.exists() or (RESULTS/f"{stem}.json").exists():
        raise FileExistsError(stem)
    with ZipFile(local/"uart.zip","w",ZIP_DEFLATED) as uart:
        for key,mic,ref,meta in cases():
            payload=struct.pack("<II",0x35434541,len(mic))+mic.tobytes()+ref.tobytes()
            log=local/f"{key}.log"
            boot(build,log,local/"flash.bin",b"SPEEX PASS",(2<<20,payload))
            pcm,costs,heap=parse(log,len(mic))
            host=cancel(api,mic,ref)
            delta=np.abs(host.astype(int)-pcm)
            scored=infer(pcm,mid,decoder,cmn)
            external=meta["role"] in ("external_only","external_over_reply","simultaneous_mentions")
            end=(meta["external_end"] if external else meta["self_end"])+160
            hits=[e for e in scored["events"]["both_proposals"] if meta["start"]<=e["sample"]<end]
            row=dict(key=key,source=meta,pcm_sha256=digest(pcm),host_sha256=digest(host),
                     mismatched_samples=int(np.count_nonzero(delta)),max_abs_diff=int(delta.max()),
                     uart_sha256=sha(log),cost_us=costs,internal_bytes=heap,detections=scored,hits=hits)
            if hits and meta["role"]!="reply_only":
                first=max(0,(hits[0]["sample"]+159)//160*160-32000)
                text=transcripts(reader,pcm[first:end])
                expected=normalized(meta["external_command"] if external else "turn on the kitchen lights.")
                row.update(first=first,end=end,transcript=text,exact_suffix=normalized(text["text"])[-len(expected):]==expected)
            rows.append(row)
            uart.write(log,log.name)
            print(key,row["mismatched_samples"],len(hits),max(costs),flush=True)
    assert len(rows)==24
    (local/"uart.zip").replace(archive)
    save(RESULTS/f"{stem}.json",dict(source_archive_sha256=EXPECTED,reference_sha256=sha(prior),
         host_library_sha256=so,qemu_sha256=sha(QEMU),elf_sha256=sha(build/"nyx_speex_echo.elf"),
         app_sha256=sha(build/"nyx_speex_echo.bin"),sdkconfig_sha256=sha(OUT/"sdkconfig"),
         uart_archive_sha256=sha(archive),asr=asr,records=rows))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--float",action="store_true")
    main(parser.parse_args().float)
