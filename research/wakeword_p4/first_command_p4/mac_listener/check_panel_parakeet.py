"""Independently transcribe the actual Nyx and official Willow Opus captures with Parakeet TDT."""
import argparse
from collections import Counter, defaultdict
from hashlib import file_digest, sha1
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import transformers
from transformers import AutoModelForTDT, AutoProcessor

import check_panel_balanced_playback as balanced
from check_c161_v5_far_command_noise import normalized
import check_timed_final_p4 as timed

diag, ch = balanced.diag, balanced.ch
MODEL = Path(".local/wakeword/c161-parakeet-tdt-v3")
REV = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
WEIGHT_SHA = "3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11"
ROOM = ch.RESULTS / "4090-c161-fixed-spoken-willow-room-uplink.json"
MUSIC = ch.RESULTS / "4090-c161-willow-marker-music-p4.json"


def version():
    # The Hub tree pins small files by Git blob ID and the weight by LFS SHA256.
    from urllib.request import urlopen
    from huggingface_hub import hf_hub_download
    MODEL.mkdir(parents=True,exist_ok=True)
    index = MODEL/"tree.json"
    if not index.exists():
        with urlopen(f"https://huggingface.co/api/models/nvidia/parakeet-tdt-0.6b-v3/tree/{REV}",timeout=30) as r:
            data = r.read()
        index.write_bytes(data)
    tree = {x["path"]:x for x in ch.read(MODEL/"tree.json")}
    files = {}
    for name in ("model.safetensors","config.json","generation_config.json","processor_config.json",
                 "tokenizer.json","tokenizer_config.json"):
        path = MODEL/name
        if not path.is_file():
            hf_hub_download("nvidia/parakeet-tdt-0.6b-v3",name,revision=REV,local_dir=MODEL)
        size = path.stat().st_size
        assert size==tree[name]["size"]
        with path.open("rb") as f:
            digest = file_digest(f,"sha256").hexdigest()
        if name=="model.safetensors":
            assert digest == WEIGHT_SHA == tree[name]["lfs"]["oid"]
        else:
            assert sha1(f"blob {size}\0".encode()+path.read_bytes()).hexdigest()==tree[name]["oid"]
        files[name] = digest
    return dict(model="nvidia/parakeet-tdt-0.6b-v3",revision=REV,files=files,device=torch.cuda.get_device_name(),
                dtype="float32",torch=torch.__version__,transformers=transformers.__version__)


def panel_inputs(report, level_matched):
    for r in report["records"]:
        if level_matched:
            if r["mode"] != "balanced":
                continue
        elif (r["name"],r["role"]) != ("bca","both"):
            continue
        cap = r.get("uplink")
        s = r["source"]
        item = (r["index"],cap["samples"],cap["pcm_sha256"],f'{r["index"]:02}-opus.log') if cap else None
        yield dict(group=f'panel.{s["external_db"]}',case=r["index"],command=s["command"],turbo=r["turbo"],
                   small=r["small"],opened=r["opened"]), item


def ordinary_inputs(report):
    for r in report["records"]:
        b = r["board"]
        s = b["source"]
        if r["local"] or (s["group"] != "common" and not s.get("assigned")):
            continue
        cap = r.get("uplink")
        item = (r["index"],cap["samples"],cap["pcm_sha256"],f'{r["index"]:02}-opus.log') if cap else None
        yield dict(group="nyx.common" if s["group"]=="common" else "nyx.music",case=s["scene"],
                   command=s["carrier"],turbo=bool(cap and cap["turbo"]["complete_suffix"]),
                   small=bool(cap and cap["recognition"]["complete_suffix"]),opened=b["opened"]),item


def willow_inputs(report, music):
    for r in report["records"]:
        if music and r["role"]!="matched_rms":
            continue
        if music:
            v = r
            case, idx, command = r["grid_index"],r["index"],r["carrier"]
            turbo, small = r["turbo_complete"],r["small_complete"]
        else:
            v = r["routes"]["willow"]
            case,idx,command = r["scene"],v.get("id"),r["command"]
            turbo = bool("full" in v and v["full"]["turbo"]["complete_suffix"])
            small = bool("full" in v and v["full"]["small"]["complete_suffix"])
        cap = v.get("capture")
        item = (idx,cap["emitted"]-cap["first"],cap["lanes"]["selected"]["decoded_sha256"],f"opus/{idx:03}-willow.log") if cap else None
        yield dict(group="willow.music" if music else "willow.common",case=case,command=command,
                   opened=[] if v.get("opened") is None else [v["opened"]],turbo=turbo,small=small), item


def main(level_matched):
    identity = version()
    proc = AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    model = AutoModelForTDT.from_pretrained(MODEL,dtype=torch.float32,local_files_only=True).to("cuda").eval()
    decoder = diag.panel.grid.gap.uplink.decoder_api()
    files = [balanced.OUT] if level_matched else [diag.BOARD,timed.OUT,ROOM,MUSIC]
    summary, rows, sources = defaultdict(Counter), [], {}
    with torch.inference_mode():
        for path in files:
            report = ch.read(path)
            archive = path.with_suffix(".zip")
            assert ch.sha(archive) == report["archive_sha256"]
            sources.update({path.name:ch.sha(path),archive.name:report["archive_sha256"]})
            if path==timed.OUT:
                cases = ordinary_inputs(report)
            elif path in (ROOM,MUSIC):
                cases = willow_inputs(report,path==MUSIC)
            else:
                cases = panel_inputs(report,level_matched)
            with ZipFile(archive) as logs:
                for row, item in cases:
                    transcript = ""
                    if item:
                        i,samples,digest,logfile = item
                        uart = logs.read(logfile).decode().replace("\r\n","\n")
                        packets = diag.panel.grid.gap.uplink.packets(uart,i,samples,0)
                        x = diag.panel.grid.gap.uplink.decode(decoder, packets)
                        assert ch.digest(x) == digest
                        inputs = proc(x.astype(np.float32)/32768,sampling_rate=16000,return_tensors="pt").to(model.device)
                        generated = model.generate(**inputs,return_dict_in_generate=True)
                        transcript, = proc.batch_decode(generated.sequences,skip_special_tokens=True)
                        row.update(pcm_sha256=digest,packets=len(packets),samples=len(x))
                    want = normalized(row["command"])
                    ok = normalized(transcript)[-len(want):]==want
                    row.update(text=transcript,complete=ok)
                    stats = summary[row["group"]]
                    stats.update(trials=1,opens=int(bool(row["opened"])),parakeet=int(ok),
                                 turbo=int(row["turbo"]),small=int(row["small"]),gained_vs_turbo=int(ok and not row["turbo"]),
                                 lost_vs_turbo=int(row["turbo"] and not ok))
                    rows.append(row)
            print(path.name,len(rows),flush=True)
    assert len(rows) == (60 if level_matched else 114)
    output = "4090-c161-panel-balanced-parakeet.json" if level_matched else "4090-c161-panel-original-parakeet.json"
    ch.save(ch.RESULTS/output,dict(records=rows,summary=summary,model=identity,
        sources={**sources,Path(__file__).name:ch.sha(Path(__file__))},policy="Decode the full selected Opus "
        "stream at the observed OPEN, validate decoded SHA, use unprompted default TDT decoding on CUDA. "
        "Score the same normalized complete suffix used for Whisper, counting closed cases as incomplete. "
        "Use all panel trials and the prespecified 16 rooms and 11 music cases for original Nyx and official Willow."))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced",action="store_true")
    main(parser.parse_args().balanced)
