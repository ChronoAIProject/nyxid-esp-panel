"""Run the frozen speech mask on server-side decoded Opus without changing the P4 wake or transport."""
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from faster_whisper import WhisperModel
import score_room_open_context as context
import train_speech_mask as fit

ch, gap, prior = context.ch, context.gap, context.prior
OUT = ch.RESULTS / (fit.STEM + "-uplink.json")


def main():
    old = ch.read(context.OUT)
    fixed_path, spoken_path = prior.replay.room.parent.OUT, prior.replay.OUT
    paths = (fixed_path, spoken_path, fixed_path.with_suffix(".zip"), spoken_path.with_suffix(".zip"))
    assert all(ch.sha(p) == old["source_sha256"][p.name] for p in paths)
    fixed, spoken = ch.read(fixed_path), ch.read(spoken_path)
    small, identity = ch.load_reader()
    info = old["turbo"]
    assert identity == old["small"] and all(ch.sha(prior.audit.TURBO/k)==v for k,v in info["files"].items())
    turbo = WhisperModel(str(prior.audit.TURBO), device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
    decoder, model = gap.uplink.decoder_api(), fit.load_model()
    results = {"fixed":{}, "spoken":{}}
    with ZipFile(paths[2]) as fixed_uart, ZipFile(paths[3]) as spoken_uart:
        for group, idx, i, opened, cap, key in context.cases(fixed,spoken):
            r = old["records"][group][str(i)]
            uart = fixed_uart if group=="fixed" else spoken_uart
            data = uart.read(key).decode().replace("\r\n","\n")
            packets = gap.uplink.packets(data,i,cap["emitted"]-cap["first"],0)
            x = gap.uplink.decode(decoder,packets)
            assert r["opened"]==opened and ch.digest(x)==r["full_sha256"]
            y = fit.enhance(model,x)
            # Supply one extra 10 ms zero hop to emit the tail still buffered by overlap-add.
            # The server retains all original samples through the last complete Opus packet.
            flushed = fit.enhance(model,np.pad(x,(0,fit.HOP)))
            assert np.max(np.abs(flushed[:len(y)].astype(int)-y.astype(int)))<=1
            aligned = flushed[fit.HOP:][:len(x)]
            row = dict(scene=idx, id=i, source_pcm_sha256=r["full_sha256"], aligned_sha256=ch.digest(aligned),
                       crop=r["cropped_start"], command=r["command"], opened=opened, room=r["room"], full={},cropped={})
            for view,signal in (("full",aligned),("cropped",aligned[r["cropped_start"]:])):
                for name,reader in zip(context.READERS,(small,turbo)):
                    scored = gap.uplink.recognition(reader,signal,r["command"])
                    row[view][name] = {k:scored[k] for k in ("text","complete_suffix","exact")}
            results[group][i]=row
    tally = defaultdict(lambda:defaultdict(Counter))
    def count(src,group,label,i):
        count = tally[src["room"]][label]
        count["trials"] += 1
        if i is None:
            return
        count["opens"] += 1
        original = old["records"][group][str(i)]
        row = results[group][i]
        for model_name in context.READERS:
            for view in ("full","cropped"):
                raw = original[view][model_name]
                masked = row[view][model_name]
                for route, score in (("raw",raw),("mask",masked)):
                    for metric in ("exact","complete_suffix"):
                        count[f"{model_name}_{view}_{route}_{metric}"] += score[metric]
                count[f"{model_name}_{view}_gained"] += masked["complete_suffix"] and not raw["complete_suffix"]
                count[f"{model_name}_{view}_lost"] += raw["complete_suffix"] and not masked["complete_suffix"]
    choices = ch.read(prior.replay.selection.FIT)["choices"]
    for a,b in zip(fixed["records"],spoken["records"],strict=True):
        count(a,"fixed","fixed_v5",a["routes"]["fixed_v5"]["capture_id"])
        for choice in choices.values():
            for mode in ("default","selected"):
                cue=b["cues"][str(choice[mode]["target"])]
                for name,r in cue["routes"].items():
                    count(a,"spoken",mode+"_"+name,r.get("capture"))
    assert len(results["fixed"])==16 and len(results["spoken"])==70
    ch.save(OUT,dict(records=results,summary=tally,small=identity,turbo=info,
        sources={p.name:ch.sha(p) for p in (context.OUT,fit.FIT,fit.WEIGHTS,Path(__file__))},
        policy="Apply the training-only model on the server after decoding the unmodified selected P4 Opus lane. "
        "Flush and account for the known 160-sample pipeline latency, then crop at the same existing clock. "
        "No labels used for inference, no change to detector. Cross-voice trials on exposed room/command cells."))


if __name__ == "__main__":
    main()
