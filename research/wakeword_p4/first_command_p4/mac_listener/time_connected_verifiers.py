"""Measure the two serial verification stages on the frozen direct-command and Nick inputs."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

import check_spelling_p4 as spell
import check_positive_readout_p4 as five_probe
import connected_nick_willow as connected
import replay_early_proposal_readout as study
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from score_paired_piper_willow import capture, host, uplink

STEM = "4090-c161-early-readout-p4"
CHAR = STEM + "-character"
WORK = Path(".local/wakeword/c161-early-readout-p4")
base, prior = study.base, study.replay.prior


def sources():
    frozen = read(RESULTS / (study.training.STEM + "-replay.json"))
    for group in ("daily", "paired"):
        for item, r in zip(study.inputs(group), frozen["records"][group], strict=True):
            i, pcm, c161, meta, _, old, _, _ = item
            assert i == r["index"] and base.poll.digest(pcm) == r["pcm_sha256"]
            accepted = {t:c["trained"] for t,c in r["checks"].items()}
            windows = {t+5760:old["checks"][str(t+5760)] for t in old["proposals"]["early"] if 0<t<=len(pcm)-5760}
            yield group,i,pcm,c161,meta,windows,accepted,r["events"]["calibrated"],old.get("captures",{})
    original = read(connected.PLAN)["records"]
    for src,r in zip(original, read(RESULTS / (connected.STEM + ".json"))["records"], strict=True):
        i = src["index"]
        pcm = host.piper.load_wav(src)
        assert i == r["index"] and base.poll.digest(pcm) == r["pcm_sha256"]
        windows = {int(t):c for t,c in r["five_spelling"].items()}
        accepted = {str(t):v["five"] for t,v in windows.items() if v["accepted"]}
        meta = dict(role=src["target_id"], cue_at=src["start"], command_at=src["end"], expected=src["carrier"])
        yield "connected",i,pcm,r["routes"]["c161"]["events"],meta,windows,accepted,r["routes"]["parent"]["events"],r["captures"]


def collect(work, char_build, source_streams=sources, refs=None):
    old, config = read(prior.CAL), read(spell.gap.FIT)
    char, native = prior.model(), prior.fit.native_frontend()
    vocab = read(prior.WAV2VEC / "vocab.json")
    assert spell.WORDS == old["target_spellings"] + old["competing_spellings"]
    models, _ = study.models()
    five = models["trained"]
    folder = work / "character"
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder/"build").exists():
        (folder/"build").symlink_to(char_build.resolve(), target_is_directory=True)
    if not (folder/"sdkconfig").exists():
        (folder/"sdkconfig").symlink_to((char_build.parent/"sdkconfig").resolve())
    rows, five_rows, streams = [], [], []
    with (folder/"blocks.bin").open("wb") as cf, (work/"five.bin").open("wb") as ff:
        for group,i,pcm,c161,_,windows,accepted,baseline,_ in source_streams():
            raw = base.native_inputs(native,pcm)[0] if windows else None
            start = len(rows)
            admitted = []
            for t,w in sorted(windows.items()):
                x = np.rint(prior.fit.room.window(raw,t).clip(0,666)*(255/666)).astype(np.uint8)
                chars = prior.outputs(char,x)
                assert base.poll.digest(chars) == w["output_sha256"]
                path = spell.gap.diagnostic.trace(chars,w["target"],vocab)
                assert path["normalized"] == w["target_value"]
                blank = path["interword_blank_frames"]
                passed = w["score"] >= config["threshold"] and blank <= config["max_gap"]
                assert passed == w["accepted"] == (str(t) in accepted)
                quantized = char.table[x]
                cf.write(quantized.tobytes()+chars.tobytes())
                idx = len(rows)
                rows.append(dict(end=t,input_sha256=base.poll.digest(quantized),output_sha256=w["output_sha256"],
                                 target=w["target"],rival=w["rival"],target_value=w["target_value"],
                                 rival_value=w["rival_value"],gap=blank,paths=w["score"]>=old["threshold"],capped=passed))
                if passed:
                    y = five.invoke(x)
                    assert y.tolist() == accepted[str(t)]
                    fx = five.table[x]
                    five_rows.append(dict(character_window=idx,host=y.tolist(),input_sha256=base.poll.digest(fx)))
                    ff.write(fx.tobytes()+y.astype(np.int8).tobytes())
                    if y[0] >= 34:
                        admitted.append(t)
            assert base.scoring.merged(c161,admitted) == baseline
            ix = list(range(start,len(rows)))
            paths = {k:base.scoring.merged(c161,[rows[j]["end"] for j in ix if rows[j][k]]) for k in ("paths","capped")}
            streams.append(dict(group=group,index=i,windows=ix,c161=c161,host=paths,baseline=baseline,pcm_sha256=base.poll.digest(pcm)))
    if refs is None:
        refs = [RESULTS/(connected.STEM+".json"),RESULTS/(study.training.STEM+"-replay.json")]
    refs = [*refs, prior.CAL, spell.gap.FIT]
    save(folder/"fixtures.json",dict(records=rows,streams=streams,inputs={p.name:sha(p) for p in refs},
         blocks_sha256=sha(folder/"blocks.bin"), tokens=[[vocab[c] for c in w] for w in spell.WORDS],
         separator=vocab["|"],max_gap=config["max_gap"],thresholds=dict(paths=old["threshold"],capped=config["threshold"])))
    save(work/"five.json",dict(records=five_rows,fixture_sha256=sha(work/"five.bin")))


def measure_five(work, stem=STEM):
    spec = read(work/"five.json")
    file = work/"five.bin"
    assert sha(file) == spec["fixture_sha256"]
    build = work/"build"
    images = read(build/"flasher_args.json")["flash_files"]
    model = read(RESULTS/(study.training.STEM+"-exports.json"))["models"]["trigger"]
    path = RESULTS/model["file"]
    assert sha(path) == model["sha256"] and sum((build/f).read_bytes().count(path.read_bytes()) for f in images.values()) == 1
    cfg = read(build/"config/sdkconfig.json")
    assert cfg["NN_ANSI_C"] and cfg["IDF_TARGET"] == "esp32p4"
    rows, batches = spec["records"], []
    archive = RESULTS/(stem+"-five-uart.zip")
    with file.open("rb") as stream, ZipFile(archive.with_suffix(".tmp"),"w", ZIP_DEFLATED) as uart:
        for start in range(0,len(rows),150):
            n = min(150,len(rows)-start)
            batch = n.to_bytes(4,"little")+stream.read(n*(144*40+5))
            log = work/f"five-{start}.log"
            five_probe.boot(build,log,work/"five-flash.bin",b"ONSET_DONE",(2<<20,batch))
            text = log.read_text()
            assert "ONSET_FC per_channel" in text and "ONSET_FAIL" not in text
            values = re.findall(r"^ONSET_ROW ([0-9 -]+)$",text,re.M)
            count,heap = map(int,re.search(r"ONSET_END count=(\d+) min_internal=(\d+)",text).groups())
            arena,init = map(int,re.search(r"ONSET_READY arena=(\d+) init_us=(\d+)",text).groups())
            assert count == n == len(values)
            for j, v in enumerate(values):
                i,*codes,us = map(int,v.split())
                assert i == j and us > 0 and codes == rows[start+j]["host"]
                rows[start+j].update(device=codes,invoke_us=us)
            batches.append(dict(start=start,count=n,arena=arena,init_us=init,min_internal_free=heap,
                                fixture_sha256=sha256(batch).hexdigest(),uart_sha256=sha(log)))
            uart.write(log,log.name)
        assert not stream.read(1)
    archive.with_suffix(".tmp").replace(archive)
    save(RESULTS/(stem+"-five.json"), dict(records=rows,batches=batches,model_sha256=sha(path),
         images={f:sha(build/f) for f in images.values()},archive_sha256=sha(archive),qemu_sha256=sha(five_probe.QEMU)))


def score():
    char_file, five_file = (RESULTS/(n+".json") for n in (CHAR,STEM+"-five"))
    char, five = read(char_file), read(five_file)
    assert all(w["mismatched_bytes"]==0 and not w["differing_fields"] for w in char["records"])
    lookup = {r["character_window"]:r for r in five["records"]}
    assert set(lookup) == {i for i,w in enumerate(char["records"]) if w["capped"]}
    reader, identity = host.piper.load_reader()
    decoder = uplink.decoder_api()
    records, summary = [], defaultdict(Counter)
    archive = RESULTS/(STEM+"-captures.zip")
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"),"w",ZIP_DEFLATED) as uart:
        work = Path(tmp)
        uplink.WORK = work
        for s, (group,i,pcm,c161,meta,_,_,baseline,known) in zip(char["streams"],sources(),strict=True):
            assert (group,i) == (s["group"],s["index"]) and s["baseline"] == baseline
            done, hits, trace = 0, [], []
            for j in s["windows"]:
                w = char["records"][j]
                cost = w["invoke_us"] + w["decode_us"]
                if j in lookup:
                    cost += lookup[j]["invoke_us"]
                done = max(done,w["end"]) + (cost*16+999)//1000
                if j in lookup and lookup[j]["device"][0]>=34:
                    hits.append(done)
                trace.append(dict(window=j, end=w["end"], service_us=cost, completed=done))
            timed = base.scoring.merged(c161,hits)
            row = dict(group=group,index=i,pcm_sha256=s["pcm_sha256"],baseline=baseline,timed=timed,queue=trace)
            for key,ts in (("host",baseline),("serial",timed)):
                total = summary[group+"."+meta.get("role","daily")+"."+key]
                total.update(dict(trials=1,hits=bool(ts),events=len(ts)))
                if group != "daily" and ts:
                    opened = (ts[0]+255)//256*256
                    timely = meta["cue_at"] <= opened < meta["command_at"]
                    data = known.get(str(opened))
                    if data is None:
                        extra = row.setdefault("extra",{})
                        if opened not in extra:
                            # Paired and connected indices overlap, so use the unique stream ordinal on UART.
                            extra[opened] = capture(reader,decoder,work,uart,len(records),key,pcm,opened,meta["expected"])
                        data = extra[opened]
                    result = data["lanes"]["selected"]["asr"]
                    total.update(dict(timely=timely,complete=result["complete_suffix"],timely_complete=timely and result["complete_suffix"]))
                    row[key+"_open"] = dict(opened=opened,timely=timely,complete=result["complete_suffix"])
            records.append(row)
        for f in sorted(work.glob("*-attempt*.gz")):
            uart.write(f,"retry/"+f.name)
    archive.with_suffix(".tmp").replace(archive)
    save(RESULTS/(STEM+".json"),dict(records=records,summary=summary,character_sha256=sha(char_file),five_sha256=sha(five_file),
         asr=identity,archive_sha256=sha(archive),script_sha256=sha(Path(__file__)),policy="One serial worker: charge measured P4 "
         "character inference plus decode for every window, and the measured five-class inference only after spelling passes. "
         "Merge independent C161 on its original clock, round up to 256 samples and encode newly opened captures with P4 "
         "Opus. Isolated guest timing excludes microphone, frontend and RTOS contention."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=("collect","character","five","score"))
    parser.add_argument("--character-build",type=Path,default=Path(".local/wakeword/c161-spelling-p4/build"))
    args = parser.parse_args()
    if args.action=="collect":
        collect(WORK,args.character_build)
    elif args.action=="character":
        with patch.object(spell,"STEM",CHAR):
            spell.run(WORK/"character",100)
    elif args.action=="five":
        measure_five(WORK)
    else:
        score()
