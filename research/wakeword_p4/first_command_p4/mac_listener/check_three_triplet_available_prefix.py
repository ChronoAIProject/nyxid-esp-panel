"""Compare two cue-start estimators on the published microphone and peer recordings."""
from collections import Counter, defaultdict
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import soundfile as sf

import score_third_triplet_player_level as score
import three_triplet_eight_voices as voices

anchor, trial, level, p = score.anchor, score.anchor.trial, score.level, score.P
locator = anchor.alignment_tool
OUT = level.OUT.with_name("4090-three-triplet-available-prefix.json")
PROTO = level.PROTOCOL.with_name("three_triplet_available_prefix_protocol.md")


def sources(work):
    first = p.read(trial.OUT)
    assert p.sha(trial.ARCHIVE) == first["archive_sha256"]
    initial = trial.cases(work, p.read(trial.chosen.REPLAY))
    for i, ((r, _), old, prior) in enumerate(zip(initial,first["records"],p.read(anchor.OUT)["records"],strict=True)):
        assert r["label"] == old["label"] == prior["label"]
        reuse = {s:dict(pcm_sha256=v["cropped_pcm_sha256"],text=v["cropped"]) for s,v in prior["readings"].items()}
        yield "first-four", i, r, old, trial.ARCHIVE, prior["timing"]["estimated"], reuse
    for name, wake in (("original",level.angles.OUT),("reduced",level.OUT)):
        doc = p.read(level.OUT.with_stem(level.OUT.stem+"-"+name+"-commands"))
        archive = level.WORK/doc["archive"]
        assert p.sha(archive) == doc["archive_sha256"]
        sends = {r["label"]:r for r in doc["records"]}
        for i,r in enumerate(p.read(wake)["records"]):
            if r["role"] == "three-triplets":
                old = sends[r["label"]]
                crop = old.get("crop", {})
                yield old["mode"], i, r, old, archive, crop.get("estimated_end"), crop.get("readings")


def main():
    base = p.read(trial.OUT)
    proc, model, qwen = score.scoring.prior.load()
    assert qwen == base["qwen"] and "4090" in qwen["device"]
    decoder = score.scoring.prior.decoder_api()
    assert decoder.opus_get_version_string().decode() == base["decoder"]
    readers = proc, model, score.scoring.prior.reader(base["turbo"]), decoder
    cue_path = trial.triple.wave(trial.chosen.NAME)
    cue, rate = sf.read(cue_path,dtype="int16")
    assert rate == 16000 and p.sha(cue_path)==p.read(trial.triple.OUT)["voices"][trial.chosen.NAME]
    rows, counts, archives = [], defaultdict(Counter), set()
    with TemporaryDirectory(dir=level.WORK) as tmp:
        work=Path(tmp)
        for group,i,r,old,archive,before,previous in sources(work):
            row = dict(group=group, label=r["label"], opened=old["opened"], command=r["command"], annotated=r["end"])
            stats = counts[group]
            stats["cases"] += 1
            if old["opened"] is not None:
                mic,sr = sf.read(r["path"],dtype="int16")
                assert sr==16000 and p.digest(mic)==r["pcm_sha256"]==old["uplink"]["pcm_sha256"]
                short,_ = locator.locate_end(mic,old["opened"],cue)
                assert short==before
                after=score.cropped_readings(r,i,old["uplink"],archive,readers,work,cue,
                                             locate=locator.locate_available_prefix,previous_readings=previous)
                row.update(mic_sha256=r["pcm_sha256"], short_end=short, available_end=after["estimated_end"],
                           short_error=short-r["end"],available_error=after["error_samples"],
                           available_projection=after["similarity"], readings=after["readings"])
                for schedule,pair in after["readings"].items():
                    stats["reused_streams"] += int(pair["pcm_sha256"]==previous[schedule]["pcm_sha256"])
                    for who in ("qwen","turbo"):
                        target=anchor.words(r["command"])
                        for method, text in (("short",previous[schedule]["text"][who]),("available",pair["text"][who])):
                            words=anchor.words(text)
                            key=f"{method}_{schedule}_{who}"
                            stats[key+"_suffix"] += (words[-len(target):]==target)
                            stats[key+"_whole"] += (words==target)
                archives.add(archive)
            rows.append(row)
        assert len(rows)==60 and sum(r["opened"] is not None for r in rows)==59
        observed={r["mic_sha256"] for r in rows if "mic_sha256" in r}
        extra=[]
        assert p.sha(voices.PACK)==p.read(voices.PLAN)["files"][str(voices.PACK)]
        with ZipFile(voices.PACK) as z:
            meta=json.loads(z.read("manifest.json"))["voices"]
            for r in p.read(voices.OUT)["records"]:
                src,sr=sf.read(BytesIO(z.read(meta[r["voice"]]["file"])),dtype="int16")
                mic,mr=sf.read(r["path"],dtype="int16")
                assert sr==mr==16000 and p.digest(src)==meta[r["voice"]]["pcm_sha256"] and p.digest(mic)==r["pcm_sha256"]
                short,_=locator.locate_end(mic,r["nyx_open"],src)
                end,value=locator.locate_available_prefix(mic,r["nyx_open"],src)
                extra.append(dict(label=r["label"],mic_sha256=r["pcm_sha256"],overlaps_bella=r["pcm_sha256"] in observed,
                                  short_error=short-r["end"],available_error=end-r["end"],projection=value))
        assert len(extra)==24
        deps=[Path(__file__).resolve().relative_to(Path.cwd()),anchor.SCRIPT,level.PROTOCOL,PROTO,voices.PACK,voices.OUT,
              anchor.OUT,trial.OUT,level.angles.OUT,level.OUT,cue_path,Path(locator.__file__).resolve().relative_to(Path.cwd()),
              Path(score.__file__).resolve().relative_to(Path.cwd()),
              *(level.OUT.with_stem(level.OUT.stem+"-"+n+"-commands") for n in ("original","reduced"))]
        deps.extend(sorted(archives))
        p.save(work/OUT.name,dict(records=rows,counts=counts,voices=extra,qwen=qwen,turbo=base["turbo"],
                                  sources={str(k):p.sha(k) for k in deps}))
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
