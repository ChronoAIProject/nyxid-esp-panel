"""Bind the unmodified upstream decoder, acoustic files and 62 frozen proposal windows."""
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tarfile

import soundfile as sf

RESULTS = Path("research/wakeword_p4/results")
CACHE = Path(".local/wakeword/c161-pocket-reference")
WORK = Path(".local/wakeword/c161-lexical-p4")
SRC_DIGEST = "17d2971e998490def9554b6ba41886708422af5b4ae888723b1b077aab0fe8d4"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def main():
    archive = CACHE / "pocketsphinx-5.0.4.tar.gz"
    assert sha(archive) == SRC_DIGEST
    with tarfile.open(archive) as source:
        for member in source:
            name = Path(member.name)
            if name.parts[:1] != ("pocketsphinx-5.0.4",) or ".." in name.parts:
                raise ValueError(member.name)
            if member.isfile():
                dest = WORK / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as entry, dest.open("wb") as out:
                    shutil.copyfileobj(entry, out)
    flash = WORK / "files"
    acoustic = flash / "am"
    acoustic.mkdir(parents=True, exist_ok=True)
    prior = RESULTS / "4090-c161-midlevel-pocketsphinx-v1.json"
    bg = RESULTS / "4090-c161-midlevel-pocketsphinx-background-v1.json"
    run = read(prior)
    files = run["implementation"]["files"]
    model = CACHE / "site/pocketsphinx/model"
    for name, entry in files.items():
        src = model / name
        assert sha(src) == entry["sha256"]
        if src.suffix == ".dict":
            lines = [s for s in src.read_text().splitlines() if s.split()[0] in ("hey", "nicks")]
            assert set(lines) == {"hey HH EY", "nicks N IH K S"}
            (flash / "two.dict").write_text("\n".join(lines) + "\n")
        elif src.name != "README":
            shutil.copyfile(src, acoustic / src.name)
    source_maps = {}
    for cohort, name in (("forward", "4090-c161-signed-forward-v1.json"), ("new", "4090-c161-pocketsphinx-willow-v1.json")):
        for i, r in enumerate(read(RESULTS / name)["records"]):
            source_maps[cohort,str(i)] = r
    for cohort in ("melo", "primed", "jenny"):
        for r in read(RESULTS / f"4090-c161-history-gain-{cohort}-v1.json")["records"]:
            if r["phase"] == "background":
                source_maps[cohort,f"{cohort}_{r['index']}"] = r["source"]
    records = []
    with (flash / "cases.bin").open("wb") as handle:
        for r in run["records"] + read(bg)["records"]:
            identity = (r["cohort"], r["key"])
            if identity not in source_maps or not r["checked"]:
                continue
            src = source_maps[identity]
            path = Path(src.get("wav_path") or src["path"])
            assert sha(path) == (src.get("wav_sha256") or src["sha256"])
            assert r["pcm_sha256"] == src["pcm_sha256"]
            for c in r["checked"]:
                start, end, t = c["start"], c["end"], c["proposal"]
                pcm, rate = sf.read(path, start=start, stop=end, dtype="int16")
                assert rate == 16000 and pcm.shape == (end-start,)
                handle.write(struct.pack("<III", start, end, t))
                handle.write(pcm.astype("<i2",copy=False).tobytes())
                records.append(dict(cohort=r["cohort"], key=r["key"], source_pcm_sha256=r["pcm_sha256"],
                                    window_pcm_sha256=hashlib.sha256(memoryview(pcm)).hexdigest(), **c))
    assert len(records) == 62
    report = dict(source_archive_sha256=SRC_DIGEST, speech_report_sha256=sha(prior), background_report_sha256=sha(bg),
                  files={str(p.relative_to(flash)):sha(p) for p in sorted(flash.rglob("*")) if p.is_file()}, records=records)
    (WORK / "inputs.json").write_text(json.dumps(report,indent=2) + "\n")


if __name__ == "__main__":
    main()
