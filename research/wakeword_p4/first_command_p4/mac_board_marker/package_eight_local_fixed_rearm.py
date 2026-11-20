"""Package and verify the rev1 public standalone application for the issued sounds."""
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from build_fixed_sound_rev1 import FIXED, verify
from build_rev1_dev import PATCH, REFERENCE_BARGE, check_sources
import eight_local_fixed_rearm as test

P, WORK = test.P, test.WORK
PACK = WORK / "hye-nyx-fixed-sound-rearm-p4-rev1-public-v1.zip"
URL = "https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/" + PACK.name
RECEIPT = test.OUT.with_stem(test.OUT.stem + "-rev1-download")


def check_trials():
    report = P.read(test.OUT)
    assert P.sha(test.OUT.with_suffix(".zip")) == report["archive_sha256"]
    primary = test.prior.indexed(test.prior.OUT)
    assert len(report["records"]) == len(primary) == 100
    targets, selfplay, daily = [], [], []
    for row in report["records"]:
        label = row["label"]
        base = primary.pop(label)
        assert row["input_sha256"] == base["input_sha256"]
        for epoch in row["epochs"]:
            assert epoch["events"]["C"] == base["c161"][0]["events"]["C"] and not epoch["events"]["M"]
        event = row["epochs"][0]["events"]
        if label.startswith("077/") or (label.startswith("079/") and base["role"] == "new"):
            targets.append(bool(event["O"]) and base["start"] <= event["O"][0] < base["end"])
        if label.startswith("panel/") and base["reference"] == "synchronous":
            selfplay.append(event["O"])
        if label.startswith("daily/"):
            daily.append(event["O"])
    assert not primary and len(targets) == 40 and all(targets)
    assert len(selfplay) == 4 and not any(selfplay) and len(daily) == 2 and not any(daily)


def build_pack():
    check_trials()
    board, build = (WORK.resolve() / name for name in ("src", "app"))
    check_sources(board, extra_patches=(FIXED, *REFERENCE_BARGE))
    metadata = verify(board, build, None)
    desc = P.read(build / "project_description.json")
    app = (build / desc["app_bin"]).read_bytes()
    name = f"fixed-nyx-rearm-rev1-{metadata['app_sha256'][:8]}.bin"
    patches = []
    for path in (PATCH, FIXED, *REFERENCE_BARGE):
        repo_path = path.resolve().relative_to(Path.cwd())
        commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", repo_path], text=True).strip()
        assert subprocess.check_output(["git", "show", f"{commit}:{repo_path}"]) == path.read_bytes()
        patches.append(dict(file=path.name, sha256=P.sha(path), source_commit=commit))
    manifest = dict(file=name, **metadata, patches=patches, simulation_report_sha256=P.sha(test.OUT))
    temp = PACK.with_suffix(".tmp")
    with ZipFile(temp, "w") as z:
        z.writestr(ZipInfo(name), app, compress_type=ZIP_DEFLATED)
        z.writestr(ZipInfo("manifest.json"), json.dumps(manifest, indent=2) + "\n", compress_type=ZIP_DEFLATED)
    temp.replace(PACK)


def downloaded(path):
    check_trials()
    assert path.read_bytes() == PACK.read_bytes()
    with ZipFile(path) as z:
        manifest = json.loads(z.read("manifest.json"))
        name = manifest["file"]
        assert z.namelist() == [name, "manifest.json"] and not z.testzip()
        files = {member: dict(bytes=z.getinfo(member).file_size, sha256=test.fixed.digest(z.read(member)))
                 for member in z.namelist()}
        assert files[name]["sha256"] == manifest["app_sha256"] and files[name]["bytes"] == manifest["app_bytes"]
        assert manifest["simulation_report_sha256"] == P.sha(test.OUT)
    P.save(RECEIPT, dict(url=URL, archive_sha256=P.sha(path), bytes=path.stat().st_size,
                         members=files, manifest=manifest))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        build_pack()
    else:
        downloaded(Path(sys.argv[1]))
