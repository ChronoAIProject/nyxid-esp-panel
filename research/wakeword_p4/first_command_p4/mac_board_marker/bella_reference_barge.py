"""Measure microphone recovery after a confirmed playback veto in the two-marker receiver."""
from pathlib import Path
import shutil
import subprocess
import sys

import bella_reference_rearm as test

P = test.P
WORK = Path(".local/wakeword/bella-reference-barge").resolve()
STEM = test.STEM.with_name("4090-bella-reference-barge")
PATCH = test.PATCH.with_name("microphone-rearm-after-veto.patch")
PARENT_WORK, PARENT_STEM, REF_PATCH = test.WORK, test.STEM, test.PATCH


def build():
    WORK.mkdir(exist_ok=True, parents=True)
    base = P.read(PARENT_WORK / "build.json")
    assert P.sha(REF_PATCH) == base["patch_sha256"]
    source = WORK / "src"
    shutil.copytree(PARENT_WORK/"src", source, dirs_exist_ok=True, symlinks=True,
                    ignore=shutil.ignore_patterns("panel_wake_runtime"))
    runtime = Path("components/panel_wake_runtime")
    if not (source/runtime).exists():
        (source/runtime).symlink_to(test.fixed.PRIMARY/runtime,target_is_directory=True)
    assert all(P.sha(source/test.SUB/k)==h for k,h in base["modified_headers"].items())
    subprocess.run(["git", "apply", "--directory="+str(source.relative_to(Path.cwd())), str(PATCH)], check=True)
    probe_source = test.PROBE.read_text()
    anchor = "        bool allowed = route.update_active"
    assert probe_source.count(anchor)==1
    probe = WORK/"probe.cpp"
    probe.write_text(probe_source.replace(anchor, "        if (raw && (!clear || state == State::confirmed)) microphone.rearm_after_playback_veto();\n"+anchor))
    subprocess.run(["c++","-std=c++17","-O2","-I",source/test.SUB,probe,"-o",WORK/"probe-candidate"],check=True)
    shutil.copy2(PARENT_WORK/"probe-original", WORK/"probe-original")
    image = test.fixed.build(source,WORK/"guest",False,test.IDF)
    binary = (WORK/"guest/board.bin").read_bytes()
    assert binary.count(test.fixed.MODEL.read_bytes())==1
    (WORK/"guest.flash").write_bytes(image)
    P.save(WORK/"build.json",dict(base,guest_sha256=test.fixed.digest(binary),patch_sha256=P.sha(PATCH),
           reference_patch_sha256=P.sha(REF_PATCH),flash_sha256=P.sha(WORK/"guest.flash"),
           probe_source_sha256=P.sha(probe),runner_sha256=P.sha(Path(__file__)),
           protocol_sha256=P.sha(test.early.ROOT/"experiments/bella_reference_barge_protocol.md"),
           modified_headers={p.name:P.sha(p) for p in (source/test.SUB).iterdir() if p.is_file()}))


def replay(part):
    test.STEM, test.WORK, test.PATCH = STEM, WORK, PATCH
    if part=="collisions":
        original = test.collisions
        old = {r["label"]:r for r in P.read(PARENT_STEM.with_name(PARENT_STEM.name+"-collisions.json"))["records"]}
        def reused():
            for key, row, mic, ref, _ in original():
                yield key, row, mic, ref, dict(input_sha256=old[key]["input_sha256"], events=old[key]["original"]["events"])
        test.collisions = reused
    test.replay(part)


if __name__=="__main__":
    stage = sys.argv[1]
    if stage == "build":
        build()
    else:
        assert stage in ("historical","collisions")
        replay(stage)
