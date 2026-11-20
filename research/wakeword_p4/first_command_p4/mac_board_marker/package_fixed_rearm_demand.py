"""Export the six continuous mono inputs for matched external-speaker board playback."""
from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import fixed_rearm_demand as test

PACK = test.WORK / "hye-nyx-willow-six-indoor-recordings-v1.zip"
URL = "https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/" + PACK.name
RECEIPT = test.OUT.with_stem(test.OUT.stem + "-mac-download")
NOTES = """# Six indoor recordings for rev1 board playback

Play each complete 16 kHz mono WAV through an external speaker. Use the same speaker level, board position and listening configuration for all receivers. Each file begins with five minutes from one microphone in the stated environment. After that segment the same noise window appears under two calls. Each call asks to turn down the hall light. The manifest gives the start of each address and command in 16 kHz samples relative to the start of the WAV. Confirm the panel has returned to listening before counting a later call as eligible. Log the actual listening windows, input M/R, UART decision, received sentence, minimum internal heap and any dropped blocks.

## Audio credits

Joachim Thiemann, Nobutaka Ito and Emmanuel Vincent recorded the DEMAND backgrounds. We selected channel 1 at its original level, repeated its final window under two synthetic calls, and inserted a one-second gap between calls. The audio adaptations in this package are distributed under the authors' Creative Commons Attribution-ShareAlike 3.0 Unported audio license. The manifest identifies each original recording and each modified WAV. [Original DEMAND recordings and author license statement](https://doi.org/10.5281/zenodo.1227121) · [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)
"""


def package():
    plan = test.P.read(test.PLAN)
    assert len(plan["records"]) == 6
    tmp = PACK.with_suffix(".tmp")
    files = {}
    with ZipFile(tmp, "w") as z:
        def add(name, raw):
            z.writestr(ZipInfo(name), raw, compress_type=ZIP_DEFLATED)
            files[name] = dict(bytes=len(raw), sha256=sha256(raw).hexdigest())
        rows = []
        for row in plan["records"]:
            name = row["label"] + ".wav"
            raw = Path(row["path"]).read_bytes()
            assert sha256(raw).hexdigest() == row["sha256"]
            add(name, raw)
            rows.append(dict(file=name, samples=row["samples"], background_samples=row["background_samples"],
                             pcm_sha256=row["pcm_sha256"], source=row["source"], calls=row["calls"]))
        add("README.md", NOTES.encode())
        manifest = dict(records=rows, members=files.copy(), source_plan_sha256=test.P.sha(test.PLAN),
                        rate=16000, channels=1, encoding="PCM16", attribution=plan["attribution"])
        add("manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    tmp.replace(PACK)


def verify(download):
    archive_hash = test.P.sha(PACK)
    assert test.P.sha(download) == archive_hash
    with ZipFile(download) as z:
        manifest = json.loads(z.read("manifest.json"))
        assert manifest["source_plan_sha256"] == test.P.sha(test.PLAN)
        plan = {r["label"]+".wav":r for r in test.P.read(test.PLAN)["records"]}
        assert len(z.namelist()) == len(plan)+2 and set(z.namelist()) == plan.keys() | {"README.md", "manifest.json"}
        files = {}
        for name in z.namelist():
            raw = z.read(name)
            files[name] = dict(bytes=len(raw), sha256=sha256(raw).hexdigest())
            if name != "manifest.json":
                assert manifest["members"][name] == files[name]
            if name in plan:
                assert plan[name]["sha256"] == files[name]["sha256"]
            if name == "README.md":
                assert raw == NOTES.encode()
    test.P.save(RECEIPT, dict(url=URL, archive_sha256=archive_hash, bytes=download.stat().st_size, files=files,
                              plan_sha256=test.P.sha(test.PLAN)))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        package()
    else:
        verify(Path(sys.argv[1]))
