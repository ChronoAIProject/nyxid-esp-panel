"""Build the Q8 panel volume and regenerate host keyword clocks for its quantized values."""
import json
import shutil
import struct
import zlib
from pathlib import Path

from pocketsphinx import Decoder
from check_c161_pocketsphinx import search
from prepare import WORK, read, sha


def main():
    work = WORK / "q8"
    pack = read(work / "packing.json")
    volume = work / "volume"
    for name, entry in pack["packed"].items():
        source = work/"spiffs"/name
        assert sha(source) == entry["sha256"]
        target = volume/name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        if name in ("am/means", "am/variances"):
            data = zlib.compress(data, 9)
        target.write_bytes(data)
    (work/"cases").mkdir(exist_ok=True)
    shutil.copyfile(work/"files/cases.bin", work/"cases/cases.bin")
    case_file = work/"cases/cases.bin"
    inputs = read(WORK/"inputs.json")
    assert sha(case_file) == inputs["files"]["cases.bin"]
    host = work/"files"
    decoder = Decoder(hmm=str(host/"am"), dict=str(host/"two.dict"), keyphrase="hey nicks",
                      kws_threshold=1e-15, loglevel="ERROR", lm=None)
    import numpy as np
    cmn = decoder.get_cmn()
    with case_file.open("rb") as f:
        for row in inputs["records"]:
            start, end, t = struct.unpack("<III", f.read(12))
            assert (start, end, t) == (row["start"], row["end"], row["proposal"])
            pcm = np.frombuffer(f.read(2*(end-start)), "<i2")
            matches = search(decoder, cmn, pcm, start)
            row["matches"] = matches
            row["confirmed"] = min((max(t, m["returned"]) for m in matches if t-7200 <= m["end"] <= end), default=None)
        assert not f.read(1)
    inputs["files"] = {n:sha(volume/n) for n in pack["packed"]}
    inputs["files"]["cases.bin"] = sha(case_file)
    inputs.update(packing_sha256=sha(work/"packing.json"), full_inputs_sha256=sha(WORK/"inputs.json"),
                  model_bytes=sum((volume/n).stat().st_size for n in pack["packed"] if not n.startswith("music/")))
    (work/"inputs.json").write_text(json.dumps(inputs,indent=2)+"\n")


if __name__ == "__main__":
    main()
