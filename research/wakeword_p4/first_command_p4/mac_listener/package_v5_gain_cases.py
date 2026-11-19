"""Package the five newly detected digital inputs and aligned playback references for Mac."""
import hashlib
import io
import json
from zipfile import ZipFile

import soundfile as sf

import check_v5_playback_gain as test
from package_context_playback import add, formatted

ch, grid = test.ch, test.grid
OUT = ch.RESULTS / "c161-v5-gain-five-pairs.zip"


def main():
    board = ch.read(test.OUT)
    cases = {}
    for trial in board["records"]:
        if trial["group"] == "gained":
            result = trial["board"]
            cases[trial["board"]["source"]["grid_index"],trial["name"],trial["local"]] = result
    assert len(cases) == 20
    original = grid.fit.original()[1]
    boosted, rate = sf.read(test.WAV, dtype="int16")
    assert rate == 16000 and ch.sha(test.WAV) == board["wav_sha256"]
    manifest = dict(rate=16000, format="mono PCM16 WAV", board_report_sha256=ch.sha(test.OUT), cases=[],
                    replay="Mic file on M. For external playback set R to digital zero. For the panel self "
                    "test inject the paired reference WAV on R starting at the same sample as M. These "
                    "files already include the simulated room/background. For physical speaker playback "
                    "use the separate dry cue files.")
    index = 0
    tmp = OUT.with_suffix(".tmp")
    with ZipFile(tmp, "w") as pack:
        for ids in (grid.fit.EVAL, test.repeat.source.EXTRA):
            for s, signals, _ in grid.scenes(ids, {"boosted_v5":boosted}):
                if (index,"v5",False) in cases:
                    row = dict(grid_index=index, source=s, routes={})
                    for name, cue in (("v5",original),("boosted_v5",boosted)):
                        external, local = [cases[index,name,flag] for flag in (False,True)]
                        x = signals[name]
                        reference = test.level.guard.refs(s,name,{name:cue})
                        assert ch.digest(x) == external["mic_sha256"] == local["mic_sha256"]
                        assert ch.digest(reference) == local["reference_sha256"]
                        assets = {}
                        for lane,pcm in (("mic",x),("reference",reference)):
                            path = f"{index:03}/{name}-{lane}.wav"
                            assets[lane] = test.audio.prior.put_audio(pack,path,pcm)
                        row["routes"][name] = dict(assets=assets,external_opened=external["opened"],
                                                   self_opened=local["opened"],c161=external["c161"],fixed=external["fixed"])
                    manifest["cases"].append(row)
                index += 1
        assert len(manifest["cases"]) == 5
        add(pack, "manifest.json", formatted(manifest))
    with ZipFile(tmp) as pack:
        assert pack.testzip() is None and len(pack.namelist()) == 21
        for row in json.loads(pack.read("manifest.json"))["cases"]:
            for route in row["routes"].values():
                for asset in route["assets"].values():
                    data = pack.read(asset["file"])
                    assert hashlib.sha256(data).hexdigest() == asset["sha256"]
                    pcm,rate = sf.read(io.BytesIO(data),dtype="int16")
                    assert rate == 16000 and len(pcm) == asset["samples"] and ch.digest(pcm) == asset["pcm_sha256"]
    tmp.replace(OUT)


if __name__ == "__main__":
    main()
