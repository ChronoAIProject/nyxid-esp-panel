"""Test the contribution of Bella's timed final /s/ in the issued dry cue."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile
import json

import numpy as np
import soundfile as sf

import check_bella_marker_only as cue

P = cue.P
OUT = P.RESULTS / "4090-c161-bella-final-s.json"


def make_tracks():
    parent = P.read(cue.OUT)["cue"]
    source = cue.gains.study.PACK
    with ZipFile(source) as pack:
        entry, = [r for r in json.loads(pack.read("manifest.json"))["records"]
                  if r["index"] == 0 and r["voice"] == "af_bella"]
        spec = entry["synthesis"]
        plain = cue.gains.study.playback.read_wav(pack, entry["tracks"]["plain"])[32000:entry["command_at"]]
    original, rate = sf.read(cue.overlap.board.bella.CUE, dtype="int16")
    stronger, rate2 = sf.read(cue.CUE, dtype="int16")
    assert rate == rate2 == 16000 and len(plain) == len(original) == len(stronger)
    assert P.sha(cue.CUE) == parent["wav_sha256"] and P.sha(cue.overlap.board.bella.CUE) == parent["original_sha256"]
    assert spec["phones"].startswith("hˈeɪ nˈɪks, ") and len(spec["duration"]) == 2 + len(spec["phones"])
    edges = np.cumsum(spec["duration"]) * 400
    ks = spec["phones"].index("ks")
    phones = {phone: [int(edges[pos]), int(edges[pos+1])] for pos, phone in ((ks, "k"), (ks+1, "s"))}
    assert edges[spec["phones"].index(", ")+2] == len(plain)
    speech = parent["original_prefix_scale"] * plain.astype(float)
    only = np.rint(speech).astype("<i2")
    mask = np.zeros(len(plain), dtype=bool)
    for lo, hi in parent["marker_samples"]:
        mask[lo:hi] = True
    assert np.array_equal(original[~mask], only[~mask]) and np.array_equal(stronger[~mask], only[~mask])
    tracks = {}
    gain = 10**(parent["code_boost_db"] / 20)
    for base_id, base in (("speech", only), ("original", original), ("boosted_markers", stronger)):
        tracks[base_id] = base
        for phone, weight, label in (("s", 0., "mute_s"), ("s", gain, "boost_s"), ("k", gain, "boost_k")):
            lo, hi = phones[phone]
            ramp = np.ones(hi-lo)
            taper = (1-np.cos(np.linspace(0, np.pi, 64)))/2
            ramp[:64], ramp[-64:] = taper, taper[::-1]
            delta = np.rint((weight-1) * ramp * speech[lo:hi]).astype(np.int32)
            changed = base.astype(np.int32)
            changed[lo:hi] += delta
            assert abs(changed).max() <= 32767
            tracks[f"{base_id}.{label}"] = changed.astype("<i2")
    return tracks, dict(source_pack_sha256=P.sha(source), track=entry["tracks"]["plain"], synthesis=spec,
                        cue_source_sha256=P.sha(cue.OUT), samples=len(plain), phone_intervals=phones,
                        marker_intervals=parent["marker_samples"], gain_db=parent["code_boost_db"],
                        speech_scale=parent["original_prefix_scale"], edited_edge_taper_samples=64)


def main():
    tracks, info = make_tracks()
    history = P.read(cue.OUT)
    proc, model, qid = cue.prior.load()
    assert qid == history["qwen"] and qid["device"].endswith("RTX 4090")
    turbo = cue.prior.reader(history["turbo"])
    rows = []
    for label, pcm in tracks.items():
        a = cue.prior.evaluate(proc, model, pcm, "Hey Nix")
        b = cue.prior.recognition(turbo, pcm, "Hey Nix")
        rows.append(dict(label=label, pcm_sha256=P.digest(pcm), qwen=a["text"], turbo=b["text"], words=b["words"]))
        print(label, a["text"], b["text"], flush=True)
    prior = P.read(P.RESULTS/"4090-c161-bella-marker-level-readings.json")["records"]
    for label, previous in (("original",prior[0]), ("boosted_markers",prior[2])):
        row, = [r for r in rows if r["label"] == label]
        assert all(row[k] == previous[k] for k in ("pcm_sha256","qwen","turbo"))
    assert P.sha(cue.GUEST) == history["guest_sha256"] and P.sha(P.QEMU) == history["qemu_sha256"]

    def cases():
        for label, pcm in tracks.items():
            mic = np.pad(pcm, (32000,32000))
            yield label, mic.tobytes(), bytes(mic.nbytes), None, None

    with TemporaryDirectory(dir=cue.WORK) as tmp:
        work = Path(tmp)
        guest = cue.overlap.board.run_guest.replay_cases(cue.GUEST,Path(".local/wakeword/c161-panel-middle-uart/build"),
                                                        P.QEMU,work,cases())
        archive = OUT.with_suffix(".zip")
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as z:
            for row, observed in zip(rows, guest["records"], strict=True):
                assert row["label"] == observed["label"]
                row.update(observed)
                log = work/"uart"/(row["input_sha256"]+".log")
                assert P.sha(log) == row["uart_sha256"]
                z.write(log,row["label"]+".log")
        P.save(work/OUT.name, dict(source=info,records=rows,qwen=qid,turbo=history["turbo"],model=history["model"],
               guest_sha256=guest["guest_sha256"],qemu_sha256=guest["qemu_sha256"],archive_sha256=P.sha(work/archive.name),
               script_sha256=P.sha(Path(__file__)),policy="One consumed synthetic address. Timing from the original "
               "Kokoro duration tokens. Edit only its pre-marker speech stem on the original sample clock. Apply "
               "the same PCM delta to all three bases. Read the complete isolated cue at its own level on CUDA. "
               "P4 digital microphone adds two seconds of silence on each side, with silent R and no TX event. "
               "M is a model decision, C is a code decision and O is the first allowed read end."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
