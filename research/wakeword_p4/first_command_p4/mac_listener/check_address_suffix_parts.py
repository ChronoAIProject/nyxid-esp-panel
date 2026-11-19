"""Rescore original and counterfactual windows with both suffix-training exports."""
import ctypes
from pathlib import Path

import numpy as np

import readout_name_parts as parts
import train_address_suffix as train
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

base = train.replay


def main():
    old = read(parts.REPORT)
    context_file = RESULTS / f"{train.STEM}-context.json"
    exported = RESULTS / f"{train.STEM}-exports.json"
    spec = read(exported)
    assert spec["fit_sha256"] == sha(train.FIT) and old["context_sha256"] == sha(parts.CONTEXT)
    models = {}
    for name, s in spec["models"].items():
        path = RESULTS / s["file"]
        assert sha(path) == s["sha256"]
        models[name] = base.poll.Model(path)
    lib = base.BUILD / "frontend.so"
    assert sha(lib) == old["frontend_sha256"]
    native = ctypes.CDLL(str(lib.resolve())).ambient_history
    native.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    streams = {r["source"]["key"]:r for r in read(context_file)["records"]}
    records = []
    for meta, pcm, _, _, _ in base.sources("context"):
        key = meta["source"]["key"]
        row = streams[key]
        for trial in (r for r in old["records"] if r["stream"] == key):
            assert base.poll.digest(pcm) == row["pcm_sha256"] == trial["stream_sha256"]
            span = trial["span"]
            lo, hi = span["start"], span["end"]
            runs = {}
            for edit, prev in trial["runs"].items():
                stop = prev["through_sample"]
                signal = pcm[:stop]
                if edit != "original":
                    signal = parts.mute(signal, *trial["edits"][edit])
                assert base.poll.digest(signal) == prev["prefix_sha256"]
                raw, _, _ = base.native_inputs(native, signal)
                scores = {}
                for c in prev["checks"] + prev["fixed_clock"]:
                    t = c["end"]
                    if t in scores:
                        assert scores[t]["patch_sha256"] == c["patch_sha256"]
                        continue
                    patch = np.rint(train.room.window(raw, t).clip(0, 666) * (255. / 666)).astype(np.uint8)
                    assert base.poll.digest(patch) == c["patch_sha256"]
                    scores[t] = dict(patch_sha256=c["patch_sha256"], prior=c["codes"],
                                     codes={m:vm.invoke(patch).tolist() for m,vm in models.items()})
                dynamic = [c["end"] for c in prev["checks"]]
                routes = {}
                for m in models:
                    pre = [c["end"] for c in row["checks"] if c["end"] < lo and c["codes"][m][0] >= 0]
                    c161 = [t for t in row["c161"] if t < lo] + prev["c161"]
                    merged = base.scoring.merged(c161, pre + [t for t in dynamic if scores[t]["codes"][m][0] >= 0])
                    routes[m] = [t for t in merged if lo <= t < hi]
                    if edit == "original":
                        assert routes[m] == [t for t in row["routes"][m] if lo <= t < hi]
                runs[edit] = dict(scores=scores, live_ends=dynamic, routes=routes)
            records.append(dict(stream=key, span=span, runs=runs))
    assert len(records) == len(old["records"])
    save(RESULTS / f"{train.STEM}-parts.json", dict(records=records, edits_sha256=sha(parts.REPORT),
         context_sha256=sha(context_file), exports_sha256=sha(exported), script_sha256=sha(Path(__file__)),
         policy="Rebuild each complete original or independently edited prefix, check its SHA and each "
         "feature window SHA, evaluate both exported INT8 models and merge with the unchanged history "
         "and replayed C161. Only live proposals can notify; fixed original windows remain diagnostic."))


if __name__ == "__main__":
    main()
