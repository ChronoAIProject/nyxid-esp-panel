"""Replay all consumed Melo US/BR speech controls with the frozen C161 and the fitted proposal models."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

import replay_connected_alignment as study
from replay_c161_history_gain_cohorts_v1 import REFERENCE

ch, score, fit = study.ch, study.score, study.training
OUT = ch.RESULTS / (fit.STEM+"-melo.json")


def main():
    spec = ch.read(REFERENCE)["cohorts"]["melo"]
    source = Path(spec["path"])
    assert ch.sha(source) == spec["sha256"]
    nets, identities = {}, {}
    for arm in ("fixed","latent","refreshed"):
        stem = fit.STEM+"-"+arm
        path = ch.RESULTS/(stem+"-v1.tflite")
        identities[arm] = ch.sha(path)
        assert identities[arm] == ch.read(ch.RESULTS/(stem+"-export-v1.json"))["sha256"]
        net = score.REF_KERNEL(model_path=str(path))
        net.allocate_tensors()
        nets[arm] = net
    native = score.fit.prior.letters.native_frontend()
    five, ids = score.verifier.models()
    ref = ch.read(study.OUT)
    assert ref["five_sha256"] == ids["trained"]
    rows = []
    for r in ch.read(source)["records"]:
        if r["phase"] != "synthetic":
            continue
        s = r["source"]
        path = Path(s["path"])
        assert ch.sha(path) == s["sha256"] and ch.sha(Path(r["trace"]["path"])) == r["trace"]["sha256"]
        mic, rate = sf.read(path,dtype="int16")
        assert rate == 16000 and ch.digest(mic) == s["pcm_sha256"]
        raw, pooled, clock = score.base.native_inputs(native,mic)
        with np.load(r["trace"]["path"],allow_pickle=False) as trace:
            np.testing.assert_array_equal(clock,trace["relative_block_return"])
            original = fit.calibration.detect(trace["c161_uint8"],clock,5,1175)
        assert original == [e["relative_sample"] for e in r["events"]["c161"]]
        routes, checks = dict(c161=original), {}
        for arm, net in nets.items():
            hits = fit.calibration.detect(score.base.infer(net,pooled),clock,5,1175)
            for t in hits:
                if t not in checks:
                    window = score.verifier.training.room.window(raw,t//160*160)
                    x = np.rint(window.clip(0,666)*(255/666)).astype(np.uint8)
                    checks[t] = five["trained"].invoke(x).tolist()
            passed = [t for t in hits if checks[t][0] >= ref["cutoff"]]
            routes[arm] = dict(proposals=hits,confirmed=passed,combined=score.base.scoring.merged(original,passed))
        rows.append(dict(id=s["id"],voice=s["voice_id"],role=s["phrase_id"],carrier=s["carrier"],text=s["text"],
            pcm_sha256=s["pcm_sha256"], generated_end=s["generated_prefix_end_sample"],
            reverberant_end=s["full_rir_prefix_end_sample"],routes=routes,name_scores=checks))
    assert len(rows)==432 and len({r["id"] for r in rows})==432
    with TemporaryDirectory(dir=OUT.parent) as tmp:
        path = Path(tmp)/OUT.name
        ch.save(path,dict(records=rows,source_sha256=spec["sha256"],models=identities,c161=ref["c161"],
            five_sha256=ref["five_sha256"],cutoff=ref["cutoff"],script_sha256=ch.sha(Path(__file__)),
            policy="Full historic consumed Melo development speech, 3 acoustic variants per source. "
                   "Original primed C161 events reconstructed from archived output bytes and return clocks. "
                   "Speech score is any trigger in the file, matching the historical container metric."))
        path.replace(OUT)


if __name__=="__main__":
    main()
