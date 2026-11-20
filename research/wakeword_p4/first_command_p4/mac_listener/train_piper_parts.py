"""Fit a matched variable-spacing head with admitted Piper voices and command-only negatives."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F

import replay_conversation_part_cutoffs as original

fit = original.fit
STEM = "4090-c161-piper-parts"
REPORT = fit.RESULTS / (STEM + "-fit.json")
WEIGHTS = fit.RESULTS / (STEM + ".npz")
PLAN = fit.RESULTS.parent / "experiments/c161_piper_parts_protocol.md"
MATERIAL = fit.RESULTS / "4090-hey-broad-piper-material-plan-v1.json"
GAPS = fit.GAPS["variable"]


def piper_sources(identity):
    speakers = sorted(fit.read(MATERIAL)["training_ids"],
                      key=lambda i: sha256(f"piper-parts-v1/{i}".encode()).digest())[:40]
    settings = {speaker: ((0.9, 1.0, 1.1)[i % 3], "calibration" if i % 5 == 0 else "fit")
                for i, speaker in enumerate(speakers)}
    for bank in ("piper", "common-piper"):
        manifest = Path(f".local/wakeword/hey-broad-{bank}-native-v1/records.jsonl")
        prefix = f"4090-hey-broad-{bank}-"
        audit = fit.RESULTS / (prefix + "independent-audit-v1.json")
        generation = fit.RESULTS / (prefix + "generation-v1.json")
        receipt = fit.read(audit)
        assert not receipt["failures"] and receipt["independently_qualified"] == receipt["cases"]
        assert fit.sha(generation) == receipt["original_report_sha256"]
        assert fit.sha(manifest) == fit.read(generation)["records_sha256"]
        identity.update({str(p): fit.sha(p) for p in (audit, generation, manifest)})
        with manifest.open() as handle:
            for line in handle:
                source = json.loads(line)
                q = source["request"]
                speaker = q.get("speaker_id", q.get("conditioning_id"))
                if q["split"] != "training" or speaker not in settings:
                    continue
                scale, split = settings[speaker]
                if q.get("length_scale", q.get("scale")) != scale:
                    continue
                wav = source["files"]["wav"]
                assert source["quality_pass"] and all(source["checks"].values())
                assert fit.sha(Path(wav["path"])) == wav["sha256"]
                pcm, rate = sf.read(wav["path"], dtype="int16")
                assert rate == 16000 and pcm.ndim == 1 and len(pcm) == source["samples"]
                digest = sha256(pcm.tobytes()).hexdigest()
                assert digest == source["pcm_sha256"]
                positive = q.get("target") == "hey_nyx"
                boundary = source["generated_prefix_end_16k_sample"] if positive else None
                row = dict(key=f"{bank}/{source['order']}", voice=f"piper/{speaker}", split=split,
                           scale=scale, text=q["text"], positive=positive, end=boundary,
                           path=wav["path"], wav_sha256=wav["sha256"], source_pcm_sha256=digest,
                           pcm_sha256=digest, start=0, conversation=q["text"] in ("Hey.", "Hello."))
                yield row, pcm
                if positive and q["carrier"] != "bare":
                    start = boundary + 1920
                    tail = pcm[start:]
                    assert len(tail) >= 3200
                    yield dict(row, key=row["key"] + "/command", positive=False, end=None, start=start,
                               conversation=True, text=q["text"].split(", ", 1)[1],
                               pcm_sha256=sha256(tail.tobytes()).hexdigest()), tail


def dataset(identity):
    prior = {r["key"]: r for r in fit.read(fit.FIT)["records"]}
    original_cal = set(fit.read(original.CAL)["rows"])
    rows, features, previous_x, previous_masks = fit.collect()
    masks = [mask[:len(values)].cpu() for mask, values in zip(previous_masks, features, strict=True)]
    del previous_x, previous_masks
    for row in rows:
        previous = prior[row["key"]]
        assert previous["pcm_sha256"] == row["pcm_sha256"]
        row.update(bank="original", split=previous["split"], conversation=row["key"] in original_cal)
    native = fit.frontend.native_frontend()
    for row, pcm in piper_sources(identity):
        _, bands, clock = fit.frontend.base.native_inputs(native, np.pad(pcm, (24000, 16000)))
        keep = clock >= -32 * 480
        clock = clock[keep]
        mask = clock > 0
        if row["positive"]:
            end = 24000 + row["end"]
            mask &= (clock >= end - 4320) & (clock <= end)
        assert mask.sum() >= 2
        rows.append(dict(row, bank="piper"))
        features.append(torch.from_numpy(bands[keep]))
        masks.append(torch.from_numpy(mask))
    assert len(rows) == 915 + 800 and len({r["pcm_sha256"] for r in rows}) == len(rows)
    assert Counter(r["split"] for r in rows[915:]) == {"calibration": 160, "fit": 640}
    x = nn.utils.rnn.pad_sequence(features, batch_first=True).cuda().transpose(1, 2).contiguous().float()/255
    valid = nn.utils.rnn.pad_sequence(masks, batch_first=True).cuda()
    return rows, features, masks, x, valid


def main():
    device = torch.cuda.get_device_name()
    assert "4090" in device and fit.sha(fit.MODEL) == fit.MODEL_SHA
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    identity = {str(p): fit.sha(p) for p in (PLAN, MATERIAL, fit.FIT, original.CAL, Path(fit.__file__),
                Path(__file__), fit.frontend.base.BUILD/"frontend.so")}
    rows, features, masks, x, valid = dataset(identity)
    pools = [np.array([i for i, r in enumerate(rows) if r["positive"] == label and r["split"] == "fit"])
             for label in (True, False)]
    rng = np.random.default_rng(fit.SEED)
    torch.manual_seed(fit.SEED)
    model = fit.Parts().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
    history = []
    for step in range(700):
        ids = np.r_[rng.choice(pools[0], 32), rng.choice(pools[1], 64)]
        peaks = model(x[ids], GAPS).masked_fill(~valid[ids], -1e5).amax(1)
        loss = F.softplus(-peaks[:32]).mean() + F.softplus(peaks[32:]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (0, 99, 349, 699):
            history.append([step + 1, loss.item()])
            print(history[-1], flush=True)
    weights = fit.quantize(model)
    for row, bands, mask in zip(rows, features, masks, strict=True):
        row["peak"] = int(fit.integer_scores(bands.numpy(), weights, GAPS)[mask.numpy()].max())
    cal = [r for r in rows if r["split"] == "calibration"]
    cutoffs = {}
    for name, count in (("original", 82), ("expanded", 122)):
        neg = [r for r in cal if not r["positive"] and r["conversation"]
               and (name == "expanded" or r["bank"] == "original")]
        assert len(neg) == count
        cutoff = max(r["peak"] for r in neg) + 1
        cutoffs[name] = dict(value=cutoff, negatives=count, keys=[r["key"] for r in neg if r["peak"] == cutoff-1],
                             targets={b: sum(r["peak"] >= cutoff for r in cal if r["positive"] and r["bank"] == b)
                                      for b in ("original", "piper")})
    with TemporaryDirectory(dir=fit.WORK) as tmp:
        w = Path(tmp) / WEIGHTS.name
        report = Path(tmp) / REPORT.name
        np.savez_compressed(w, **weights)
        fit.save(report, dict(records=rows, sources=identity, cutoffs=cutoffs, weights_sha256=fit.sha(w),
                 bytes=sum(a.nbytes for a in weights.values()), c161=fit.MODEL_SHA, gaps=GAPS, gpu=device,
                 torch=torch.__version__, seed=fit.SEED, steps=700, loss=history))
        w.replace(WEIGHTS)
        report.replace(REPORT)


if __name__ == "__main__":
    main()
