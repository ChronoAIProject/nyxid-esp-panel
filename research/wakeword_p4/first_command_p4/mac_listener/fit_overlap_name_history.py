"""Compare two tiny integer readouts at the frozen overlap proposer's current clock."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn import functional as F

import score_connected_overlap as score

ch = score.ch
STEM = "4090-c161-overlap-name-history"
FIT = ch.RESULTS / (STEM + "-fit.json")
PAST = 1920
SEED = 162018


def features(raw, t, five):
    codes = []
    for end in (t-PAST, t):
        # The front end includes its normal 81,600-sample priming prefix.
        patch = score.verifier.training.room.window(raw, end)
        inputs = np.rint(patch.clip(0, 666)*(255/666)).astype(np.uint8)
        codes.extend(five.invoke(inputs).tolist())
    return codes


def collect():
    models, ids = score.load_models()
    five, five_ids = score.verifier.models()
    native = score.fit.prior.letters.native_frontend()
    rows = []
    for src, pcm in score.verifier.training.sources():
        raw, bands, clock = score.base.native_inputs(native, pcm)
        codes = score.base.infer(models["overlap"], bands)
        ts = [t for t in score.times(score.decode(codes, clock)) if 0 < t <= len(pcm)]
        samples = []
        for t in ts:
            # Only train positive events close enough to the annotated name and
            # deliverable before the command. Other positive-file clocks are unused.
            block = (t+255)//256*256
            if src["positive"] and not max(src["start"], src["tail"]-4800) <= block < src["tail"]:
                continue
            samples.append(dict(t=t, codes=features(raw, t, five["trained"])))
        source_id = src["source_sha256"]
        fold = int(sha256(source_id.encode()).hexdigest()[:8], 16) % 5
        rows.append(dict(source=src, pcm_sha256=ch.digest(pcm), fold=fold, proposals=ts, examples=samples))
        if len(rows) % 500 == 0:
            print("training audio", len(rows), flush=True)
    return rows, ids["overlap"], five_ids["trained"]


def dot(codes, arm):
    x = np.asarray(codes, dtype=np.int64)[..., -len(arm["weights"]):]
    return x @ np.array(arm["weights"], dtype=np.int64)


def fit(rows, width, original_cutoff):
    train, calibration = [], []
    for r in rows:
        pool = calibration if r["fold"] == 0 else train
        pool.extend((r["source"]["positive"], e["codes"][-width:]) for e in r["examples"])
    y, x = zip(*train)
    y = np.asarray(y, dtype=bool)
    assert all(np.count_nonzero(y==v) >= 20 for v in (True, False))
    x = torch.tensor(x, device="cuda", dtype=torch.float32)/128
    pos, neg = x[y], x[~y]
    w = torch.zeros(width, device="cuda", requires_grad=True)
    bias = torch.zeros((), device="cuda", requires_grad=True)
    opt = torch.optim.Adam([w, bias], lr=.03)
    for _ in range(320):
        loss = .5 * (F.softplus(-(pos @ w + bias)).mean() + F.softplus(neg @ w + bias).mean())
        loss = loss + .005*w.square().sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # A constant intercept is absorbed into the integer threshold below.
    arm = dict(weights=(w.detach().cpu()*4096).round().int().tolist(),
               train_positive=len(pos), train_negative=len(neg), loss=loss.item())
    cy, cx = zip(*calibration)
    cy, cx = np.asarray(cy, bool), np.asarray(cx)
    scores = dot(cx, arm)
    allowed = np.count_nonzero(cx[~cy,-5] >= original_cutoff)
    negatives = np.sort(scores[~cy])[::-1]
    assert 0 <= allowed < len(negatives) and cy.any()
    cutoff = int(negatives[allowed])+1
    arm.update(cutoff=cutoff, calibration_positive=int(cy.sum()), calibration_negative=int((~cy).sum()),
               allowed_negative=int(allowed), accepted_negative=int(np.count_nonzero(scores[~cy]>=cutoff)),
               accepted_positive=int(np.count_nonzero(scores[cy]>=cutoff)),
               original_positive=int(np.count_nonzero(cx[cy,-5]>=original_cutoff)))
    return arm


def main():
    assert "4090" in torch.cuda.get_device_name()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rows, front, five = collect()
    reference = ch.read(score.OUT)
    assert reference["models"]["overlap"] == front and reference["five_sha256"] == five
    for_training = {r["source"]["source_sha256"] for r in rows if r["fold"] != 0}
    calibration = {r["source"]["source_sha256"] for r in rows if r["fold"] == 0}
    assert for_training.isdisjoint(calibration)
    arms = {k:fit(rows, n, reference["cutoff"]) for k,n in (("current",5),("history",10))}
    with TemporaryDirectory(dir=FIT.parent) as folder:
        target = Path(folder)/FIT.name
        ch.save(target, dict(records=rows, arms=arms, overlap_sha256=front, five_sha256=five,
                original_cutoff=reference["cutoff"], past_samples=PAST, torch=torch.__version__,
                gpu=torch.cuda.get_device_name(), seed=SEED, script_sha256=ch.sha(Path(__file__)),
                stream_script_sha256=ch.sha(Path(score.verifier.training.__file__)),
                policy="Training-only synthetic dry and room speech, admitted everyday phrases, music and "
                       "LibriSpeech train-clean. Actual frozen proposer events. Positive windows deliver "
                       "during the last 300 ms before the annotated name end. Four SHA-grouped folds for "
                       "full-batch class-balanced logistic regression, one training-source fold for threshold "
                       "selection. Quantize coefficients first, select the most permissive integer threshold "
                       "that does not exceed the original negative count on that fold. No evaluation fitting."))
        target.replace(FIT)
    print("fitted", arms, flush=True)


if __name__ == "__main__":
    main()
