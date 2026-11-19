"""Adapt the 18 keyword-state mixture distributions of the packed English Q8 model."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from check_c161_compact_lexical import WORK as LEXICAL
from check_c161_lexical_history import PACK, PLAN as SOURCES
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from train_c161_word_window_v1 import sources

STEM = "4090-c161-q8-keyword-map"
ROOT = Path(".local/wakeword/c161-q8-adapt")
FILES = LEXICAL / "q8/files"
WEIGHTS = RESULTS / f"{STEM}.sendump"
PRIOR = 128
FRAME = np.dtype([("tags", "<i4", (4,)), ("feature", "<f4", (3,13))])
GAUSSIAN_SHAPE = (42, 3, 128, 13)
MAP = LEXICAL / "compact/two-words.txt"
HELPER = Path(__file__).with_name("c161_align_keyword_states.c")


def build():
    src = Path(".local/wakeword/c161-pocket-reference/pocketsphinx-5.0.4")
    build = ROOT / "build"
    subprocess.run(["cmake","-S",str(src),"-B",str(build),"-G","Ninja","-DCMAKE_BUILD_TYPE=Release",
                    "-DBUILD_TESTING=OFF"],check=True)
    subprocess.run(["cmake","--build",str(build),"--target","pocketsphinx","--parallel","2"],check=True)
    subprocess.run(["cc","-O2","-DHAVE_CONFIG_H",*(f"-I{p}" for p in (src/"include", src/"src", build, build/"include")),
                    str(HELPER),str(build/"libpocketsphinx.a"),"-lm","-o",str(ROOT/"align")],check=True)


def main():
    ROOT.mkdir(exist_ok=True)
    assert "4090" in torch.cuda.get_device_name()
    torch.set_num_threads(1)
    for name, expected in read(PACK)["files"].items():
        assert sha(FILES/name) == expected
    model = FILES/"am"
    weights = (model/"sendump").read_bytes()
    size = 3*128*144
    old = np.frombuffer(weights[-size:],np.uint8).reshape(3,128,144)
    assert np.frombuffer(weights[-size-8:-size],"<u4").tolist() == [128,144]
    target = {}
    for line in MAP.read_text().splitlines():
        fields = line.split()
        if len(fields)==10 and fields[3] in ("b","i","e"):
            target[int(fields[5])] = [int(v) for v in fields[6:9]]
    assert sorted(v for states in target.values() for v in states) == list(range(126,144))
    source = {r["key"]:r for r in read(SOURCES)["training"]}
    rule = dict(sources_sha256=sha(SOURCES), parent_sha256=sha(PACK), pseudo_frames_per_state=PRIOR,
                seed=None, iterations=1, source="all distinct positive marked prefixes, no score selection",
                alignment="forced hey nicks; six phones, three states each; 0.5s leading and 0.3s trailing silence",
                adapt="posterior mixture counts plus prior; update only columns 126..143, keep CI phone loop",
                evaluation="original source proposals, original and expanded features, 1e-15, 1.5s history, up to 0.3s future",
                code_sha256={p.name:sha(p) for p in (HELPER,Path(__file__),MAP)})
    plan = RESULTS/f"{STEM}-plan.json"
    if plan.exists():
        assert read(plan) == rule
    else:
        save(plan,rule)
    build()
    stats, rows, seen = [], [], set()
    for src, pcm in sources():
        meta = source.pop(src["key"])
        assert digest(pcm) == meta["pcm_sha256"] and src["positive"] == meta["positive"]
        if not src["positive"]:
            continue
        assert src["end"] == meta["word_end"] and 0 < src["end"] <= len(pcm)
        prefix = pcm[:src["end"]]
        key = digest(prefix)
        row = dict(key=src["key"], voice=src["voice"], pcm_sha256=meta["pcm_sha256"],prefix_sha256=key,
                   end_sample=src["end"])
        rows.append(row)
        if key in seen:
            row["status"] = "duplicate"
            continue
        seen.add(key)
        trial = ROOT/"prefix.pcm"
        np.pad(prefix,(8000,4800)).astype("<i2").tofile(trial)
        result = subprocess.run([str(ROOT/"align"),str(model),str(FILES/"two.dict"),str(trial)],capture_output=True)
        if result.returncode == 2:
            row.update(status="alignment failed",error=result.stderr.decode()[-500:])
            continue
        result.check_returncode()
        frames = np.frombuffer(result.stdout,FRAME)
        tags = frames["tags"]
        if set(tags[:,1]) != set(target):
            row.update(status="incomplete phones",frames=len(tags))
            continue
        assert np.all((tags[:,2] >= 0) & (tags[:,2] < 3)) and np.isfinite(frames["feature"]).all()
        row.update(status="aligned",frames=len(tags),alignment_sha256=__import__("hashlib").sha256(result.stdout).hexdigest())
        stats.append(frames)
    assert not source and stats
    frames = np.concatenate(stats)
    log_unit = 1024*np.log(1.0001)
    raw = [np.frombuffer((model/n).read_bytes()[72:-4],"<f4").reshape(GAUSSIAN_SHAPE).copy() for n in ("means","variances")]
    means, variances = (torch.from_numpy(a).cuda().double() for a in raw)
    variances = variances.clamp_min(.0001)
    updated = old.copy()
    counts = {}
    for phone, states in target.items():
        for ordinal, state in enumerate(states):
            selected = (frames["tags"][:,1]==phone) & (frames["tags"][:,2]==ordinal)
            x = torch.from_numpy(frames["feature"][selected].copy()).cuda().double()
            prior = torch.softmax(-torch.tensor(old[:,:,state].copy(),device="cuda").double()*log_unit, dim=1)
            occup = torch.zeros_like(prior)
            for batch in x.split(256):
                diff = batch[:,:,None,:]-means[phone]
                densities = -.5*(diff.square()/variances[phone]+variances[phone].log()).sum(-1)
                occup += (densities+prior.log()).softmax(-1).sum(0)
            posterior = (occup+PRIOR*prior)/(len(x)+PRIOR)
            assert torch.allclose(posterior.sum(1),torch.ones_like(posterior[:,0]))
            # Upstream logmath shifts signed logs arithmetically before negation.
            q = torch.ceil(-posterior.log()/log_unit).clamp(0,159).byte()
            updated[:,:,state] = q.cpu().numpy()
            counts[state] = int(len(x))
    assert np.array_equal(updated[:,:,:126],old[:,:,:126]) and sum(counts.values())==len(frames)
    with WEIGHTS.open("xb") as f:
        f.write(weights[:-size]+updated.tobytes())
    folder = ROOT/"am"
    folder.mkdir(exist_ok=True)
    for name in model.iterdir():
        if name.name != "sendump":
            shutil.copyfile(name,folder/name.name)
    shutil.copyfile(WEIGHTS,folder/"sendump")
    save(RESULTS/f"{STEM}-fit.json",dict(plan_sha256=sha(plan),helper_sha256=sha(ROOT/"align"),
         device=torch.cuda.get_device_name(),torch_version=torch.__version__,records=rows,frames_by_state=counts,
         parent_sendump_sha256=sha(model/"sendump"),adapted_sendump_sha256=sha(WEIGHTS),bytes=WEIGHTS.stat().st_size,
         changed_weights=int(np.count_nonzero(updated!=old))))
    print("fit",len(rows),len(stats),len(frames),counts,flush=True)


if __name__ == "__main__":
    main()
