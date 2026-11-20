"""Compare paired native opens to command onset and verify the reused erased controls."""
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import bella_player_gain_margin as trial
import check_address_gain_wakes as wakes

OUT = trial.PLAN.with_name("4090-bella-player-gain-margin.json")
P = wakes.room
sha = wakes.digest


def first_read(samples, record):
    if not samples:
        return dict(timely=False, sample=None, lead_ms=None)
    first = samples[0]
    return dict(timely=record["start"] <= first < record["end"],
                sample=first, lead_ms=(record["end"] - first) / 16)


def main():
    plan = P.read(trial.PLAN)
    baseline = P.read(trial.rates.OUT)
    base_plan = P.read(trial.rates.PLAN)
    old = {r["label"]: r for r in P.read(wakes.OUT)["records"]}
    erased = {r["label"]: r for r in baseline["records"] if r["role"] == "erased"}
    base_inputs = {r["label"]: r for r in base_plan["records"]}
    dual, willow = (P.read(trial.WORK / (name + ".json")) for name in ("dual", "willow"))
    plan_sha = sha(trial.PLAN.read_bytes())
    assert plan_sha == dual["plan_sha256"] == willow["plan_sha256"]
    assert dual["guest_sha256"] == baseline["guest_sha256"]
    assert dual["qemu_sha256"] == baseline["qemu_sha256"]
    assert dual["model_sha256"] == baseline["model_sha256"]
    rows, controls, files = [], {}, {}
    timely = defaultdict(list)
    zip_path = OUT.with_suffix(".zip")
    with TemporaryDirectory(dir=trial.WORK) as scratch:
        tmp = Path(scratch)
        with ZipFile(tmp / zip_path.name, "w", ZIP_DEFLATED) as dest:
            for name, doc in (("dual", dual), ("willow", willow)):
                path = trial.WORK / (name + ".zip")
                assert sha(path.read_bytes()) == doc["archive_sha256"]
                with ZipFile(path) as archive:
                    for row in doc["records"]:
                        for tag, checksum in row["members"].items():
                            member = row["label"] + "/" + tag if name == "willow" else tag
                            raw = archive.read(member)
                            assert sha(raw) == checksum
                            key = name + "/" + member
                            assert key not in files
                            dest.writestr(key, raw)
                            files[key] = checksum
        with ZipFile(trial.rates.OUT.with_suffix(".zip")) as prior:
            for r, n, w in zip(plan["records"], dual["records"], willow["records"], strict=True):
                assert r["label"] == n["label"] == w["label"] and r["pcm_sha256"] == w["pcm_sha256"]
                if r["rate"] == "nominal":
                    before = old[r["source"]]
                    assert (n["input_sha256"], n["events"], w["events"]) == (
                        before["c161_marker"]["input_sha256"], before["c161_marker"]["events"], before["willow"]["events"])
                label = r["erased_control"]
                if label not in controls:
                    e = erased[label]
                    er = base_inputs[label]
                    assert er["pcm_sha256"] == r["erased_pcm_sha256"] == e["willow"]["pcm_sha256"]
                    assert er["ref_sha256"] == r["ref_sha256"] and er["end"] == r["end"]
                    expected = {"dual/" + k: v for k, v in e["nyx"]["members"].items()}
                    expected.update({"willow/" + label + "/" + k: v for k, v in e["willow"]["members"].items()})
                    assert all(sha(prior.read(k)) == baseline["files"][k] == v for k, v in expected.items())
                    controls[label] = dict(pcm_sha256=er["pcm_sha256"], nyx=e["nyx"]["events"],
                                           willow=e["willow"]["events"], files=expected)
                readings = dict(nyx=first_read(n["events"]["O"], r), willow=first_read([e[1] for e in w["events"]], r))
                for which, read in readings.items():
                    if read["timely"]:
                        timely[(r["gain_db"], r["rate"], r["role"], which)].append(r["scene"])
                rows.append(dict(label=r["label"], scene=r["scene"], role=r["role"], gain_db=r["gain_db"],
                                 rate=r["rate"], start=r["start"], end=r["end"], reads=readings,
                                 erased=label, clipped=r["clipped"], dual=n, willow=w))
        assert len(rows) == 90 and len(controls) == 15
        gains = sorted({r["gain_db"] for r in rows})
        grid = {str(g): {tag: sorted(timely[g, tag, "nyx", "nyx"]) for tag, _ in trial.SPEEDS} for g in gains}
        qualifies = [g for g in gains if all(grid[str(g)][tag] == [77, 79, 90] for tag, _ in trial.SPEEDS[:3])]
        P.save(tmp / OUT.name, dict(records=rows, controls=controls, timely_nyx_scenes=grid,
               selected_gain_db=min(qualifies, default=None), model_sha256=dual["model_sha256"],
               guest_sha256=dual["guest_sha256"], qemu_sha256=dual["qemu_sha256"], plan_sha256=plan_sha,
               files=files, archive_sha256=sha((tmp / zip_path.name).read_bytes()),
               erased_report_sha256=sha(trial.rates.OUT.read_bytes()), erased_archive_sha256=baseline["archive_sha256"],
               report_script_sha256=sha(Path(__file__).read_bytes())))
        (tmp / zip_path.name).replace(zip_path)
        (tmp / OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
