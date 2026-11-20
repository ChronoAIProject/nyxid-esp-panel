"""Replay actual Nyx and Willow listeners after a rate change on the external player alone."""
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path("research/wakeword_p4")
PLAN = ROOT / "results/4090-bella-external-player-rates-plan.json"
OUT = PLAN.with_name("4090-bella-external-player-rates.json")
WORK = Path(".local/wakeword/bella-external-player-rates")
PROTOCOL = ROOT / "experiments/bella_external_player_rate_protocol.md"
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())
RATES = (("nominal", Fraction(1)), ("slow100ppm", Fraction(9999,10000)), ("fast100ppm", Fraction(10001,10000)),
         ("slow1pct", Fraction(99,100)), ("fast1pct", Fraction(101,100)),
         ("slow5pct", Fraction(95,100)), ("fast5pct", Fraction(105,100)),
         ("44k1over48k", Fraction(441,480)), ("48kover44k1", Fraction(480,441)))


def prepare():
    import numpy as np
    from scipy.signal import resample_poly
    import soundfile as sf
    import bella_two_sentence_rooms as quiet
    import bella_address_gain as gain
    import check_bella_daily_overlap as daily
    import check_address_gain_wakes as previous

    p = daily.P
    control = p.read(quiet.PLAN)
    weak = p.read(gain.PLAN)
    assert p.sha(quiet.PACK) == control["playback_sha256"] and p.sha(gain.PACK) == weak["archive_sha256"]
    files = {name:weak["files"][name] for name in weak["runtime"].values()}
    for path in (SCRIPT, PROTOCOL, quiet.PLAN, quiet.PACK, gain.PLAN, gain.PACK, previous.OUT,
                 quiet.PLAN.with_name(quiet.STEM+".json"), daily.OUT, Path(daily.__file__)):
        files[str(path)] = p.sha(path)
    WORK.mkdir(exist_ok=True, parents=True)
    rooms, taps = daily.rooms()
    render_by_file = {case["file"]:render for case, _, render, _, _, _ in rooms[-6]}
    weak_orig = {r["index"]:r for r in p.read(daily.OUT)["records"] if r["role"] == "original"}
    records = []
    for is_quiet, source, archive in ((True, control, quiet.PACK), (False, weak, gain.PACK)):
        with ZipFile(archive) as src:
            for r in source["records"]:
                if is_quiet and not r["label"].startswith("00-"):
                    continue
                if not is_quiet and r["level"] not in ("plus2dB", "erased"):
                    continue
                if is_quiet:
                    scene = "quiet"
                    raw = src.read(r["label"]+".wav")
                    assert sha256(raw).hexdigest() == r["sha256"]
                    dry, sr = sf.read(BytesIO(raw), dtype="int16")
                    ref_path = None
                else:
                    scene = str(r["scene"])
                    raw = src.read(r["member"])
                    assert sha256(raw).hexdigest() == r["speaker_sha256"]
                    packed, sr = sf.read(BytesIO(raw), dtype="int32")
                    assert np.all(packed % 16384 == 0)
                    dry = packed // 16384
                    ref_path = Path(r["reference_path"])
                    files[str(ref_path)] = p.sha(ref_path)
                    ref, ref_rate = sf.read(ref_path, dtype="int16")
                    assert ref_rate == 16000 and p.digest(ref) == r["reference_sha256"]
                    render = render_by_file[weak_orig[r["scene"]]["file"]]
                assert sr == 16000
                for tag, speed in RATES:
                    if speed == 1:
                        played = dry
                    else:
                        played = np.rint(resample_poly(dry.astype(float), speed.denominator, speed.numerator))
                    if is_quiet:
                        clipped = int(np.count_nonzero((played < -32768) | (played > 32767)))
                        mic = played.clip(-32768,32767).astype("<i2")
                        ref_hash = p.digest(np.zeros_like(mic))
                    else:
                        # Apply the filter to the complete, time-warped source including its original tail.
                        rear = daily.board.bella.rear_filter(played, taps)
                        rear = np.pad(rear, (0, max(0, len(dry)-len(rear))))[:len(dry)]
                        mic, clipped = render(rear)
                        assert len(mic) == len(ref)
                        ref_hash = r["reference_sha256"]
                    if speed == 1:
                        assert p.digest(mic) == r["pcm_sha256"]
                    role = "erased" if r["role"] == "absent" else r["role"]
                    label = f"{scene}-{role}-{tag}"
                    path = WORK/(label+".wav")
                    sf.write(path, mic, 16000, subtype="PCM_16")
                    files[str(path)] = p.sha(path)
                    records.append(dict(label=label, scene=scene, role=role, rate=tag, ratio=str(speed),
                        source=r["label"], path=str(path), sha256=files[str(path)], pcm_sha256=p.digest(mic),
                        samples=len(mic), clipped=clipped, external_samples=len(played), ref_sha256=ref_hash,
                        reference_path=str(ref_path) if ref_path else None, command=r["command"],
                        start=None if r["start"] is None else round(r["start"]/speed), end=round(r["end"]/speed)))
    assert len(records) == len({r["label"] for r in records}) == 108
    doc = dict(records=records, files=files, runtime=weak["runtime"], threshold_requests=["DEFAULT"])
    if PLAN.exists():
        assert p.read(PLAN) == doc
    else:
        p.save(PLAN, doc)


def dual():
    import numpy as np
    import soundfile as sf
    import check_address_gain_wakes as previous
    fixed, room = previous.fixed, previous.room
    meta = room.read(fixed.REPORT)
    image = previous.fixed_image(meta, "dual")
    cache = WORK / "dual" / previous.digest(bytes.fromhex(meta["qemu_sha256"])+image[:2<<20])
    cache.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=WORK) as folder:
        temp = Path(folder)
        rows = []
        with ZipFile(temp/"dual.zip","w",ZIP_DEFLATED) as logs:
            for r in room.read(PLAN)["records"]:
                mic, sr = sf.read(r["path"], dtype="int16")
                assert sr == 16000 and previous.digest(mic.tobytes()) == r["pcm_sha256"]
                ref = np.zeros_like(mic)
                if r["reference_path"]:
                    ref, sr = sf.read(r["reference_path"], dtype="int16")
                assert sr == 16000 and previous.digest(ref.tobytes()) == r["ref_sha256"]
                block = fixed.run_guest.payload(mic.tobytes(), ref.tobytes())
                for attempt in range(2):
                    try:
                        epochs, log, faults = fixed.boot(cache, image, cache, block, 1)
                        break
                    except AssertionError:
                        if attempt:
                            raise
                members = {}
                for path in (log, *faults):
                    name = r["label"] + (".log" if path == log else "/" + path.name)
                    logs.write(path, name)
                    members[name] = previous.digest(path.read_bytes())
                rows.append(dict(label=r["label"], events=epochs[0]["events"], members=members,
                                 input_sha256=previous.digest(block)))
        room.save(temp/"dual.json",dict(records=rows, plan_sha256=previous.digest(PLAN.read_bytes()),
                  guest_sha256=meta["guest_sha256"]["dual"], model_sha256=meta["model_sha256"],
                  qemu_sha256=meta["qemu_sha256"], archive_sha256=previous.digest((temp/"dual.zip").read_bytes())))
        for name in ("dual.zip","dual.json"):
            (temp/name).replace(WORK/name)


def report():
    import check_address_gain_wakes as previous
    import bella_two_sentence_rooms as quiet
    room = previous.room
    plan, n, w = (room.read(p) for p in (PLAN, WORK/"dual.json", WORK/"willow.json"))
    assert n["plan_sha256"] == w["plan_sha256"] == previous.digest(PLAN.read_bytes())
    base = {r["label"]:(r["c161_marker"]["events"],r["willow"]["events"]) for r in room.read(previous.OUT)["records"]}
    for r in room.read(quiet.PLAN.with_name(quiet.STEM+".json"))["records"]:
        base[r["label"]] = r["dual"]["events"], r["willow"]["events"]
    counts = defaultdict(Counter)
    rows = []
    for r, left, right in zip(plan["records"], n["records"], w["records"], strict=True):
        assert r["label"] == left["label"] == right["label"] and r["pcm_sha256"] == right["pcm_sha256"]
        if r["rate"] == "nominal":
            assert (left["events"],right["events"]) == base[r["source"]]
        item = dict(label=r["label"], role=r["role"], scene=r["scene"], rate=r["rate"], end=r["end"],
                    clipped=r["clipped"], nyx=left, willow=right)
        for det, events in (("nyx",left["events"]["O"]),("willow",[t[1] for t in right["events"]])):
            opened = next(iter(events),None)
            c = counts["/".join(("quiet" if r["scene"]=="quiet" else "weak",r["rate"],det,r["role"]))]
            timely = opened is not None and r["start"] is not None and r["start"] <= opened < r["end"]
            c.update(cases=1, opens=opened is not None, timely=timely)
            item[det+"_lead_ms"] = (r["end"]-opened)/16 if opened is not None else None
        rows.append(item)
    members = {}
    with TemporaryDirectory(dir=WORK) as folder:
        temp = Path(folder)
        archive = OUT.with_suffix(".zip")
        with ZipFile(temp/archive.name,"w",ZIP_DEFLATED) as dest:
            for name, doc in (("dual",n),("willow",w)):
                path = WORK/(name+".zip")
                assert previous.digest(path.read_bytes()) == doc["archive_sha256"]
                with ZipFile(path) as src:
                    for row in doc["records"]:
                        for tag, digest in row["members"].items():
                            member = row["label"]+"/"+tag if name=="willow" else tag
                            raw = src.read(member)
                            assert previous.digest(raw)==digest
                            key = name+"/"+member
                            dest.writestr(key,raw)
                            members[key] = digest
        room.save(temp/OUT.name,dict(records=rows, counts=counts, files=members, plan_sha256=n["plan_sha256"],
                  qemu_sha256=n["qemu_sha256"], guest_sha256=n["guest_sha256"], model_sha256=n["model_sha256"],
                  archive_sha256=previous.digest((temp/archive.name).read_bytes())))
        (temp/archive.name).replace(archive)
        (temp/OUT.name).replace(OUT)


if __name__ == "__main__":
    action = sys.argv.pop(1)
    if action == "case":
        sys.path.insert(0, str(ROOT/"qemu_hey_reference_v2"))
        import run_willow_mac_matched_prime_v1 as official
        official.PLAN, official.__file__ = PLAN, str(SCRIPT)
        official.case()
    elif action == "willow":
        from check_address_gain_wakes import willow_cases
        willow_cases(PLAN, WORK, SCRIPT)
    else:
        {"prepare":prepare, "dual":dual, "report":report}[action]()
