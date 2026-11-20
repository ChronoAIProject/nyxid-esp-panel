"""Replay supplemental model decisions through the shipped fixed-marker receiver."""
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

import check_bella_daily_overlap as overlap
import check_c161_event_patch_rescue as rescue
import check_replay as receiver
import connected_nick_willow as connected

fit, base = rescue.fit, rescue.base
OUT = base.RESULTS / (fit.STEM + "-receiver.json")
BOARD = Path(".local/wakeword/c161-mac-board-marker/src")


def room_cases():
    oracle = {r["label"]: r for r in base.read(overlap.OUT)["records"]}
    for info, role, mic, ref, clips in overlap.mixtures():
        if role == "enhanced":
            continue
        key = f'{info["index"]:03}/{role}'
        old = oracle[key]
        assert old["mic_sha256"] == overlap.P.digest(mic) and old["clipped"] == clips
        assert old["reference_sha256"] == overlap.P.digest(ref)
        for field, value in info.items():
            assert old[field] == value
        block = overlap.board.run_guest.payload(mic.tobytes(), ref.tobytes())
        assert sha256(block).hexdigest() == old["input_sha256"]
        yield dict(group=role, label=key, cue_start=old["cue_start"], command_at=old["command_at"],
                   oracle=old["events"], uart_sha256=old["uart_sha256"]), mic, ref


def dry_cases():
    plan = base.read(connected.PLAN)
    assert base.sha(connected.PACK) == plan["audio_sha256"]
    with ZipFile(connected.PACK) as pack:
        for row in plan["records"]:
            if row["target_id"] == "fixed-v5":
                entry = dict(file=row["member"], sha256=row["sha256"], pcm_sha256=row["pcm_sha256"])
                mic = overlap.study.playback.read_wav(pack, entry)
                yield dict(group="dry_fixed", label=row["member"], cue_start=row["start"],
                           command_at=row["end"]), mic, np.zeros_like(mic)


def evaluate(native, exe, models, case, mic, ref):
    mic_hash, ref_hash = overlap.P.digest(mic), overlap.P.digest(ref)
    pad = (0, -len(mic) % 256)
    mic, ref = np.pad(mic, pad), np.pad(ref, pad)
    # The host frontend can emit a partial final output. Supply lookahead zeros, then
    # retain only calls that the board could have made on complete 480-sample blocks.
    bands, clock, codes, sums = fit.stream(native, np.pad(mic, (0, 480)))
    live = clock <= len(mic) // 480 * 480
    clock, codes, sums = clock[live], codes[live], sums[live]
    np.testing.assert_array_equal(clock[clock > 0], np.arange(480, len(mic) + 1, 480))
    proposals = np.flatnonzero((clock > 0) & (sums >= 960))
    scores = rescue.infer(bands[live], proposals, models)
    hits = rescue.combined(sums, clock, proposals, scores, models)
    signals = {}
    events = {}
    for arm, ticks in hits.items():
        signature = tuple(ticks)
        if signature not in signals:
            signals[signature] = receiver.run(exe, mic.tobytes(), ref.tobytes(), ticks, read_size=256)
        events[arm] = signals[signature]
    if "oracle" in case:
        assert events["baseline"] == case["oracle"], (case["label"], events["baseline"], case["oracle"])
    # Last read that can return strictly before the command.
    pre = (clock >= case["cue_start"]) & ((clock + 255) // 256 * 256 < case["command_at"])
    return dict(case, mic_sha256=mic_hash, reference_sha256=ref_hash, codes_sha256=sha256(codes.tobytes()).hexdigest(),
                peak_before_command=int(sums[pre].max()), proposal_clocks=clock[proposals].tolist(),
                proposal_sums=sums[proposals].tolist(), scores=scores, notifications=hits, events=events)


def main():
    historical = base.read(overlap.board.BASE)
    train = base.read(fit.FIT)
    assert base.sha(base.MODEL) == historical["model_sha256"] == train["model_sha256"] == base.MODEL_SHA
    assert base.read(overlap.OUT)["guest_sha256"] == historical["guest_sha256"]
    service = BOARD / "components/panel_services"
    local = (service / "panel_voice.c", service / "fixed_marker/board_marker.cpp")
    for path in local:
        assert base.sha(path) == historical["source_sha256"][path.name]
    models = {}
    for name, model in train["models"].items():
        path = base.RESULTS / model["weights"]
        assert base.sha(path) == model["sha256"]
        with np.load(path) as archive:
            models[name] = dict(archive), model["cut"]
    native = fit.native_frontend()
    with TemporaryDirectory(dir=fit.WORK) as temp:
        folder = Path(temp)
        exe = receiver.build_replay(BOARD.resolve(), folder)
        rows = []
        counts = defaultdict(lambda: defaultdict(Counter))
        for case, mic, ref in chain(room_cases(), dry_cases()):
            row = evaluate(native, exe, models, case, mic, ref)
            baseline = row["events"]["baseline"]["O"]
            for arm, result in row["events"].items():
                opened = result["O"]
                counts[row["group"]][arm].update(cases=1, opened=int(bool(opened)),
                    timely=int(bool(opened and row["cue_start"] <= opened[0] < row["command_at"])),
                    changed=int(opened != baseline), allowed_model=int(bool(result["M"])),
                    allowed_marker=int(bool(result["C"])))
            rows.append(row)
        assert Counter(r["group"] for r in rows) == {"original": 96, "absent": 96, "dry_fixed": 12}
        assert len({(r["group"], r["label"]) for r in rows}) == len(rows)
        sources = (Path(__file__), overlap.OUT, overlap.board.BASE, fit.FIT, connected.PLAN, connected.PACK,
                   Path(rescue.__file__), Path(receiver.__file__), receiver.HERE / "replay.c", *local,
                   *sorted((service / "fixed_marker").glob("*.h")), BOARD / "boards/lcd_x8/product_audio_layout.h")
        base.save(folder / OUT.name, dict(records=rows, summary=counts, model_sha256=base.MODEL_SHA,
                  sources={p.name:base.sha(p) for p in sources},
                  policy="Host-compiled board C feeder with physical reference, no TX, 256-sample reads and "
                         "complete 480-sample model notifications. Each room baseline is checked against the "
                         "archived P4 guest. Original-level and address-erased pairs use the identical command "
                         "and scene. The dry fixed calls use zero reference. All inputs are consumed development."))
        (folder / OUT.name).replace(OUT)
    print(dict(counts), flush=True)


if __name__ == "__main__":
    main()
