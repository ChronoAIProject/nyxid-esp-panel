"""Measure agreement of two frozen C161 streaming clocks on consumed synthetic inputs."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import replay_c161_event_patch as replay
from replay_c161_vowel_name_pair_v1 import cases

base, fit = replay.base, replay.fit
OUT = base.RESULTS / "4090-c161-phase-agreement.json"
BLOCK = 480
OFFSET = BLOCK // 2
WINDOW = 5 * BLOCK


def stream(native, pcm, offset):
    # Extend only for the host frontend. Discard all returns beyond the original file.
    _, clock, codes, _ = fit.stream(native, np.pad(pcm, (offset, BLOCK)))
    clock = clock - offset
    keep = clock <= len(pcm)
    clock, codes = clock[keep], codes[keep]
    np.testing.assert_array_equal(clock[clock > 0], np.arange(BLOCK-offset, len(pcm)+1, BLOCK))
    events = replay.times(replay.decode(codes, clock))
    assert all(t > 0 for t in events)
    return events, sha256(codes.tobytes()).hexdigest()


def agreement(first, shifted):
    # Both crossings must be observable and their five-score windows must overlap.
    # Each lane's 67-step refractory is longer than twice the matching window.
    return [max(a, b) for a in first for b in shifted if abs(a-b) < WINDOW]


def compare(native, pcm):
    first, h0 = stream(native, pcm, 0)
    shifted, h1 = stream(native, pcm, OFFSET)
    paired = agreement(first, shifted)
    return dict(pcm_sha256=sha256(pcm.tobytes()).hexdigest(), samples=len(pcm),
                trace_sha256=[h0,h1], baseline=first, shifted=shifted, agreement=paired)


def counts(rows):
    totals = defaultdict(lambda: defaultdict(Counter))
    for row in rows:
        for arm in ("baseline", "shifted", "agreement"):
            hits = row[arm]
            stats = totals[row["cohort"] + "." + row["role"]][arm]
            stats.update(clips=1, opened=int(bool(hits)), events=len(hits))
            if "command_at" in row:
                stats.update(before_command=int(any(row["cue_start"] <= (t+255)//256*256 < row["command_at"]
                                                    for t in hits)))
    return totals


def main():
    assert base.sha(base.MODEL) == base.MODEL_SHA
    native = fit.native_frontend()
    historical = base.read(replay.OUT)["records"]
    six = base.read(replay.six.OUT)
    paired_oracle = {(r["speech_key"], r["lead"], r["phase"]):r for r in six["records"]}
    rows, daily = [], []
    for (meta, pcm, _), old in zip(replay.inputs(), historical, strict=True):
        row = dict(meta, **compare(native, pcm))
        assert row["pcm_sha256"] == old["pcm_sha256"]
        # Historical single-clip scoring also evaluated its padded last partial return.
        expected = [t for t in old["events"]["raw_1175"] if t % BLOCK == 0 and t <= len(pcm)]
        assert row["baseline"] == expected, (meta, row["baseline"], expected)
        rows.append(row)
        if meta["cohort"] == "six_cups":
            key, lead, phase = meta["id"].split("/")
            phase, lead = int(phase), int(lead)
            other_phase = (phase+OFFSET) % BLOCK
            # The second lane sees the same speech phase as this other P4 boot.
            # Translate its reported block ends to the current placement in physical time.
            oracle = paired_oracle[key,lead,other_phase]["events"]["M"]
            assert row["shifted"] == [t + phase - other_phase for t in oracle], meta
        if meta["cohort"] == "daily":
            for phase in range(0, BLOCK, 80):
                signal = np.pad(pcm, (24960+phase, 16000))
                daily.append(dict(meta, phase=phase, **compare(native, signal)))
    sessions = []
    for meta, pcm in cases("context"):
        if "timeline" in meta:
            result = compare(native, pcm)
            spans = [dict(group=s["group"], start=s["start"], end=s["end"], text=s["text"],
                          **{a:[t for t in result[a] if s["start"]<t<=s["end"]]
                             for a in ("baseline", "shifted", "agreement")})
                     for s in meta["timeline"] if s["kind"]=="clip"]
            sessions.append(dict(key=meta["key"], spans=spans, **result))
    assert (len(rows),len(daily),len(sessions)) == (1356,576,3)
    with TemporaryDirectory(dir=fit.WORK) as temp:
        path = Path(temp) / OUT.name
        base.save(path,dict(records=rows, summary=counts(rows), daily_offsets=daily,
                  daily_summary=counts(daily), sessions=sessions, model_sha256=base.MODEL_SHA,
                  sources={p.name:base.sha(p) for p in (Path(__file__), replay.OUT, replay.six.OUT)},
                  policy="Two independent cold streams, each primed with 81600 zeros, start their 480-sample "
                         "blocks 240 samples apart. Both use the frozen 1175 five-score sum, 67-step cooldown "
                         "and half-level rearm. Report agreement on the later observed crossing when their "
                         "five-score windows overlap. Complete blocks only. Confirm both lanes against the "
                         "36 existing P4 six-cups placements. Additional 96 daily clips at six phases are "
                         "correlated placements of consumed PCM. Model notifications only, no new FAR hours."))
        path.replace(OUT)


if __name__ == "__main__":
    main()
