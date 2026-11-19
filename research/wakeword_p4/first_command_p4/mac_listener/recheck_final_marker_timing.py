"""Repeat the recorded 16 ms overrun on the same microphone with both P4 listeners."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import check_final_marker_p4 as board

ch = board.ch
OUT = ch.RESULTS / "4090-c161-final-marker-timing.json"


def counters(result):
    line, = [line for line in result["uart_events"] if line.startswith("FINISH ")]
    _, _, _, blocks, total, peak, over, heap = map(int, line.split()[1:])
    return dict(blocks=blocks, mean_us=total / blocks, peak_us=peak, over_16ms=over, min_internal_bytes=heap)


def main():
    history = ch.read(board.OUT)
    original = [r for r in history["records"] if counters(r["board"])["over_16ms"]]
    targets = {(r["board"]["source"]["group"],r["board"]["source"]["scene"]) for r in original}
    assert targets
    rows = []
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        strict = ch.binding(work / "strict.so", coded="v5")
        prefix = ch.binding(work / "prefix.so", coded="aligned")
        candidate = board.host.binding(work)
        for meta, mode, local, pcm, ref, _, _ in board.grid_trials():
            if (meta["group"],meta["scene"]) not in targets or mode != "original" or local:
                continue
            assert not np.any(ref)
            for repeat in range(3):
                for name,detector,build in (("original",strict,board.phase.handoff.test.BUILD),("candidate",candidate,board.BUILD)):
                    expected, = [r for r in history["records"] if r["board"]["source"] == meta and r["mode"] == name and not r["local"]]
                    result = board.phase.handoff.probe.score(work,meta,pcm,ref,detector,len(rows),build=build,prefix=prefix,
                        qualify=board.phase.handoff.probe.qualify,wait_for_third=True,frame_activity=True,qualify_capture_frame=True)
                    assert all(result[k] == expected["board"][k] for k in ("mic_sha256","reference_sha256","fixed","c161","opened"))
                    rows.append(dict(repeat=repeat, mode=name, board=result, timing=counters(result)))
    assert len(rows) == 6 * len(targets)
    ch.save(OUT,dict(records=rows, original_overruns=[dict(index=r["index"], timing=counters(r["board"])) for r in original],
                     sources={p.name:ch.sha(p) for p in (board.OUT,Path(__file__))},
                     policy="Three interleaved repeats of both builds on the same external input. QEMU block times "
                     "and heap refer to this simulated listener under host scheduling."))


if __name__ == "__main__":
    main()
