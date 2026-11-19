"""Localize a background notification to PCM, signed template correlations and P4 UART."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf

import check_final_marker_background as background
import check_final_marker_p4 as board
from diagnose_c161_v5_room_misses import correlations
from run_c161_first_command_v5 import transcripts

host, ch = board.host, board.ch
OUT = ch.RESULTS / "4090-c161-final-marker-background-event.json"


def best_chain_at(x, pos):
    c = correlations(x[:pos])
    # Anchor the last correlation at the reported match, then search its preceding
    # middle and first matching windows under the firmware's two timing constraints.
    last = pos // 4 - 1
    sign = np.sign(c[2, last])
    results = []
    for middle in range(last - 486, last - 473):
        for first in range(middle - 646, middle - 633):
            score = c[[0, 1, 2], [first, middle, last]]
            if all(sign * score >= [.3, .3, .2]):
                results.append((min(sign * score[:2]), [4*(p+1) for p in (first, middle, last)], score.tolist()))
    _, ends, corr = max(results)
    return dict(ends=ends, corr=corr, raw_rms_pcm16=[float(np.sqrt(np.mean(x[t-1280:t].astype(float)**2))) for t in ends])


def main():
    history = ch.read(background.OUT)
    events = [(k, r, e["t"]) for k, g in history["groups"].items() for r in g["records"] for e in r["candidate_fixed"]]
    small, identity = ch.load_reader()
    previous = ch.read(board.prior.prior.OUT)
    turbo_dir = board.prior.prior.audio.prior.context.prior.audit.TURBO
    assert identity == previous["small"] and all(ch.sha(turbo_dir/k) == h for k,h in previous["turbo"]["files"].items())
    turbo = WhisperModel(str(turbo_dir), device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
    rows = []
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        work = Path(tmp)
        strict, candidate = ch.binding(work / "strict.so", coded="v5"), host.binding(work)
        prefix = ch.binding(work / "prefix.so", coded="aligned")
        for cohort, row, t in events:
            old = ch.read(ch.RESULTS / f"4090-c161-history-gain-{cohort}-v1.json")
            s, = [r["source"] for r in old["records"] if r["phase"] == "background" and r["source"]["pcm_sha256"] == row["pcm_sha256"]]
            file = Path(s.get("wav_path") or s["path"])
            with sf.SoundFile(file) as stream:
                begin = (t - 4 * 16000) // 256 * 256
                stream.seek(begin)
                x = stream.read(8 * 16000, dtype="int16")
            relative = t-begin
            hit = board.detect(candidate, [x])
            assert [e["t"] for e in hit] == [relative]
            ref = np.zeros_like(x)
            reads = {k:transcripts(model, x) for k,model in (("small",small), ("turbo",turbo))}
            shape = best_chain_at(x, relative)
            r = dict(cohort=cohort, source=file.name, source_pcm_sha256=row["pcm_sha256"], offset=begin,
                     event_in_file=t, clip_pcm_sha256=ch.digest(x), relative_event=relative, samples=len(x),
                     chain=shape, context=reads, boards={})
            for i,(name,detector,build) in enumerate((("strict",strict,board.phase.handoff.test.BUILD),("candidate",candidate,board.BUILD))):
                result = board.phase.handoff.probe.score(work, dict(carrier="background", cue_at=0, command_at=len(x)),
                              x, ref, detector, i, build=build, prefix=prefix, qualify=board.phase.handoff.probe.qualify,
                              wait_for_third=True, frame_activity=True, qualify_capture_frame=True)
                logs.write(work/"serial.log", f"{name}-full-uart.log")
                r["boards"][name] = result
            rows.append(r)
    archive.with_suffix(".tmp").replace(archive)
    ch.save(OUT, dict(records=rows, sources={p.name:ch.sha(p) for p in (background.OUT, Path(__file__))},
                      small=identity, turbo=previous["turbo"], archive_sha256=ch.sha(archive),
                      policy="Eight-second clip around each fixed notification, aligned to the 256-sample receiver block. "
                      "Full-file hashes were checked during the continuous replay. Word timestamps are unprompted estimates. "
                      "Cold-start P4 clips verify the local trigger. Context does not establish a repeatable lexical collision."))


if __name__ == "__main__":
    main()
