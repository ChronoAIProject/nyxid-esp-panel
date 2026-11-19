"""Count the final-marker candidate's own events on previously consumed long recordings."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import soundfile as sf

import check_final_marker_confirmation as host
from replay_c161_pcm_cues_v1 import detect, joined

ch = host.ch
OUT = ch.RESULTS / "4090-c161-final-marker-background.json"


def main(*, output=OUT, make_matcher=host.binding, grid_report=host.OUT, extra_sources=()):
    groups, sources = {}, {}
    with TemporaryDirectory() as folder:
        detector = make_matcher(Path(folder))
        for cohort in ("melo", "primed", "jenny"):
            path = ch.RESULTS / f"4090-c161-history-gain-{cohort}-v1.json"
            rows = []
            stats = Counter()
            for r in ch.read(path)["records"]:
                if r["phase"] != "background":
                    continue
                s = r["source"]
                wav = Path(s.get("wav_path") or s["path"])
                digest = sha256()
                with sf.SoundFile(wav) as stream:
                    assert (stream.samplerate, stream.channels, stream.subtype, len(stream)) == (16000, 1, "PCM_16", s["samples"])
                    events = detect(detector, stream.blocks(blocksize=96000, dtype="int16"), digest)
                assert digest.hexdigest() == s["pcm_sha256"]
                combined = joined(r["c161"], events)
                rows.append(dict(source=wav.name, samples=s["samples"], pcm_sha256=digest.hexdigest(),
                                 candidate_fixed=events, c161=r["c161"], combined=combined))
                stats.update(files=1, samples=s["samples"], fixed=len(events), c161=len(r["c161"]), combined=len(combined))
            groups[cohort] = dict(records=rows, summary=stats, hours=stats["samples"]/57600000)
            sources[path.name] = ch.sha(path)
            print(cohort, stats, flush=True)
    ch.save(output, dict(groups=groups, sources={**sources, **{p.name:ch.sha(p) for p in (grid_report, *extra_sources)},
                     Path(__file__).name:ch.sha(Path(__file__))}, policy="Previously consumed background recordings. "
                     "Stream from the start of each file, verify complete PCM hashes, preserve state across blocks. "
                     "Include previously frozen C161 events in the combined count. Report each exposure separately."))


if __name__ == "__main__":
    main()
