"""Compare compact and full PocketSphinx segment clocks on all previously proposed windows."""
import argparse
from pathlib import Path

from pocketsphinx import Decoder

from check_c161_event_transcripts_v1 import choose, event_inputs
from check_c161_midlevel_pocketsphinx import FRESH, ROOMS, STEM, load_wav
from check_c161_pocketsphinx import confirmations
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

WORK = Path(".local/wakeword/c161-lexical-p4")
OUT = RESULTS / "4090-c161-lexical-p4-compact-transfer.json"


def sources():
    for cohort in ("context", "forward", "daily", "joe", "mac", "bot", "transfer"):
        for _, key, _, pcm in event_inputs(cohort):
            yield cohort, key, pcm
    for cohort, path in (("new", FRESH), ("room", ROOMS)):
        for i, row in enumerate(read(path)["records"]):
            yield cohort, str(i), load_wav(row)


def main(output):
    source = RESULTS / f"{STEM}-v1.json"
    rows = {(r["cohort"], r["key"]): r for r in read(source)["records"]}
    files = WORK / "compact/files"
    manifest = read(WORK / "compact/inputs.json")
    assert all(sha(files / n) == h for n, h in manifest["files"].items())
    decoder = Decoder(hmm=str(files/"am"), dict=str(files/"two.dict"), keyphrase="hey nicks",
                      kws_threshold=1e-15, loglevel="ERROR", lm=None)
    cmn = decoder.get_cmn()
    records = []
    for cohort, key, pcm in sources():
        full = rows.pop((cohort, key))
        assert digest(pcm) == full["pcm_sha256"]
        checked = confirmations(decoder, cmn, pcm, [c["proposal"] for c in full["checked"]])
        events = choose(full["baseline"], [c["confirmed"] for c in checked if c["confirmed"] is not None])
        for a, b in zip(checked, full["checked"], strict=True):
            assert all(a[k] == b[k] for k in ("start", "end", "proposal"))
        records.append(dict(cohort=cohort, key=key, windows=len(checked), exact=checked == full["checked"],
                            decisions_equal=events == full["events"],
                            differences=[dict(compact=a, full=b) for a,b in zip(checked, full["checked"]) if a != b]))
    assert not rows
    save(output, dict(full_reference_sha256=sha(source), compact_inputs_sha256=sha(WORK/"compact/inputs.json"),
                      records=records))
    print(len(records), "streams;", sum(r["windows"] for r in records), "windows;",
          sum(not r["exact"] for r in records), "streams with segment/score changes;",
          sum(not r["decisions_equal"] for r in records), "streams with notification changes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    main(parser.parse_args().output)
