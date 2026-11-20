"""Repeat the frozen 100 ms crop on the historical music and panel interruptions."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import crop_received_command as crop
import score_bella_plus2_commands as historical
import score_prospective_wake_crop as trial

P, scoring = trial.P, trial.scoring
OUT = historical.OUT.with_name("4090-bella-plus2-commands-wake-crop.json")


def main():
    source, reference = P.read(historical.OUT), P.read(trial.OUT)
    assert P.sha(historical.TRACE) == source["archive_sha256"]
    proc, model, identity = scoring.prior.load()
    decoder = scoring.prior.decoder_api()
    assert decoder.opus_get_version_string().decode() == source["decoder"] == reference["decoder"]
    assert identity == source["qwen"] == reference["qwen"] and "4090" in identity["device"]
    assert source["turbo"] == reference["turbo"] and reference["margin_samples"] == 1600
    readers = proc, model, scoring.prior.reader(source["turbo"]), decoder
    counts = defaultdict(Counter)
    records = []
    with TemporaryDirectory(dir=historical.trial.gain.WORK) as folder, ZipFile(historical.TRACE) as logs:
        work = Path(folder)
        for row in source["records"]:
            assert row["opened"] is not None and row["opened"] < row["command_at"]
            item = {k: row[k] for k in ("index", "label", "file", "command", "level", "opened", "selected")}
            item["conditions"] = {}
            # Three archived gain-selection captures retain their original UART trial number.
            measured = Counter(cases=1, opened=1)
            trial.score_opened(work, logs, row, readers, measured, item["conditions"],
                               expected_index=None if row["reused"] else row["index"])
            for group in ("all", str(row["level"]), "selected" if row["selected"] else "other"):
                counts[group].update(measured)
            records.append(item)
            if len(records) % 24 == 0:
                print(len(records), flush=True)
        assert len(records) == len({r["index"] for r in records}) == 96 and counts["selected"]["cases"] == 3
        paths = (Path(__file__), Path(trial.__file__), Path(crop.__file__), trial.OUT, historical.OUT)
        P.save(work / OUT.name, dict(records=records, counts=counts, qwen=identity, turbo=source["turbo"],
               decoder=source["decoder"], archive_sha256=source["archive_sha256"], margin_samples=1600,
               sources={p.name: P.sha(p) for p in paths}, scope="All 96 previously consumed +2 dB mixtures "
               "with music, panel playback and a competing voice. Full and clock-cropped views of each "
               "decoded send under both service schedules. Three prior gain-selection trials included."))
        (work / OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
