"""Apply the previously frozen v5 command crop to the new continuous command recordings."""
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import connected_nick_willow as connected
import score_coded_piper_command_start as earlier
from fit_c161_history_gain_v1 import RESULTS, read, save, sha


def main():
    report = RESULTS / (connected.STEM + ".json")
    archive = RESULTS / (connected.STEM + "-uart.zip")
    inputs = read(report)
    plan = read(connected.PLAN)
    assert sha(connected.PLAN) == inputs["plan_sha256"] and sha(archive) == inputs["archive_sha256"]
    old = RESULTS / (earlier.study.STEM + "-command-start.json")
    rule = read(old)
    assert sha(Path(earlier.__file__)) == rule["script_sha256"] and rule["guard_samples"] == earlier.GUARD
    reader, identity = earlier.fixed.host.piper.load_reader()
    decoder = earlier.study.uplink.decoder_api()
    assert identity == rule["asr"] == inputs["asr"]
    counts, rows = Counter(), []
    with ZipFile(archive) as uart:
        for r in inputs["records"]:
            if r["role"] != "fixed-v5":
                continue
            src = plan["records"][r["index"]]
            assert r["board"] and src["pcm_sha256"] == r["pcm_sha256"]
            meta = dict(command_at=src["end"], carrier=src["carrier"])
            result = earlier.score_capture(r["board"], meta, uart, reader, decoder, rule["tail_samples"])
            for lane, data in result["lanes"].items():
                for method in ("full", "from_wake", "from_code"):
                    for key in ("exact", "complete_suffix"):
                        counts[".".join((lane,method,key))] += data[method][key]
            rows.append(dict(source=meta, voice=r["voice"], **result))
    assert len(rows) == 12
    save(RESULTS / (connected.STEM+"-crops.json"), dict(records=rows, summary=counts, asr=identity,
         input_sha256=sha(report), uart_sha256=sha(archive), prior_rule_sha256=sha(old), script_sha256=sha(Path(__file__)),
         policy="Reuse the original 80 ms source tail, 20 ms headroom and zero-reference lane delay. No new "
                "timing fit. Decode both actual P4 Opus channels and verify their complete PCM hashes first."))


if __name__ == "__main__":
    main()
