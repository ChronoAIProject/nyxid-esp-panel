"""Measure English model and dictionary loading in isolated host processes on identical streams."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from pocketsphinx import Config, Decoder

from calibrate_c161_pocketsphinx import OUT
from check_c161_pocketsphinx import BLOCK, PHRASE, STEM, search
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from select_c161_synthesized_cue_v2 import load_wav


def memory_kib():
    stats = {}
    for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("Rss", "Pss", "Private_Dirty"):
            stats[key] = int(value.split()[0])
    return stats


def measure(mode, destination):
    threshold = 10.**(-read(OUT)["selected_exponent"])
    config = Config(keyphrase=PHRASE, loglevel="ERROR", kws_threshold=threshold)
    original = Path(config["dict"])
    compact = destination.parent/"hey-nicks.dict"
    if mode != "stock":
        expected = {"hey HH EY", "nicks N IH K S"}
        chosen = [s for s in original.read_text().splitlines() if s.split()[0] in ("hey", "nicks")]
        assert set(chosen) == expected
        compact.write_text("\n".join(chosen)+"\n")
        config["dict"] = str(compact)
    config["mmap"] = mode != "compact-read"
    inputs = [(r, load_wav(r)) for r in read(RESULTS/"4090-c161-signed-forward-v1.json")["records"]]
    before = memory_kib()
    start = time.process_time()
    decoder = Decoder(config)
    initialized = time.process_time()
    init_memory = memory_kib()
    baseline = decoder.get_cmn()
    rows = []
    for r, pcm in inputs:
        detections = search(decoder, baseline, pcm, 0)
        rows.append(dict(pcm_sha256=r["pcm_sha256"], samples=len(pcm), matches=detections))
    finish = time.process_time()
    ready = memory_kib()
    dictionary = Path(config["dict"])
    destination.write_text(json.dumps(dict(mode=mode, mmap=config["mmap"], dict_sha256=sha(dictionary),
        dictionary_bytes=dictionary.stat().st_size, before_kib=before, initialized_kib=init_memory,
        decoded_kib=ready, initialization_cpu_seconds=initialized-start, replay_cpu_seconds=finish-initialized,
        samples=sum(len(p) for _,p in inputs), records=rows)))


def main():
    source = RESULTS/f"{STEM}-calibrated-v1.json"
    originals = {r["pcm_sha256"]:r["direct"] for r in read(source)["records"] if r["cohort"] == "forward"}
    results = []
    with tempfile.TemporaryDirectory() as scratch:
        for arm in ("stock","compact-mapped","compact-read"):
            out = Path(scratch)/(arm+".json")
            subprocess.run([sys.executable,__file__,arm,str(out)],check=True)
            row = read(out)
            for r in row["records"]:
                r["exact_to_published_stock"] = r["matches"] == originals[r["pcm_sha256"]]
            assert len(row["records"])==len(originals)==24
            results.append(row)
    save(RESULTS/f"{STEM}-dictionary-v1.json",dict(stock_reference_sha256=sha(source), calibration_sha256=sha(OUT),
         platform="Linux x86_64 host, independent Python processes, 30 ms online polling, smaps_rollup snapshots",
         block=BLOCK, records=results))


if __name__=="__main__":
    if len(sys.argv)==1:
        main()
    else:
        measure(sys.argv[1],Path(sys.argv[2]))
