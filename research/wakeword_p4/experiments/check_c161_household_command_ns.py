"""Apply the two frozen command-only enhancement configurations to the new household mixtures."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile, ZIP_DEFLATED

from faster_whisper import WhisperModel

import check_c161_v5_command_ns as p4
from c161_v5_household_rooms import OUT, PLAN, STEM
from check_c161_v5_command_deepfilter import BINARY, BINARY_SHA, enhance
from check_c161_v5_far_command_noise import normalized
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from run_c161_first_command_v5 import transcripts
from select_c161_synthesized_cue_v2 import load_wav


def main():
    prior = read(RESULTS/"4090-c161-v5-command-ns.json")
    assert sha(p4.BUILD/"nyx_first_command.elf") == prior["elf_sha256"]
    assert sha(BINARY) == BINARY_SHA
    assert all(sha(Path(k)) == v for k, v in prior["model_files"].items())
    audio = read(RESULTS/f"{STEM}-audio.json")
    assert audio["plan_sha256"] == sha(PLAN)
    rows = [(s, load_wav(s)) for s in read(PLAN)["records"] if s["role"] == "v5"]
    raw = [r for r in audio["records"] if "command_transcript" in r]
    assert len(raw) == len(rows) == 24
    path = OUT/"noise-suppression"
    with patch.object(p4,"OUT",path):
        p4.replay(rows)
    reader = WhisperModel(".local/wakeword/whisper-small-en", device="cuda", compute_type="float16",cpu_threads=1,num_workers=1)
    records = []
    with TemporaryDirectory(dir=OUT) as tmp:
        for group in range(0,len(rows),3):
            for j,(origin,ns_out,allocated) in enumerate(p4.frames(path/f"{group}.log", rows, group)):
                i = group+j
                src, pcm = rows[i]
                onset = src["command_at"]
                lag, corr = p4.aligned_lag(pcm, ns_out, origin, onset)
                native = ns_out[onset-origin+lag:len(pcm)-origin+lag]
                host, clips = enhance(pcm,Path(tmp))
                assert len(native) == len(host[onset:]) == len(pcm[onset:])
                row = dict(source_index=raw[i]["index"], pcm_sha256=src["pcm_sha256"], voice=src["voice"],
                           noise=src["noise_type"], snr_db=src["command_snr_db"], raw=raw[i]["command_transcript"]["text"],
                           command=src["carrier"], ns_delay=lag, ns_correlation=corr, ns_allocated=allocated, host_clips=clips)
                expected = normalized(src["carrier"])
                for key, segment in (("p4",native),("host",host[onset:])):
                    result = transcripts(reader,segment)
                    words = normalized(result["text"])
                    row[key] = dict(transcript=result, sha256=p4.digest(segment), first=words[:1]==expected[:1], exact=words==expected)
                records.append(row)
    archive = RESULTS/f"{STEM}-ns-uart.zip"
    with ZipFile(archive, "x", ZIP_DEFLATED) as z:
        for group in range(0,len(rows),3):
            z.write(path/f"{group}.log",f"{group}.log")
    save(RESULTS/f"{STEM}-ns.json",dict(plan_sha256=sha(PLAN), settings_sha256=sha(RESULTS/"4090-c161-v5-command-ns.json"),
         host_binary_sha256=BINARY_SHA, archive_sha256=sha(archive), records=records))


if __name__ == "__main__":
    main()
