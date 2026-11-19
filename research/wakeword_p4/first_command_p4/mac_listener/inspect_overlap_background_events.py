"""Recover the event-local mic context for the 23 background notifications after immediate checking."""
from pathlib import Path
from tempfile import TemporaryDirectory

import soundfile as sf

import check_overlap_background as background
import check_balanced_connected_commands as balanced
from run_c161_first_command_v5 import transcripts

ch = background.ch
OUT = ch.RESULTS/(background.score.fit.STEM+"-background-context.json")


def main():
    report = ch.read(background.OUT)
    qwen = balanced.original.asr
    processor, model, qid = qwen.load()
    prior = ch.read(balanced.OUT)
    assert qid == prior["qwen"]
    turbo = balanced.tx.diag.reader(prior["turbo"])
    rows = []
    inputs = {}
    for cohort in ("melo","primed","jenny"):
        path = ch.RESULTS/f"4090-c161-history-gain-{cohort}-v1.json"
        inputs.update({f'{cohort}_{r["index"]}':r["source"] for r in ch.read(path)["records"] if r["phase"]=="background"})
    for src in report["records"]:
        if not src["events"]["confirm_0"]:
            continue
        record = inputs[src["key"]]
        p = Path(record.get("wav_path") or record["path"])
        pcm, rate = sf.read(p, dtype="int16")
        assert rate==16000 and len(pcm)==src["samples"] and ch.digest(pcm)==src["pcm_sha256"]
        for t in src["events"]["confirm_0"]:
            before = max(0,t-48000)
            after = min(len(pcm),t+16000)
            views = {}
            for name, start, end in (("context",before,after),("detector_window",max(0,t-15360),t)):
                x = pcm[start:end]
                whisper = transcripts(turbo,x)
                views[name] = dict(start=start,end=end,pcm_sha256=ch.digest(x),
                                   qwen=qwen.transcribe(processor,model,x),turbo=whisper)
            rows.append(dict(source=src["key"],sample=t,source_pcm_sha256=src["pcm_sha256"],views=views))
    assert len(rows)==report["notifications"]["confirm_0"]
    with TemporaryDirectory(dir=OUT.parent) as folder:
        tmp=Path(folder)/OUT.name
        ch.save(tmp,dict(records=rows,source_sha256=ch.sha(background.OUT),qwen=qid,turbo=prior["turbo"],
                         script_sha256=ch.sha(Path(__file__)),policy="Transcribe every immediate-confirmation "
                         "background event from original PCM, with a four-second context and the 960 ms detector "
                         "window ending at the event. No target prompts. Text is automatic, without human annotation "
                         "or word-level causal attribution. No new exposure."))
        tmp.replace(OUT)


if __name__ == "__main__":
    main()
