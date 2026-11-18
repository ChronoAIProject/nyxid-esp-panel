"""Transcribe captured commands on the calibrated lexical clocks with and without panel pre-roll."""
from pathlib import Path

from faster_whisper import WhisperModel

from check_c161_v5_command_ns import digest
from check_c161_v5_far_command_noise import normalized
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from run_c161_first_command_v5 import transcripts
from select_c161_synthesized_cue_v2 import load_wav

BASE = "4090-c161-pocketsphinx"
FRESH = BASE+"-willow"


def inputs():
    old = read(RESULTS / f"{BASE}-calibrated-v1.json")
    forward = read(RESULTS/"4090-c161-signed-forward-v1.json")["records"]
    for r in old["records"]:
        if r["cohort"] == "forward":
            src = forward[int(r["key"])]
            if src["target_id"] == "nyx":
                yield "forward",src,dict(start=src["start"],end=src["end"],first_phone=src["first_phone"],
                                          command=src["carrier"]), dict(direct=r["direct_events"], combined=r["merged"])
    sources = read(RESULTS/f"{FRESH}-v1.json")["records"]
    cases = read(RESULTS/f"{FRESH}-lexical.json")["records"]
    for src, r in zip(sources,cases,strict=True):
        assert src["pcm_sha256"] == r["pcm_sha256"]
        for span in src["timeline"]:
            if span["role"]=="nyx":
                yield "new",src,dict(start=span["start"],end=span["end"],first_phone=span["first_phone"],
                                     command=span["text"].split(",",1)[1].strip()),r["search"]["calibrated"]


def main():
    path = Path(".local/wakeword/whisper-small-en")
    expected = read(RESULTS/"4090-parallel-melo-context-asr-plan-v1.json")["files"]
    identity = {p.name:sha(p) for p in path.iterdir() if p.is_file()}
    assert all(expected[str(path/name)]==h for name,h in identity.items())
    model = WhisperModel(str(path), device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
    assert model.model.device=="cuda"
    rows, readings = [], {}
    for cohort, src, span, routes in inputs():
        pcm = load_wav(src)
        for mode in ("direct","combined"):
            hits=[r["sample"] for r in routes[mode] if span["start"]<=r["sample"]<span["end"]]
            row=dict(cohort=cohort,voice=src["voice"],mode=mode,source_pcm_sha256=digest(pcm),span=span,hits=hits,captures={})
            if hits:
                block_end=(hits[0]+159)//160*160
                for label, lookback in (("immediate",0),("preroll",32000)):
                    start=max(0,block_end-lookback)
                    audio=pcm[start:span["end"]]
                    key=digest(audio)
                    if key not in readings:
                        readings[key]=transcripts(model,audio)
                    text=normalized(readings[key]["text"])
                    command=normalized(span["command"])
                    row["captures"][label]=dict(first_sample=start,samples=len(audio),sha256=key,
                        onset_retained=start<=span["first_phone"],first_word_present=command[0] in text,
                        exact_command_suffix=text[-len(command):]==command)
            rows.append(row)
    save(RESULTS/f"{BASE}-first-command.json",dict(model_files=identity, asr="Whisper small.en CUDA FP16, unprompted English",
         forward_sha256=sha(RESULTS/"4090-c161-signed-forward-v1.json"),
         forward_results_sha256=sha(RESULTS/f"{BASE}-calibrated-v1.json"), new_results_sha256=sha(RESULTS/f"{FRESH}-lexical.json"),
         cap="mono source WAV, 160-sample delivery boundary, 32000-sample host pre-roll", records=rows, transcripts=readings))


if __name__=="__main__":
    main()
