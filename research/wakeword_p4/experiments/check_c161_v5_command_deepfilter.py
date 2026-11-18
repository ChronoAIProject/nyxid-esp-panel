"""Score the released DeepFilterNet executable on the exact raw first-command streams."""
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from check_c161_v5_command_ns import cases, digest
from check_c161_v5_far_command_noise import normalized
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from run_c161_first_command_v5 import WORK, transcripts
from select_c161_synthesized_cue_v2 import load_wav

STEM = "4090-c161-v5-command-deepfilter"
BINARY = WORK/"deepfilter/deep-filter-0.5.6-x86_64-unknown-linux-musl"
BINARY_SHA = "70775e251eee44c0f2451a1e833326cf8bcbbe304d3e7cd12851e6fce72ef7da"
URL = "https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/" + BINARY.name


def enhance(pcm, directory):
    """Each stream starts a fresh process. Give the model sufficient tail to flush its lookahead."""
    path = directory/"input.wav"
    padded = np.pad(pcm.astype(np.float32)/32768, (0, 1600))
    sampled = resample_poly(padded, 3, 1)
    sf.write(path, sampled, 48000, subtype="FLOAT")
    subprocess.run(["taskset", "-c", str(min(os.sched_getaffinity(0))), str(BINARY), "-D", "-o",
                    str(directory/"out"), str(path)], check=True, capture_output=True)
    processed, sr = sf.read(directory/"out"/path.name, dtype="float32")
    assert sr == 48000 and np.isfinite(processed).all()
    restored = resample_poly(processed, 1, 3)
    assert len(restored) >= len(pcm)
    integers = np.rint(restored[:len(pcm)] * 32768)
    clips = int(np.count_nonzero((integers < -32768) | (integers > 32767)))
    return integers.clip(-32768, 32767).astype("<i2"), clips


def main():
    from faster_whisper import WhisperModel
    assert sha(BINARY) == BINARY_SHA
    prior = RESULTS/"4090-c161-v5-command-ns.json"
    reference = read(prior)
    model = Path(".local/wakeword/whisper-small-en")
    assert {str(p): sha(p) for p in model.iterdir() if p.is_file()} == reference["model_files"]
    reader = WhisperModel(str(model), device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
    source_plan = RESULTS/"4090-c161-signed-forward-plan-v1.json"
    originals = {(r["voice"], r["scene"]): r for r in read(source_plan)["records"] if r["target_id"] == "nyx"}
    records, boundaries = [], []
    with TemporaryDirectory(dir=WORK) as tmp:
        for (src, pcm), ref in zip(cases(), reference["records"], strict=True):
            assert digest(pcm) == ref["full_input_sha256"]
            output, clips = enhance(pcm, Path(tmp))
            onset = src["end"]
            result = transcripts(reader, output[onset:])
            expected, got = normalized(ref["expected"]), normalized(result["text"])
            row = dict(voice=src["voice"], scene=src["scene"], channel=src["channel"], expected=ref["expected"],
                       input_pcm_sha256=digest(pcm), output_pcm_sha256=digest(output), render_clips=clips,
                       transcript=result, first=got[:1] == expected[:1], exact=got == expected)
            if src["voice"] == "af_aoede" and src["scene"] == 0 and src["channel"] == "far":
                wav = RESULTS/f"{STEM}-far.wav"
                with wav.open("xb") as f:
                    sf.write(f, output, 16000, format="WAV", subtype="PCM_16")
                row.update(wav=wav.name, wav_sha256=sha(wav))
            records.append(row)
            if src["channel"] == "digital":
                # Test whether the annotated crop itself makes an otherwise clear verb disappear.
                old = originals[src["voice"],src["scene"]]
                source = load_wav(old)
                cut = old["first_phone"]
                np.testing.assert_array_equal(pcm[onset:], source[cut:])
                probes = dict(at_annotation=source[cut:], silence_200ms=np.r_[np.zeros(3200,np.int16), source[cut:]],
                              previous_200ms=source[cut-3200:], full_request=source[old["start"]:])
                trials = {k: dict(pcm_sha256=digest(v), transcript=transcripts(reader,v)) for k,v in probes.items()}
                assert trials["at_annotation"]["transcript"]["text"] == ref["input"]["transcript"]["text"]
                boundaries.append(dict(voice=src["voice"], scene=src["scene"], expected=ref["expected"],
                                       original_wav_sha256=old["sha256"], annotated_command_onset=cut, trials=trials))
    save(RESULTS/f"{STEM}.json",dict(release_url=URL, executable_sha256=BINARY_SHA,
         source_plan_sha256=sha(source_plan), reference_sha256=sha(prior),
         configuration="built-in model, default parameters, compensated latency, fresh process per input, 48 kHz resampling",
         model_files=reference["model_files"], records=records, source_boundary_checks=boundaries))


if __name__ == "__main__":
    main()
