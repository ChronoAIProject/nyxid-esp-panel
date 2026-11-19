"""Verify the portable reader on synthesized captures with known clock and channel faults."""
from hashlib import sha256
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import wave
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

import measure_cue_clock as reader

BASE = reader.RESULTS/"4090-c161-cue-clock-reader"
NAME = "hye-nyx-v5-clock-63s.wav"


def observed(original, ppm, delay, step=0):
    n = len(original)
    x = np.arange(n)
    t = (x-delay-np.where(x >= 35*reader.RATE, step, 0)) / (1 + ppm/1e6)
    return np.rint(np.interp(t, x, original, left=0, right=0)).astype("<i2")


def main():
    data = reader.playback()
    digest = sha256(data).hexdigest()
    with wave.open(io.BytesIO(data)) as w:
        assert w.getparams()[:4] == (1, 2, 16000, 63*16000)
        signal = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    archive = BASE.with_suffix(".zip")
    with ZipFile(archive.with_suffix(".tmp"), "w") as z:
        z.writestr(ZipInfo(NAME), data, compress_type=ZIP_DEFLATED, compresslevel=9)
    archive.with_suffix(".tmp").replace(archive)
    with ZipFile(archive) as z:
        assert z.testzip() is None and z.read(NAME) == data
    rows = []
    with TemporaryDirectory() as tmp:
        for layout, mic_rate, ref_rate, fault in (
                (2, 0, 0, "none"), (4, 100, 100, "none"), (2, -100, 100, "none"),
                (4, 100, -100, "none"), (2, 0, 0, "reference_missing_one"),
                (4, 0, 0, "reference_step_256"), (2, 0, 0, "reference_silent")):
            mic = observed(signal, mic_rate, 320)
            ref = observed(signal, ref_rate, 64, 256 if fault == "reference_step_256" else 0)
            if fault == "reference_missing_one":
                ref[20*reader.RATE:22*reader.RATE] = 0
            if fault == "reference_silent":
                ref[:] = 0
            channels = [mic, ref] if layout == 2 else [mic, ref, np.zeros_like(ref), mic]
            path = Path(tmp)/"capture.wav"
            with wave.open(str(path), "wb") as w:
                w.setparams((layout, 2, reader.RATE, 0, "NONE", "not compressed"))
                w.writeframes(np.column_stack(channels).astype("<i2").tobytes())
            result = reader.measure(path)
            assert abs(result["mic"]["excess_capture_ppm"]-mic_rate) < 5
            if fault == "none":
                assert abs(result["reference"]["excess_capture_ppm"]-ref_rate) < 5
                assert abs(result["ref_minus_mic_ppm_on_source_clock"] - (ref_rate-mic_rate)) < 5
                if mic_rate == ref_rate:
                    assert result["ref_minus_mic_samples"] == [-256] * len(reader.STARTS)
            else:
                status = "inconclusive_irregular_spacing" if fault == "reference_step_256" else "inconclusive_event_count"
                assert result["reference"]["status"] == status and "ref_minus_mic_samples" not in result
            rows.append(dict(channels=layout, injected_mic_ppm=mic_rate, injected_ref_ppm=ref_rate,
                             fault=fault, result=result))
    out = dict(records=rows, source_name=NAME, source_bytes=len(data), source_wav_sha256=digest,
               archive_sha256=sha256(archive.read_bytes()).hexdigest(),
               source_code_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in
                                   (Path(__file__), Path(reader.__file__), Path(reader.__file__).with_name("read_mr_capture.cpp"),
                                    Path(reader.__file__).with_name("read_mr_capture.py"))})
    BASE.with_suffix(".json").write_text(json.dumps(out, indent=2)+"\n")


if __name__ == "__main__":
    main()
