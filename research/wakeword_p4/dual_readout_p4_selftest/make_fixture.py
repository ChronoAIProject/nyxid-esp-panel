"""Rebuild one forward stream's actual auxiliary input and both reference outputs."""
import ctypes
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

directory = Path(__file__).resolve().parent
sys.path.insert(0, str(directory.parent / "experiments"))
from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, read, sha
from select_c161_synthesized_cue_v2 import load_wav
import replay_c161_ambient_history_v1 as frontend

p = RESULTS / "4090-c161-four-voice-forward-plan-v1.json"
row = read(p)["records"][12]
assert (row["voice"], row["scene"], row["target_id"]) == ("bf_lily", 0, "nyx")
lib = frontend.BUILD / "frontend.so"
assert sha(lib) == read(frontend.BUILD / "build.json")["library_sha256"]
native = ctypes.CDLL(str(lib.resolve())).ambient_history
native.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
with patch.object(frontend, "score", lambda x, clk: x):
    features, clock, _, _ = frontend.extract(native, load_wav(row))
it = REF_KERNEL(model_path=str(RESULTS / "4090-c161-dual-readout-v1.tflite"))
it.allocate_tensors()
scale, zero = it.get_input_details()[0]["quantization"]
codes = np.clip(np.rint(features["ambient"] / scale) + zero, -128, 127).astype(np.int8).reshape(-1, 120)
arrays = np.empty((len(codes), 122), np.uint8)
arrays[:, :120] = codes.view(np.uint8)
for j, (data, report, key, label) in enumerate([
    ("scores", "", "12_codes", "scores_sha256"), ("trained-scores", "-review", "12", "scores_sha256")]):
    path = RESULTS / f"4090-c161-four-voice-forward-{data}-v1.npz"
    assert sha(path) == read(RESULTS / f"4090-c161-four-voice-forward{report}-v1.json")[label]
    with np.load(path) as z:
        arrays[:, 120+j] = z[key][1]
dest = directory / "main/fixture.bin"
if dest.exists():
    assert dest.read_bytes() == arrays.tobytes()
else:
    dest.write_bytes(arrays.tobytes())
print(len(arrays), np.count_nonzero(arrays[:, 120] != arrays[:, 121]), sha(dest))
