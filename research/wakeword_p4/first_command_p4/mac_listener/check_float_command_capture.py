"""Check both float handoff channels and locate the known 079 command through the CLI."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from scipy.io import wavfile

import measure_command_capture as capture


def main(archive, receipt):
    identity = json.loads(receipt.read_bytes())
    assert sha256(archive.read_bytes()).hexdigest() == identity["archive_sha256"]
    with TemporaryDirectory() as tmp, ZipFile(archive) as z:
        folder = Path(tmp)
        meta = json.loads(z.read("manifest.json"))
        start = meta["spoken_command_at"]
        for name in ("old", "new", "willow", "erased"):
            member = "079/"+name+".wav"
            path = folder/(name+".wav")
            blob = z.read(member)
            assert sha256(blob).hexdigest() == identity["files"][member]
            path.write_bytes(blob)
            rate, stereo = wavfile.read(path)
            for i in (0,1):
                pcm, raw, size, sr = capture.read_channel(path,i)
                assert size == 8 and sr == rate == 16000 and raw == stereo[:,i].tobytes()
                np.testing.assert_array_equal(pcm, stereo[:,i]*32768)
            actual = json.loads(subprocess.check_output(
                [sys.executable, capture.__file__, str(path), str(path), "--command-at", str(start)]))
            assert actual["command_pcm_sha256"] == sha256(stereo[start:,0].tobytes()).hexdigest()
            assert actual["status"] == "three_windows_matched"
            assert all(abs(anchor["lag_samples"]-start)<=4 for anchor in actual["anchors"].values())
        scratch = folder/"float32.wav"
        signal = np.array([-.75, .125, .75], dtype="<f4")
        wavfile.write(scratch,48000,signal)
        pcm,raw,size,sr = capture.read_channel(scratch,0)
        assert sr==48000 and size==4 and raw==signal.tobytes()
        np.testing.assert_array_equal(pcm, signal*32768)
        try:
            capture.read_channel(scratch,1)
        except ValueError as exc:
            assert "outside" in str(exc)
        else:
            raise AssertionError("Accepted a missing channel")
    print("Four issued float64 stereo WAVs, both channels, four CLI command-clock and hash checks, float32 mono passed.")


if __name__ == "__main__":
    main(*map(Path,sys.argv[1:]))
