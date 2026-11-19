"""Compare the LCD-X8 idle reference selection and the captured MIC3 slot in P4 QEMU."""
import ctypes
from hashlib import sha256
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import butter, sosfilt

import third_reference as probe
from transition import samples, weights

listener = probe.listener
BASE = listener.RESULTS / "4090-c161-lcd-x8-reference"
HEADER = "boards/lcd_x8/product_audio_layout.h"
SERVICE = "components/panel_services/panel_voice.c"


def board_adapter(work):
    def source(path):
        return subprocess.check_output(["git", "show", f"{listener.MAC}:{path}"])

    header, service = source(HEADER), source(SERVICE)
    # Guard the reference selection being emulated against the pinned Mac loop.
    assert b"mic[i]=capture[2*i];ref[i]=c?capture[2*i+1]:0;" in service
    (work / "product_audio_layout.h").write_bytes(header)
    wrapper = work / "tdm.c"
    wrapper.write_text('''#include "product_audio_layout.h"
void read_mr(int16_t *out, const int16_t *tdm, size_t n) {
    while (n) {
        size_t take = n < 128 ? n : 128;
        product_lcd_x8_normalize_capture(out, tdm, take);
        out += take * 2;
        tdm += take * 4;
        n -= take;
    }
}
''')
    library = work / "tdm.so"
    subprocess.run(["cc", "-O2", "-shared", "-fPIC", str(wrapper), "-o", str(library)], check=True)
    unpack = ctypes.CDLL(str(library)).read_mr
    ptr = np.ctypeslib.ndpointer(dtype=np.int16, ndim=2, flags="C_CONTIGUOUS")
    unpack.argtypes = [ptr, ptr, ctypes.c_size_t]
    unpack.restype = None
    return unpack, {path:sha256(data).hexdigest() for path, data in ((HEADER, header), (SERVICE, service))}


def channel(signal, setting, noise):
    if setting in ("idle_mask", "ideal"):
        return signal
    # Stress assumptions only. The Gaussian floor is in signed PCM16 units.
    band = (100, 6500) if setting != "telephone" else (300, 3400)
    gain = .25 if setting != "weak" else .02
    filtered = sosfilt(butter(2, band, btype="bandpass", fs=16000, output="sos"), signal.astype(float))
    floor = 24 if setting != "quiet_line" else 3
    return np.rint(-gain*filtered + floor*noise).clip(-32768, 32767).astype("<i2")


def fifo(work, archive, index, mic, ref, result, reader, decoder):
    log = (work / "serial.log").read_text()
    emitted, first, count, blocks, *_ = listener.values(log, "FINISH")
    assert first == max(0, result["opened"][0]//160*160 - 32000)
    assert count == emitted-first
    mic = np.pad(mic, (0, blocks*256-len(mic)))
    ref = np.pad(ref, (0, blocks*256-len(ref)))
    blend = weights(ref[:emitted])[first:]
    dry = np.pad(mic, (1440, 0))[first:emitted]
    processed = samples(log, "PROCESSED", count)
    mixed = dry*(160-blend) + processed*blend
    selected = samples(log, "AUDIO", count)
    np.testing.assert_array_equal(selected, np.sign(mixed)*(np.abs(mixed)//160))
    uplink = probe.encode_uplink(work, archive, index, selected, mic[first:emitted], result["source"]["carrier"],
                                 reader, decoder)
    return dict(**uplink, samples=count, processed_samples=int((blend == 160).sum()),
                delayed_raw_samples=int((blend == 0).sum()))


def main():
    previous = listener.read(probe.BASE.with_suffix(".json"))
    fw = probe.BUILD / "nyx_mac_listener.bin"
    assert listener.sha(fw) == previous["listener_sha256"]
    assert listener.sha(probe.uplink.CODEC/"nyx_first_command.bin") == previous["opus_sha256"]
    identity = {(r["source"]["room"], r["source"]["mode"], r["source"]["noisy"], r["source"]["player"]):r
                for r in previous["records"] if r["group"] == "rooms"}
    reader, asr = listener.load_reader()
    assert asr == previous["asr_files"]
    decoder = probe.uplink.decoder_api()
    records = []
    archive = BASE.with_suffix(".zip")
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        work = Path(tmp)
        unpack, files = board_adapter(work)
        full = probe.fixed.ch.binding(work/"full.so", coded="v5")
        pair = probe.fixed.ch.binding(work/"pair.so", coded="aligned")
        for source, mic, player, _ in probe.shower.streams():
            if not source["noisy"] or source["mode"] == "voip-48":
                continue
            old = identity[source["room"], source["mode"], True, source["player"]]
            assert listener.digest(mic) == old["mic_sha256"] and listener.digest(player) == old["reference_sha256"]
            noise = np.random.default_rng(18221).standard_normal(len(mic))
            for setting in ("idle_mask", "ideal", "quiet_line", "noisy_line", "weak", "telephone"):
                captured = channel(player, setting, noise)
                tdm = np.column_stack((mic, captured, np.bitwise_not(mic), np.bitwise_not(captured)))
                mr = np.empty((len(mic), 2), dtype="<i2")
                unpack(mr, tdm, len(mic))
                np.testing.assert_array_equal(mr[:, 0], mic)
                np.testing.assert_array_equal(mr[:, 1], captured)
                ref = np.zeros_like(captured) if setting == "idle_mask" else mr[:, 1].copy()
                meta = dict(source, reference_path=setting)
                i = len(records)
                result = probe.score(work, meta, mr[:, 0].copy(), ref, full, i, build=probe.BUILD, prefix=pair,
                                     qualify=probe.qualify, wait_for_third=True)
                assert (result["c161"], result["fixed"]) == (old["c161"], old["fixed"])
                if setting == "ideal":
                    assert result["opened"] == old["opened"]
                result.update(tdm_sha256=listener.digest(tdm), captured_reference_sha256=listener.digest(captured),
                              floor_rms=float(np.sqrt(np.mean(captured[:source["cue_at"]].astype(float)**2))),
                              peak_reference=int(abs(captured.astype(np.int32)).max()))
                if source["player"] == "external" and result["opened"]:
                    result["uplink"] = fifo(work, logs, i, mic, ref, result, reader, decoder)
                logs.writestr(f"{i:02}-listener.log", "\n".join(result["uart_events"])+"\n")
                records.append(result)
                print(i, source["player"], setting, result["opened"], flush=True)
        assert len(records) == 48
    archive.with_suffix(".tmp").replace(archive)
    listener.save(BASE.with_suffix(".json"), dict(records=records, mac_commit=listener.MAC, mac_files=files,
        prior_sha256=listener.sha(probe.BASE.with_suffix(".json")), cue_sha256=listener.sha(probe.fixed.CUE),
        model_sha256=listener.C161, listener_sha256=listener.sha(fw), opus_sha256=previous["opus_sha256"],
        qemu_sha256=listener.sha(listener.QEMU), asr=asr, archive_sha256=listener.sha(archive),
        code_sha256=listener.sha(Path(__file__)), policy="Simulated analog reference, no board measurements. "
        "Source and microphone unchanged within each pair. External inputs have a silent clean player lane. "
        "Idle mask mirrors Mac idle AEC ref assignment. All conditions use the same third-reference research "
        "listener, including reference-based call FIFO selection. The pinned LCD-X8 normalizer runs on the host."))


if __name__ == "__main__":
    main()
