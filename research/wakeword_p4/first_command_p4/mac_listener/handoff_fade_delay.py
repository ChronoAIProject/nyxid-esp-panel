"""Delay the same fade by one or four 20-ms audio periods after the recorded OPEN."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

import handoff_barge_in as handoff

listener = handoff.listener
BASE = listener.RESULTS / "4090-c161-handoff-fade-delay"
DELAYS_MS = (20, 80)


def main():
    parent = handoff.BASE.with_suffix(".json")
    history = listener.read(parent)
    reader, asr = listener.load_reader()
    assert asr == history["asr"]
    decoder = handoff.probe.uplink.decoder_api()
    firmware = handoff.test.BUILD / "nyx_mac_listener.bin"
    assert listener.sha(firmware) == history["firmware_sha256"]
    assert listener.sha(handoff.probe.uplink.CODEC/"nyx_first_command.bin") == history["codec_sha256"]
    rows = []
    archive = BASE.with_suffix(".zip")
    partial = BASE.with_suffix(".tmp")
    with TemporaryDirectory() as tmpdir, ZipFile(partial, "w", ZIP_DEFLATED) as logs:
        work = Path(tmpdir)
        detector = handoff.fixed.ch.binding(work/"full.so", coded="v5")
        pair = handoff.fixed.ch.binding(work/"pair.so", coded="aligned")
        for (source, dry, player, response), old in zip(handoff.scenes(), history["records"], strict=True):
            assert source == old["source"]
            noise = np.random.default_rng(18221).standard_normal(len(player))
            gain = source["echo_gain"]
            original_mic, original_ref, _ = handoff.render(dry, player, response, gain, noise)
            assert listener.digest(original_mic) == old["continue"]["mic_sha256"]
            assert listener.digest(original_ref) == old["continue"]["reference_sha256"]
            opened, = old["continue"]["opened"]
            for delay in DELAYS_MS:
                stop = opened + 16*delay
                ref = player.copy()
                fade = handoff.FADE
                ref[stop:stop+fade] = np.rint(ref[stop:stop+fade]*np.arange(fade-1, -1, -1)/fade)
                ref[stop+fade:] = 0
                mic, adc, clips = handoff.render(dry, ref, response, gain, noise)
                np.testing.assert_array_equal(mic[:stop], original_mic[:stop])
                np.testing.assert_array_equal(adc[:stop], original_ref[:stop])
                i = len(rows)
                result = handoff.probe.score(work, source, mic, adc, detector, i, build=handoff.test.BUILD,
                        prefix=pair, qualify=handoff.probe.qualify, wait_for_third=True,
                        frame_activity=True, qualify_capture_frame=True)
                assert result["opened"] == [opened]
                logs.write(work/"serial.log", f"{i:02}-listener.log")
                uplink = handoff.bench.fifo(work, logs, i, mic, adc, result, reader, decoder, True, True)
                rows.append(dict(delay_ms=delay, fade_at=stop, fade_samples=fade, clipped_samples=clips,
                                 speaker_sha256=listener.digest(ref), result=result, uplink=uplink))
                print(i, delay, uplink["recognition"]["text"], flush=True)
    assert len(rows) == 32
    partial.replace(archive)
    listener.save(BASE.with_suffix(".json"), dict(records=rows, parent_sha256=listener.sha(parent),
            archive_sha256=listener.sha(archive), asr=asr, code_sha256=listener.sha(Path(__file__)),
            policy="Same 16 paired handoff scenes as the immediate fade. Delays from measured P4 OPEN "
                   "are 20 and 80 ms; same 10 ms fade, unaltered room decay. Prefix through delay "
                   "checked on both channels. No physical output latency claimed."))


if __name__ == "__main__":
    main()
