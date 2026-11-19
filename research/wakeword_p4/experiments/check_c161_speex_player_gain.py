"""Test a 500 ms ordinary playback lead and pre/post-volume reference on identical microphones."""
import numpy as np
from scipy.signal import fftconvolve

from check_c161_speex_cold_guard import STEM as PARENT, delayed, replay
from check_c161_speex_hold_rooms import LENGTH, RATE, ROOM_PAIRS, materials, rir, scenes
from fit_c161_history_gain_v1 import RESULTS, read
from check_c161_v5_command_ns import digest

STEM = "4090-c161-speex-player-gain"


def pcm(x, scale):
    return np.rint(x * scale).astype("<i2")


def player_cases():
    cue, narration, _, _ = materials()
    originals = {k: (m, r, meta) for k, m, r, meta in scenes()}
    for room, (local_rir, _) in enumerate(ROOM_PAIRS):
        h, _ = rir(local_rir)
        _, cold_ref, meta = originals[f"{room}-0-self-0-0"]
        ext_warm, _, ext_meta = originals[f"{room}-1-external_only-0-0"]
        assert meta["scale"] == ext_meta["scale"] == 1
        ext = np.r_[ext_warm[5*RATE:], np.zeros(5*RATE, dtype=ext_warm.dtype)]
        command_length = ext_meta["command_end"] - 5*RATE
        at = RATE // 2
        for active in (False, True):
            # A fixed 500 ms from the beginning of the existing narration, followed by the original v5.
            ref = delayed(cold_ref, at)
            if active:
                ref[:at] = narration[:at]
            assert np.array_equal(ref[at:at+len(cue)], cue)
            echo = .45 * fftconvolve(ref.astype(float), h)[:LENGTH]
            for role, start in (("self", None), ("simultaneous", at), ("external_at_boot", 0)):
                mic = echo.copy()
                if start is not None:
                    mic += delayed(ext, start) if start else ext
                key = f"{room}-lead-{int(active)}-{role}"
                scale = min(1., 28000 / max(abs(mic).max(), abs(ref.astype(float)).max()))
                yield key, pcm(mic, scale), pcm(ref, scale), dict(
                    meta, role=role, history="500ms panel narration" if active else "500ms digital silence",
                    scale=scale, known_playback=[at, at+len(cue)], external_start=start,
                    offset=start-at if start is not None else None,
                    command_end=start+command_length if start is not None else None)
        _, source, warm = originals[f"{room}-1-self-0-0"]
        at = warm["known_playback"][0]
        # The player applies gain before the speaker-to-microphone convolution.
        post = source.astype(float)
        post[at:] *= 2
        echo = .45 * fftconvolve(post, h)[:LENGTH]
        for role, start in (("self", None), ("simultaneous", at)):
            mic = echo if start is None else echo + ext_warm
            scale = min(1., 28000 / max(abs(mic).max(), abs(post).max()))
            for tap, reference in (("before", source), ("after", post)):
                yield f"{room}-volume-{tap}-{role}", pcm(mic, scale), pcm(reference, scale), dict(
                    warm, role=role, history="player gain doubles at cue onset", reference_tap=tap,
                    scale=scale, external_start=start, external_db=-6.0206 if start else None,
                    command_end=start+command_length if start else None)


def main():
    parent = RESULTS / f"{PARENT}.json"
    cases = list(player_cases())
    for room in (0, 1):
        by_key = {key: digest(mic) for key, mic, _, _ in cases}
        for role in ("self", "simultaneous"):
            assert by_key[f"{room}-volume-before-{role}"] == by_key[f"{room}-volume-after-{role}"]
    assert len(cases) == 20 and len(read(parent)["records"]) == 34
    replay(STEM, parent, cases, {})


if __name__ == "__main__":
    main()
