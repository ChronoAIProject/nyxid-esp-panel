"""Check whether the isolated background event recurs on ordinary synthetic uses of 'across'."""
from collections import Counter, defaultdict
from fractions import Fraction
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve, resample_poly
import soundfile as sf
from kokoro_onnx import Kokoro

import inspect_final_marker_event as previous
from check_c161_wake_phone_duration_v1 import session
from select_room_spoken_cues import VOICES

host, ch, grid = previous.host, previous.ch, previous.host.grid
OUT = ch.RESULTS/"4090-c161-across-marker-event.json"
PHRASES = ("Across.", "Across the street.", "We walked across the room.", "For major tenants across the portfolio.")
SPEEDS = (.85, 1., 1.15)


def main():
    files = [Path(".local/wakeword/hey-common-carrier-kokoro-seeded-v1.onnx"), Path(".local/wakeword/kokoro-models/voices-v1.0.bin")]
    inventory = ch.read(ch.RESULTS/"4090-hey-common-carrier-kokoro-generation-execution-v1.json")
    assert all(ch.sha(p) == inventory["files"][str(p)] for p in files)
    engine = session(str(files[0]))
    synth = Kokoro.from_session(engine, str(files[1]))
    rooms = []
    for prefix, kind in (("h012_", "air_conditioner"), ("h004_", "music")):
        h, path, checksum = grid.room.rooms.read_room(prefix)
        noise, noise_path, noise_checksum = grid.room.rooms.read_noise(kind)
        rooms.append((h, noise, dict(kind=kind, room=path, rir_sha256=checksum, noise=noise_path, noise_sha256=noise_checksum)))
    native = grid.room.rooms.c161_binding()
    rows, recorded, summary = [], [], defaultdict(Counter)
    event, = ch.read(previous.OUT)["records"]
    background = ch.read(ch.RESULTS/"4090-c161-history-gain-jenny-v1.json")
    source, = [r["source"] for r in background["records"] if r["phase"] == "background" and r["source"]["pcm_sha256"] == event["source_pcm_sha256"]]
    with sf.SoundFile(source["wav_path"]) as stream:
        stream.seek(event["offset"])
        excerpt = stream.read(event["samples"], dtype="int16")
    assert ch.digest(excerpt) == event["clip_pcm_sha256"]
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as z:
        work = Path(tmp)
        candidate, strict = host.binding(work), ch.binding(work/"strict.so", coded="v5")

        def score(x):
            events = host.detect(candidate, [x])
            before = grid.fit.fixed.ch.routes(strict, native, x)
            return dict(pcm_sha256=before["pcm_sha256"], strict=before["fixed"], c161=before["c161"],
                        candidate=events, combined=host.joined(before["c161"], events))

        for ppm in (-10000, -1000, 0, 1000, 10000):
            changed = grid.drift.project(excerpt, Fraction(1000000, 1000000+ppm))
            for gain in (.5, 1.):
                x = grid.room.quantize(changed.astype(float)*gain)
                val = score(x)
                if ppm == 0 and gain == 1:
                    assert val["pcm_sha256"] == event["clip_pcm_sha256"] and val["combined"] == [event["relative_event"]]
                recorded.append(dict(player_ppm=ppm, gain=gain, **val))
        for voice in VOICES:
            lang = "en-gb" if voice.startswith("b") else "en-us"
            for k, text in enumerate(PHRASES):
                phones = synth.tokenizer.phonemize(text, lang)
                ids = synth.tokenizer.tokenize(phones)
                for speed in SPEEDS:
                    samples, durations = engine.run(None,dict(input_ids=np.array([[0,*ids,0]],np.int64),
                        style=np.asarray(synth.get_voice_style(voice)[len(ids)],np.float32),speed=np.array([speed],np.float32)))
                    dry = grid.room.quantize(resample_poly(samples.ravel(), 2, 3)*32768)
                    assert len(dry) == int(durations.sum())*400
                    member = f"{len(rows):03}.flac"
                    buf = io.BytesIO()
                    sf.write(buf, dry, 16000, format="FLAC", subtype="PCM_16")
                    z.writestr(member, buf.getvalue())
                    cases = {"direct":score(np.pad(dry, (16000, 16000)))}
                    for h, noise, meta in rooms:
                        wet = fftconvolve(dry.astype(float), h)
                        n = len(wet)+32000
                        offset = (len(rows)*9731)%len(noise)
                        bg = noise[(offset+np.arange(n))%len(noise)]
                        level = np.linalg.norm(wet) / (np.linalg.norm(bg[16000:16000+len(wet)]) * 2)
                        mixed = bg*level
                        mixed[16000:16000+len(wet)] += wet
                        gain = min(1., 28000 / np.abs(mixed).max())
                        cases[meta["kind"]] = dict(**score(grid.room.quantize(mixed*gain)), offset=offset, noise_gain=level, gain=gain)
                    for channel, vals in cases.items():
                        summary[channel].update(trials=1, **{name:bool(vals[name]) for name in ("candidate","strict","c161","combined")})
                    rows.append(dict(voice=voice, text=text, phones=phones, speed=speed, member=member, dry_pcm_sha256=ch.digest(dry), cases=cases))
            print(voice, len(rows), flush=True)
    with ZipFile(archive.with_suffix(".tmp")) as z:
        assert len(rows) == 96 and z.testzip() is None
        for row in rows:
            pcm, rate = sf.read(io.BytesIO(z.read(row["member"])), dtype="int16")
            assert rate == 16000 and ch.digest(pcm) == row["dry_pcm_sha256"]
    archive.with_suffix(".tmp").replace(archive)
    ch.save(OUT, dict(records=rows, recorded_excerpt=recorded, summary=summary, rooms=[r[2] for r in rooms],
                      archive_sha256=ch.sha(archive), sources={p.name:ch.sha(p) for p in [*files, previous.OUT, Path(__file__)]},
                      policy="Targeted post-event test on the transcribed word across. Four phrases by eight existing "
                      "synthetic voices at three rates on direct PCM and two simulated rooms with +6 dB wet-utterance "
                      "RMS over the background. No threshold tuning. Separately perturb the exact recorded eight seconds "
                      "in player rate and digital gain. The original alert and all post hoc controls remain in the record."))


if __name__ == "__main__":
    main()
