"""Apply the three original v5 markers to whole-sentence synthesis at its existing command clock."""
from collections import Counter, defaultdict
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.signal import fftconvolve
import soundfile as sf

import check_connected_panel_rooms as study

tx, ch, panel = study.tx, study.ch, study.panel
OUT = ch.RESULTS / "4090-c161-marked-connected-panel.json"
PLAY = ch.RESULTS / "c161-marked-connected-playback.zip"


def master(pcm):
    # A file renderer can inspect future samples. Smooth the local limiting gain
    # over +/-5 ms. Every contributing minimum includes the sample being scaled.
    x = pcm.astype(float) * 10**(6/20)
    cap = 32768 * 10**(-1/20)
    gain = np.minimum(1., cap/np.maximum(1., np.abs(x)))
    safe = minimum_filter1d(gain, size=161, mode="nearest")
    smooth = np.convolve(np.pad(safe, 80, mode="edge"), np.ones(161)/161, mode="valid")
    result = np.rint(x*smooth)
    assert result.shape == pcm.shape and np.abs(result).max() <= np.ceil(cap)
    return result.astype("<i2")


def render(group, v5, spec, markers):
    row, plain = group["Hey Nyx"]
    end = row["end"]
    tracks = {}
    for name in ("marked", "reversed_final"):
        x = plain.astype(np.int32)
        for i, (pos, code) in enumerate(zip(spec["code_positions"], markers, strict=True)):
            start = end - len(v5) + pos
            assert row["start"] <= start and start + len(code) <= end
            x[start:start+len(code)] += code[::-1] if i == 2 and name == "reversed_final" else code
        assert np.abs(x).max() <= 32767
        tracks[name] = x.astype("<i2")
        np.testing.assert_array_equal(tracks[name][end:], plain[end:])
    louder = tracks["marked"].copy()
    louder[end:] = master(plain[end:])
    tracks["marked_louder_command"] = louder
    return tracks


def save_speakers(z, row, tracks):
    entries = {}
    for name, pcm in tracks.items():
        member = f'{row["index"]:03}/{name}.wav'
        buf = BytesIO()
        sf.write(buf, pcm, 16000, format="WAV", subtype="PCM_16")
        wav = buf.getvalue()
        z.writestr(member, wav)
        entries[name] = dict(file=member, bytes=len(wav), sha256=study.sha256(wav).hexdigest(),
                             pcm_sha256=ch.digest(pcm), suffix_sha256=ch.digest(pcm[row["end"]:]))
    return dict(index=row["index"], voice=row["voice"], command=row["carrier"], start=row["start"],
                command_at=row["end"], source_sha256=row["sha256"], tracks=entries)


def main():
    spec, v5, _, markers = tx.study.grid.fit.original()
    groups = study.sources()
    base = ch.read(study.OUT)
    inputs = { (r["source"]["index"], r["source"]["external_db"]): r
              for r in base["records"] if r["source"]["view"] == "both" }
    history = ch.read(panel.OUT)
    quiet, sr = sf.read(panel.demo.WAVE, dtype="int16")
    assert sr == 16000 and ch.sha(panel.demo.WAVE) == base["sources"][panel.demo.WAVE.name]
    bed, _, bed_sha = tx.study.grid.room.rooms.read_noise("music")
    small, identity = ch.load_reader()
    assert identity == history["small"]
    turbo = tx.diag.reader(history["turbo"])
    decoder = tx.panel.grid.gap.uplink.decoder_api()
    builds = dict(bca=panel.middle.WORK / "build")
    assert ch.sha(builds["bca"] / "nyx_mac_listener.bin") == history["firmware"]
    assert tx.panel.grid.gap.uplink.firmware() == history["codec"]
    rows, guests, speakers, summary = [], [], [], defaultdict(Counter)
    tmp, pack_tmp = OUT.with_suffix(".tmp"), PLAY.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(tmp, "w", ZIP_DEFLATED) as z, ZipFile(pack_tmp, "w", ZIP_DEFLATED) as pack:
        work = Path(folder)
        matcher = tx.single.timed.binding(work)
        native = tx.panel.grid.room.rooms.c161_binding()
        for (voice, command), group in groups.items():
            row, plain = group["Hey Nyx"]
            fixed_row, fixed = group["fixed-v5"]
            tracks = render(group, v5, spec, markers)
            speakers.append(save_speakers(pack, row, tracks))
            for db in study.LEVELS:
                ordinary = inputs[row["index"], db]
                s = ordinary["source"]
                (hp, _, ph), (hx, _, eh) = tx.diag.room(s["geometry"])
                assert (ph,eh,bed_sha) == (s["panel_rir_sha256"],s["external_rir_sha256"],s["music_sha256"])
                local = fftconvolve(quiet.astype(float), hp)
                length = max(max(len(x) for _,x in group.values())+len(hx)-1, s["panel_at"]+len(local)) + 4800
                bg = bed[(s["music_offset"] + np.arange(length)) % len(bed)]
                ref = fftconvolve(v5.astype(float),hp)[:len(v5)]
                mix = bg * (0.35*np.linalg.norm(ref)/np.sqrt(len(ref))/np.sqrt(np.mean(bg**2)))
                mix[s["panel_at"]:s["panel_at"]+len(local)] += local

                def scene(wave):
                    wet = fftconvolve(wave.astype(float), hx)
                    signal = (mix + np.pad(wet, (0, length-len(wet))) * 10**(db/20)) * s["preamp_gain"]
                    rounded = np.rint(signal)
                    clipped = int(np.count_nonzero((rounded < -32768) | (rounded > 32767)))
                    return tx.study.grid.room.quantize(signal), clipped

                assert ch.digest(scene(plain)[0]) == ordinary["result"]["pcm_sha256"]
                assert ch.digest(scene(fixed)[0]) == inputs[fixed_row["index"],db]["result"]["pcm_sha256"]
                before_command = None
                for name, wave in tracks.items():
                    mic, clipped = scene(wave)
                    if name == "marked":
                        before_command = mic[:s["command_at"]]
                    if name == "marked_louder_command":
                        np.testing.assert_array_equal(before_command, mic[:s["command_at"]])
                    zero = np.zeros_like(mic)
                    clock = s["transmit_event"]
                    raw = tx.panel.grid.fit.fixed.ch.routes(matcher, native, mic)
                    result = panel.middle.pair.admit(raw, clock, panel.middle.single.hits(mic,clock), zero)
                    timely = study.tally(summary[name], s, result)
                    rows.append(dict(source=s, variant=name, result=result, timely=timely, clipped_samples=clipped))
                    # Replay every assigned family and every external gain for the louder commands.
                    vi = study.connected.VOICES.index(voice)
                    ci = study.connected.CARRIERS.index(command)
                    if db != study.LEVELS[(vi+ci)%3] and name != "marked_louder_command":
                        continue
                    case = (len(rows)-1, s, "bca", "both", mic, zero, result)
                    guest = tx.panel.capture(work, z, len(guests), case, small, turbo, decoder, builds, tx_pair=clock)
                    guests.append(dict(guest, variant=name, timely=timely))
            print("families",len(speakers),"P4",len(guests),flush=True)
        manifest = dict(sample_rate=16000, format="mono PCM16", records=speakers, source_plan_sha256=ch.sha(study.connected.PLAN),
            source_pack_sha256=ch.sha(study.connected.PACK), panel_wav=panel.demo.WAVE.name,
            panel_wav_sha256=ch.sha(panel.demo.WAVE), scene_report=OUT.name,
            instructions="Play the complete external file including leading silence, with unchanged hardware "
            "gain and position. Start the quiet panel WAV at panel_at samples after external file start, "
            "using the row for the same index and external gain. All clocks refer to 16 kHz.")
        pack.writestr("manifest.json",json.dumps(manifest,indent=2)+"\n")
    assert (len(rows),len(guests),len(speakers)) == (108,60,12)
    for partial, final in ((pack_tmp, PLAY),(tmp, OUT.with_suffix(".zip"))):
        with ZipFile(partial) as z:
            assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist()))
        partial.replace(final)
    ch.save(OUT, dict(records=rows, guest=guests, summary=summary, model=history["model"], firmware=history["firmware"],
        codec=history["codec"], small=identity, turbo=history["turbo"], archive_sha256=ch.sha(OUT.with_suffix(".zip")),
        speakers_sha256=ch.sha(PLAY), sources={p.name:ch.sha(p) for p in (study.OUT,panel.demo.WAVE,study.connected.PLAN,
        study.connected.PACK,tx.study.grid.fit.fixed.SOURCE_INFO,Path(study.__file__),Path(__file__))},
        policy="Add the original v5 code waveforms 440, 280 and 160 ms before the whole sentence's command; "
        "keep the utterance, command and pauses sample-aligned. Reverse only the last marker for the sham. "
        "Third view raises only the command by 6 dB with a -1 dBFS smoothed peak limiter. Scene gain and "
        "music fixed by the paired experiment. Verify uncoded and fixed microphone hashes and the entire "
        "pre-command mic of the two marked positives. Development replay, no new FAR exposure."))


if __name__ == "__main__":
    main()
