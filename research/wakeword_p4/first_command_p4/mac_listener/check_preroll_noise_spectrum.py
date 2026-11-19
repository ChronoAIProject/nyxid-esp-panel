"""Check a fixed noise spectrum from the received pre-roll against complete, unprompted commands."""
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy import signal

import check_panel_parakeet as reference
import complete_quieter_panel_matrix as panel

ch, tx = panel.ch, panel.tx
OUT = ch.RESULTS / "4090-c161-preroll-noise-spectrum.json"


def attenuate(pcm):
    # Estimate noise on the first half second available to the receiver. A frame
    # includes 400 samples. Exclude any frame that reaches outside that interval.
    fs = 16000
    _, times, z = signal.stft(pcm.astype(np.float64), fs=fs, nperseg=400, noverlap=240, nfft=512)
    inside = (times * fs >= 200) & (times * fs <= 7800)
    assert inside.any()
    power = abs(z) ** 2
    profile = np.mean(power[:, inside], axis=1, keepdims=True)
    # Power spectral subtraction with a -12 dB amplitude floor.
    gain = np.sqrt(np.maximum(10**(-12/10), 1 - profile / np.maximum(power, 1e-12)))
    _, y = signal.istft(z * gain, fs=fs, nperseg=400, noverlap=240, nfft=512)
    clipped = np.count_nonzero(abs(y[:len(pcm)]) > 32767)
    return tx.study.grid.room.quantize(y[:len(pcm)]), int(clipped)


def main():
    current = ch.read(panel.OUT)
    small, identity = ch.load_reader()
    assert identity == current["small"]
    readers = dict(small=small, turbo=tx.diag.reader(current["turbo"]))
    uplink = tx.panel.grid.gap.uplink
    decoder = uplink.decoder_api()
    sources, rows, summary = {}, [], defaultdict(Counter)
    with ExitStack() as stack:
        def archive(name, digest):
            path = ch.RESULTS / name
            assert ch.sha(path) == digest
            sources[name] = digest
            return stack.enter_context(ZipFile(path))

        def evaluate(group, index, command, before, audio=None):
            stats = summary[group]
            stats.update(trials=1, opened=int(audio is not None))
            entry = dict(group=group, index=index, command=command, before=before)
            if audio is None:
                rows.append(entry)
                return
            logs, member, idx, samples, digest = audio
            uart = logs.read(member).decode().replace("\r\n", "\n")
            packets = uplink.packets(uart, idx, samples, 0)
            pcm = uplink.decode(decoder, packets)
            assert ch.digest(pcm) == digest
            filtered, clipped = attenuate(pcm)
            assert len(filtered) == len(pcm)
            entry.update(pcm_sha256=digest, filtered_sha256=ch.digest(filtered), clipped=clipped, readings={})
            for tag, recognizer in readers.items():
                result = uplink.recognition(recognizer, filtered, command)
                after = result["complete_suffix"]
                entry["readings"][tag] = result
                stats.update({tag+"_before": int(before[tag]), tag+"_after": int(after),
                              tag+"_gained": int(after and not before[tag]), tag+"_lost": int(before[tag] and not after)})
            rows.append(entry)

        logs = {n: archive(n, h) for n,h in current["archives"].items()}
        for row in current["records"]:
            cap = row["uplink"]
            audio = (logs[row["archive"]], f'{row["index"]:02}-opus.log', row["index"], cap["samples"], cap["pcm_sha256"])
            evaluate(f'panel.{row["source"]["external_db"]}', row["host_index"], row["source"]["command"],
                     {k:row[k] for k in readers}, audio)
        sources[panel.OUT.name] = ch.sha(panel.OUT)
        for src in (reference.timed.OUT, reference.ROOM, reference.MUSIC):
            data = ch.read(src)
            sources[src.name] = ch.sha(src)
            z = archive(src.with_suffix(".zip").name, data["archive_sha256"])
            cases = (reference.ordinary_inputs(data) if src == reference.timed.OUT
                     else reference.willow_inputs(data, src == reference.MUSIC))
            for row, item in cases:
                audio = None
                if item:
                    idx, samples, digest, member = item
                    audio = z, member, idx, samples, digest
                evaluate(row["group"], row["case"], row["command"], {k:row[k] for k in readers}, audio)
    assert len(rows) == 114
    ch.save(OUT, dict(records=rows, summary=summary, sources={**sources, Path(__file__).name: ch.sha(Path(__file__))},
        small=identity, turbo=current["turbo"], policy="Complete selected Opus, original opening and command scores. "
        "Estimate per-frequency noise power from frames wholly within the initial 0.5 s of each received stream. "
        "Apply fixed power subtraction with -12 dB floor and preserve sample length. No oracle noise, command "
        "start, transcript, loudness normalization or source separation. No new wake-word exposure."))


if __name__ == "__main__":
    main()
