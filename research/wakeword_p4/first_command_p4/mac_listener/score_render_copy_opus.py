"""Compare time-matched P4 Opus captures during electrical playback reference loss."""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

import check_render_copy_on_dropout as dropout
import score_calibrated_echo_opus as codec

ch, timing = dropout.ch, dropout.timing
OUT = ch.RESULTS / "4090-c161-render-copy-dropout-opus.json"
OFFSETS = {"stable_writer": 0, "writer_20ms_earlier": 320, "writer_20ms_later": -320}


def main():
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    original = ch.read(dropout.OUT)
    costs = ch.read(timing.names.OUT)
    assert ch.sha(dropout.COEFF) == original["coeff_sha256"]
    assert ch.sha(timing.names.COEFF) == costs["coeff_sha256"]
    archived = {(r["mic_sha256"], r["mode"]): r for r in original["records"]}
    fw = codec.uplink.firmware()
    decoder = codec.uplink.decoder_api()
    reader, identity = ch.load_reader()
    assert original["asr"] == identity
    cases, captures, cache = [], [], {}
    summary = defaultdict(lambda: defaultdict(Counter))
    archive = OUT.with_suffix(".zip")
    temp_zip = archive.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(temp_zip, "w", ZIP_DEFLATED) as logs, \
            np.load(dropout.COEFF, allow_pickle=False) as writes, \
            np.load(timing.names.COEFF, allow_pickle=False) as measured:
        work = Path(folder)
        codec.uplink.WORK = work
        for row, mic, ref, _ in dropout.prior.sources(costs):
            if row["role"] != "Hey Nyx":
                continue
            bot = row["bot"]
            mic_hash = ch.digest(mic)
            adc = dropout.prior.electrical(ref, row["start"], "reference_lost")
            physical = timing.names.estimate(adc, measured[bot])
            for mode, offset in OFFSETS.items():
                src = archived[mic_hash, mode]
                writer = dropout.writer(ref, row["start"], offset)
                assert src["write_sha256"] == ch.digest(writer) and src["adc_sha256"] == ch.digest(adc)
                prediction = timing.names.estimate(writer, writes[bot])
                signals, _ = dropout.fallback(mic, adc, physical, prediction, costs["cutoff_samples"])
                result = dict(mode=mode, voice=row["voice"], db=row["db"], mic_sha256=mic_hash, routes={})
                for name in ("electrical", "guarded"):
                    total = summary[mode][name]
                    total["trials"] += 1
                    previous = src["routes"][name]
                    pcm = signals[name]
                    assert previous["pcm_sha256"] == ch.digest(pcm)
                    result["routes"][name] = None
                    if not previous["hits"]:
                        continue
                    clock = previous["hits"][0]
                    first = max(0, ((clock-dropout.HOP+159)//160)*160-32000)
                    assert previous["first"] == first
                    end = row["end"]
                    residual_sha = ch.digest(pcm[first:end])
                    assert residual_sha == previous["capture_sha256"]
                    key = (mic_hash, first, end, residual_sha)
                    if key not in cache:
                        cap = codec.capture(work, logs, decoder, reader, mic, pcm, row, first, len(captures))
                        assert cap["lanes"]["residual"]["input_sha256"] == residual_sha
                        assert cap["lanes"]["raw"]["input_sha256"] == ch.digest(mic[first:end])
                        cache[key] = cap["index"]
                        captures.append(cap)
                    index = cache[key]
                    res = captures[index]["lanes"]["residual"]
                    timely = clock < row["command_at"]
                    total.update(opened=1, timely=timely, host_complete=previous["complete"],
                                 opus_complete=res["complete"], timely_complete=timely and res["complete"],
                                 raw_complete=captures[index]["lanes"]["raw"]["complete"])
                    result["routes"][name] = dict(clock=clock, capture=index, host_complete=previous["complete"])
                cases.append(result)
            print(len(cases), "conditions", len(captures), "captures", flush=True)
        retries = sorted(work.glob("*-attempt*.gz"))
        for path in retries:
            logs.write(path, "retry/"+path.name)
    assert len(cases) == 36
    with ZipFile(temp_zip) as logs:
        assert logs.testzip() is None
        for cap in captures:
            data = logs.read(cap["uart"])
            assert hashlib.sha256(data).hexdigest() == cap["uart_sha256"]
            text = data.decode().replace("\r\n", "\n")
            for lane, (name, info) in enumerate(cap["lanes"].items()):
                assert name == ("residual", "raw")[lane]
                packets = codec.uplink.packets(text, cap["index"], cap["count"], lane)
                assert len(packets) == info["packets"]
                assert ch.digest(codec.uplink.decode(decoder, packets)) == info["decoded_sha256"]
    temp_zip.replace(archive)
    ch.save(OUT, dict(records=cases, captures=captures, summary=summary, asr=identity,
                      archive_sha256=ch.sha(archive), firmware=fw, retries=[p.name for p in retries],
                      sources={p.name: ch.sha(p) for p in (dropout.OUT, dropout.COEFF, timing.names.OUT,
                               timing.names.COEFF, Path(codec.__file__), Path(dropout.__file__), Path(__file__))},
                      policy="Previously exposed four-voice lock commands at three levels, active panel playback, "
                      "electrical reference loss, fixed write lead or either 20 ms jump. Reuse each reported "
                      "OPEN and host PCM interval. Frozen P4 Opus encoder in QEMU, independent decoding twice "
                      "and unprompted small.en. Processing delay and real ADC/DMA, VAD and transport unmeasured."))


if __name__ == "__main__":
    main()
