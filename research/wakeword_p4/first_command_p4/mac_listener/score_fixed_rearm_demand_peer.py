"""Read complete peer sends after the level-matched indoor calls."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import soundfile as sf

import fixed_rearm_demand_levels as levels
import measure_command_capture as matching
import score_eight_local_peer as peer

P, scoring = peer.P, peer.scoring
WORK = levels.WORK / "peer"
OUT = levels.OUT.with_stem("4090-fixed-rearm-demand-peer")
PACK = OUT.with_name("hye-nyx-willow-indoor-whole-speaker-v1.zip")
TRACE = WORK / OUT.with_suffix(".zip").name
PROTOCOL = levels.PROTOCOL.with_name("fixed_rearm_demand_peer_protocol.md")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())


def clean_command():
    long = levels.long
    assert P.sha(long.QUIET) == P.read(long.QUIET_REPORT)["audio_sha256"]
    with ZipFile(long.QUIET) as source:
        meta = json.loads(source.read("manifest.json"))
        raw = source.read("erased.wav")
        assert sha256(raw).hexdigest() == meta["files"]["erased.wav"]["sha256"]
        pcm, rate = sf.read(BytesIO(raw), dtype="int16")
    assert rate == 16000 and not pcm[:meta["command_at"]].any()
    return pcm[meta["command_at"]:], meta["command"]


def prepare(clean):
    long = levels.long
    plan = P.read(levels.PLAN)
    wake = P.read(levels.OUT)
    assert P.sha(levels.PLAN) == wake["plan_sha256"]
    assert P.sha(levels.PROGRAMS) == plan["files"][str(levels.PROGRAMS)]
    long_rows = {r["label"]:r for r in P.read(long.PLAN)["records"]}
    rows = []
    with ZipFile(levels.PROGRAMS) as inputs, ZipFile(WORK / PACK.name, "w") as output:
        for r in wake["records"]:
            raw = inputs.read(Path(r["path"]).name)
            assert sha256(raw).hexdigest() == r["sha256"]
            pcm, rate = sf.read(BytesIO(raw), dtype="int16")
            assert rate == 16000 and P.digest(pcm) == r["pcm_sha256"]
            base = long_rows[r["scene"]]
            restored, sr = sf.read(long.WORK / (r["scene"] + ".wav"), dtype="int16")
            assert sr == rate and P.digest(restored) == base["background_pcm_sha256"]
            noise = restored[-len(clean):].astype(np.int32)
            pos = r["command"]
            np.testing.assert_array_equal(pcm[pos:], clean.astype(np.int32)+noise)
            assert r["timely"]["fixed"] == r["timely"]["c161"]
            event = r["timely"]["willow"] if r["role"] == "willow" else r["timely"]["fixed"]
            if r["role"] == "erased":
                assert not any(r["timely"].values())
            base_path = Path(r["path"])
            assert P.sha(base_path) == r["sha256"]
            variants = [("issued", pcm, 1., base_path)]
            if r["mode"] == "zero-snr" and r["role"] != "erased":
                new = pcm.copy()
                scaled = np.rint(clean.astype(float)*r["gain"]).astype(np.int32)+noise
                assert np.all((-32768 <= scaled) & (scaled <= 32767))
                new[pos:] = scaled
                assert np.array_equal(new[:pos], pcm[:pos])
                path = WORK / (r["label"] + "-whole.wav")
                sf.write(path, new, rate, subtype="PCM_16")
                output.writestr(ZipInfo(path.name), path.read_bytes(), compress_type=ZIP_DEFLATED)
                variants.append(("whole", new, r["gain"], path))
            for variant, wave, gain, path in variants:
                # Use the same clean-speech active samples for both gain measurements.
                active = np.abs(clean.astype(np.int32)) > np.abs(clean.astype(np.int32)).max()*10**(-35/20)
                signal = wave[pos:].astype(np.int32)-noise
                snr = long.rms_dbfs(signal[active])-long.rms_dbfs(noise[active])
                rows.append(dict(label=r["label"]+"-"+variant, scene=r["scene"], role=r["role"], mode=r["mode"],
                                 variant=variant, path=str(path), sha256=P.sha(path), pcm_sha256=P.digest(wave),
                                 prefix_sha256=P.digest(wave[:pos]), measured_input_sha256=r["pcm_sha256"],
                                 timely_open=event, command_at=pos, command_gain=gain, active_command_snr_db=snr))
        packed = [r for r in rows if r["variant"] == "whole"]
        info = dict(records=[dict(file=Path(r["path"]).name, **{k:v for k,v in r.items() if k!="path"}) for r in packed],
                    license="CC BY-SA 3.0 Unported", authors="Joachim Thiemann, Nobutaka Ito and Emmanuel Vincent",
                    source="https://doi.org/10.5281/zenodo.1227121",
                    modifications="Channel 1 noise under a generated address and command with constant source gain.")
        output.writestr(ZipInfo("manifest.json"), json.dumps(info, indent=2)+"\n", compress_type=ZIP_DEFLATED)
    assert len(rows) == 42 and len(packed) == 12
    (WORK / PACK.name).replace(PACK)
    return rows


def check_sends(shard, name, index, uplink, decoder, clean, work):
    with ZipFile(shard) as z:
        uart = z.read(name + "/opus.log")
        sender = work / "readback.log"
        sender.write_bytes(z.read(name + "/sender.log"))
    assert sha256(uart).hexdigest() == uplink["opus_uart_sha256"] and P.sha(sender) == uplink["sender_uart_sha256"]
    count = uplink["samples"]
    packets = scoring.prior.packets(uart.decode().replace("\r\n","\n"), index, count, 0)
    sends = scoring.sender.compare(sender,index,uplink["opened"],uplink["first"],count,packets)
    matches = {}
    for schedule, (frames, meta) in zip(("immediate","waited"), sends, strict=True):
        pcm = scoring.prior.decode(decoder, frames)
        assert P.digest(pcm) == uplink["readings"][schedule]["qwen"]["pcm_sha256"]
        matches[schedule] = matching.locate(clean, pcm)
    return matches


def main():
    WORK.mkdir(exist_ok=True)
    clean, command = clean_command()
    rows = prepare(clean)
    history = P.read(peer.OUT)
    scoring.sender.BUILD = peer.prior.SENDER
    images = (scoring.clock.BUILD/"nyx_mac_listener.bin", scoring.prior.CODEC/"nyx_first_command.bin",
              scoring.sender.BUILD/"board.bin")
    assert all(history["sources"][str(p)] == P.sha(p) for p in images)
    proc, model, qid = scoring.prior.load()
    assert qid == history["qwen"] and "4090" in qid["device"]
    decoder = scoring.prior.decoder_api()
    assert decoder.opus_get_version_string().decode() == history["decoder"]
    readers = proc,model,scoring.prior.reader(history["turbo"]),decoder
    ref = WORK / "reference.wav"
    zeros = np.zeros(85200, dtype="<i2")
    sf.write(ref, zeros, 16000, subtype="PCM_16")
    cache = WORK / ("runs-"+P.sha(levels.OUT)[:12]+"-"+P.sha(PACK)[:12])
    cache.mkdir(exist_ok=True)
    scoring.clock.reference.WORK = cache
    files = {}
    with TemporaryDirectory(dir=WORK) as tmp:
        work = Path(tmp)
        with ZipFile(work/TRACE.name,"w",ZIP_DEFLATED) as logs:
            for i,r in enumerate(rows):
                if not r["timely_open"]:
                    continue
                record = dict(r, reference_path=str(ref), reference_sha256=P.digest(zeros), command=command)
                opened, = r["timely_open"]
                observation, shard = scoring.score_case(record,i,opened,readers,work,cache)
                r["uplink"] = observation
                r["windows"] = check_sends(shard,r["label"],i,observation,decoder,clean,work)
                with ZipFile(shard) as saved:
                    for name in saved.namelist():
                        if name.endswith(".log"):
                            raw = saved.read(name)
                            assert name not in files
                            files[name] = sha256(raw).hexdigest()
                            logs.writestr(name,raw)
                print(i, r["label"], r["windows"]["immediate"]["status"],flush=True)
        with ZipFile(work/TRACE.name) as logs:
            assert len(logs.namelist()) == len(files) and not logs.testzip()
            assert all(sha256(logs.read(k)).hexdigest()==v for k,v in files.items())
        deps = [SCRIPT, PROTOCOL, peer.OUT, levels.OUT, levels.PLAN, levels.long.QUIET_REPORT,
                Path(matching.__file__).resolve().relative_to(Path.cwd()),*images]
        P.save(work/OUT.name,dict(records=rows,files=files,sources={str(p):P.sha(p) for p in deps},
               qwen=qid,turbo=history["turbo"],decoder=history["decoder"],qemu_sha256=P.sha(P.QEMU),
               archive=TRACE.name,archive_sha256=P.sha(work/TRACE.name),whole_playback_sha256=P.sha(PACK)))
        (work/TRACE.name).replace(TRACE)
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
