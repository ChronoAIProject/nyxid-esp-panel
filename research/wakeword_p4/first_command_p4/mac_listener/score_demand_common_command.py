"""Compare Nyx and Willow at two identical command levels in each 0 dB room."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import soundfile as sf

import score_fixed_rearm_demand_peer as peer

P = peer.P
WORK = peer.WORK / "common-command"
OUT = peer.OUT.with_stem("4090-demand-common-command")
ARCHIVE = WORK / OUT.with_suffix(".zip").name
PROTOCOL = peer.PROTOCOL.with_name("fixed_rearm_demand_common_command.md")
LICENSE = ("The playback WAVs use DEMAND channel 1 by Joachim Thiemann, Nobutaka Ito and Emmanuel Vincent, "
           "https://doi.org/10.5281/zenodo.1227121. The generated address and spoken command were mixed with "
           "the original background. The adapted WAVs are distributed under CC BY-SA 3.0 Unported, "
           "https://creativecommons.org/licenses/by-sa/3.0/.")


def whole_sources(history):
    records = [r for r in history["records"] if r["variant"] == "whole"]
    assert len(records) == 12 and P.sha(peer.PACK) == history["whole_playback_sha256"]
    sources = {}
    with ZipFile(peer.PACK) as archive:
        for row in records:
            data = archive.read(row["label"] + ".wav")
            assert sha256(data).hexdigest() == row["sha256"]
            pcm, rate = sf.read(BytesIO(data), dtype="int16")
            at = row["command_at"]
            assert rate == 16000 and at == 44800 and P.digest(pcm) == row["pcm_sha256"]
            assert P.digest(pcm[:at]) == row["prefix_sha256"]
            sources[row["scene"], row["role"]] = row, pcm
    assert len(sources) == len(records)
    return sources


def planned_trials(sources):
    for scene in sorted({scene for scene, role in sources}):
        for level, donor in (("lower", "nyx"), ("higher", "willow")):
            suffix_record, suffix_pcm = sources[scene, donor]
            at = suffix_record["command_at"]
            for role in ("nyx", "willow"):
                prefix_record, prefix_pcm = sources[scene, role]
                mixed = np.concatenate((prefix_pcm[:at], suffix_pcm[at:]))
                row = dict(label=f"{scene}-{role}-{level}", scene=scene, role=role, level=level,
                           command_at=at, pcm_sha256=P.digest(mixed), prefix_sha256=P.digest(mixed[:at]),
                           suffix_sha256=P.digest(mixed[at:]), prefix_from=prefix_record["label"],
                           suffix_from=suffix_record["label"], command_gain=suffix_record["command_gain"],
                           active_command_snr_db=suffix_record["active_command_snr_db"],
                           timely_open=prefix_record["timely_open"], reused=role == donor)
                assert row["prefix_sha256"] == prefix_record["prefix_sha256"]
                if row["reused"]:
                    assert row["pcm_sha256"] == prefix_record["pcm_sha256"]
                    if row["timely_open"]:
                        prior = prefix_record["uplink"]
                        assert prior["opened"] == row["timely_open"][0] and prior["pcm_sha256"] == row["pcm_sha256"]
                        row["readings"] = prior["readings"]
                yield row, mixed


def counts(rows):
    common = {r["scene"] for r in rows if r["role"] == "willow" and r["timely_open"]}
    result = []
    for group in ("all", "both_open"):
        for level in ("lower", "higher"):
            for role in ("nyx", "willow"):
                selected = [r for r in rows if (r["level"], r["role"]) == (level, role)
                            and (group == "all" or r["scene"] in common)]
                live = [r for r in selected if r["timely_open"]]
                if group == "both_open":
                    assert len(live) == len(selected)
                scores = {schedule: {reader: sum(r["readings"][schedule][reader][metric] for r in live)
                                     for reader, metric in (("qwen", "complete"), ("turbo", "complete_suffix"))}
                          for schedule in ("immediate", "waited")}
                result.append(dict(group=group, level=level, role=role, scenes=len(selected),
                                   timely=len(live), exact_suffix=scores))
    return result


def main():
    WORK.mkdir(exist_ok=True)
    history = P.read(peer.OUT)
    assert P.sha(peer.TRACE) == history["archive_sha256"]
    sources = whole_sources(history)
    clean, command = peer.clean_command()
    scoring = peer.scoring
    scoring.sender.BUILD = peer.peer.prior.SENDER
    images = [scoring.clock.BUILD/"nyx_mac_listener.bin", scoring.prior.CODEC/"nyx_first_command.bin",
              scoring.sender.BUILD/"board.bin"]
    assert all(P.sha(image) == history["sources"][str(image)] for image in images)
    assert P.sha(P.QEMU) == history["qemu_sha256"]
    reference = peer.WORK / "reference.wav"
    ref, rate = sf.read(reference, dtype="int16")
    assert rate == 16000 and len(ref) == 85200 and not ref.any()
    proc, model, qwen = scoring.prior.load()
    assert qwen == history["qwen"] and "4090" in qwen["device"]
    decoder = scoring.prior.decoder_api()
    assert decoder.opus_get_version_string().decode() == history["decoder"]
    readers = proc, model, scoring.prior.reader(history["turbo"]), decoder
    cache = WORK / (P.sha(peer.OUT)[:12] + "-" + P.sha(PROTOCOL)[:12])
    cache.mkdir(exist_ok=True)
    scoring.clock.reference.WORK = cache
    records, files = [], {}
    with TemporaryDirectory(dir=WORK) as directory:
        tmp = Path(directory)
        with ZipFile(tmp/ARCHIVE.name, "w", ZIP_DEFLATED) as archive:
            for i, (row, mixed) in enumerate(planned_trials(sources)):
                records.append(row)
                if row["reused"]:
                    continue
                wav = tmp / (row["label"] + ".wav")
                sf.write(wav, mixed, rate, subtype="PCM_16")
                files[wav.name] = P.sha(wav)
                archive.writestr(ZipInfo(wav.name), wav.read_bytes(), compress_type=ZIP_DEFLATED)
                if not row["timely_open"]:
                    continue
                opened, = row["timely_open"]
                case = dict(row, path=str(wav), command=command, reference_path=str(reference), reference_sha256=P.digest(ref))
                uplink, shard = scoring.score_case(case, i, opened, readers, tmp, cache)
                row["windows"] = peer.check_sends(shard, row["label"], i, uplink, decoder, clean, tmp)
                row["uplink"] = {k:v for k,v in uplink.items() if k != "readings"}
                row["readings"] = uplink["readings"]
                row["native_index"] = i
                with ZipFile(shard) as saved:
                    for name in saved.namelist():
                        if name.endswith(".log"):
                            data = saved.read(name)
                            files[name] = sha256(data).hexdigest()
                            archive.writestr(ZipInfo(name), data, compress_type=ZIP_DEFLATED)
                print(row["label"], uplink["packets"], flush=True)
            manifest = dict(command=command, sample_rate=rate, playback_license=LICENSE, sha256=files,
                            inputs=[{k:r[k] for k in ("label", "command_at", "prefix_from", "suffix_from")}
                                    for r in records if not r["reused"]])
            data = (json.dumps(manifest, indent=2) + "\n").encode()
            archive.writestr(ZipInfo("manifest.json"), data, compress_type=ZIP_DEFLATED)
            files["manifest.json"] = sha256(data).hexdigest()
        assert len(records) == 24 and len(files) == 12 + 9 * 3 + 1
        for i in range(0,len(records),2):
            assert records[i]["suffix_sha256"] == records[i+1]["suffix_sha256"]
        with ZipFile(tmp/ARCHIVE.name) as saved:
            assert len(saved.namelist()) == len(files) and set(saved.namelist()) == set(files)
            assert all(sha256(saved.read(name)).hexdigest() == digest for name,digest in files.items())
        deps = [PROTOCOL, Path(__file__).resolve().relative_to(Path.cwd()), peer.OUT, peer.PACK, *images]
        P.save(tmp/OUT.name, dict(records=records, summary=counts(records), command=command, qwen=qwen,
               turbo=history["turbo"], decoder=history["decoder"], qemu_sha256=history["qemu_sha256"],
               archive=ARCHIVE.name, archive_sha256=P.sha(tmp/ARCHIVE.name), files=files,
               original_uart_sha256=history["archive_sha256"], sources={str(p):P.sha(p) for p in deps},
               playback_license=LICENSE))
        (tmp/ARCHIVE.name).replace(ARCHIVE)
        (tmp/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
