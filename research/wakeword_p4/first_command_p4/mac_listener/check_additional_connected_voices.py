"""Test the frozen marked connected-call recipe on a second set of speakers and commands."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import check_connected_uplink_prefix as prefix
import train_c161_onset_names as onset

timing, ch, playback = prefix.timing, prefix.ch, prefix.playback
mixed, tx = timing.mixed, timing.tx
STEM = "4090-c161-additional-connected-voices"
PACK = ch.RESULTS / "c161-additional-connected-voices.zip"
OUT = ch.RESULTS / (STEM + ".json")
GUEST = ch.RESULTS / (STEM + "-p4.json")
VOICES = ("af_bella", "am_echo", "bf_isabella", "bm_lewis")
COMMANDS = ("is the kitchen window open?", "please turn on the porch lights.", "lower the bedroom speaker volume.")


def prepare():
    spec, v5, _, markers = tx.study.grid.fit.original()
    assert set(VOICES).isdisjoint(mixed.study.connected.VOICES)
    assert set(COMMANDS).isdisjoint(mixed.study.connected.CARRIERS)
    inventory = ch.RESULTS / "4090-hey-common-carrier-kokoro-generation-execution-v1.json"
    tx_in_wave = tx.event_time()
    rows = []
    with TemporaryDirectory(dir=OUT.parent) as folder:
        dest = Path(folder)/PACK.name
        with ZipFile(dest, "w", ZIP_DEFLATED) as z, patch.multiple(
                onset, VOICES=VOICES, CARRIERS=COMMANDS, ADDRESSES=("Hey Nyx",), LABELS=(0,)):
            for s, pcm in onset.synthesize():
                words = list(re.finditer(r"[^ ,.!?]+", s["phones"]))
                edges = np.cumsum(s["duration"], dtype=int)*400 + 32000
                pos = words[2].start()
                while s["phones"][pos] in "ˈˌ":
                    pos += 1
                start, end = int(edges[words[0].start()]), int(edges[pos])
                wave = np.pad(pcm, (32000,16000))
                row = dict(index=len(rows), voice=s["voice"], carrier=s["carrier"], start=start, end=end,
                           sha256=ch.digest(wave))
                group = {"Hey Nyx":(row,wave)}
                adjusted, gain = tx.balanced.match_level(v5,wave[end:])
                command = mixed.master(adjusted)
                plain = np.r_[wave[:end], command]
                removed = plain.copy()
                removed[:end] = 0
                tracks = dict(plain=plain, no_address=removed)
                first_marker = end-len(v5)+spec["code_positions"][0]
                if first_marker >= start:
                    marked = mixed.render(group, v5, spec, markers)["marked"]
                    tracks["marked"] = np.r_[marked[:end], command]
                saved = mixed.save_speakers(z,row,tracks)
                saved.update(synthesis=s, gain=gain, panel_at=end-1280-tx_in_wave, transmit_event=end-1280,
                             fits_marker=first_marker>=start, first_marker=first_marker)
                assert all(np.array_equal(x[end:],command) for x in tracks.values())
                rows.append(saved)
            assert len(rows) == len(VOICES)*len(COMMANDS)
            z.write(mixed.panel.demo.WAVE, "panel-minus6.wav")
            z.writestr("manifest.json",json.dumps(dict(records=rows, rate=16000, format="mono PCM16",
                panel_file="panel-minus6.wav",panel_sha256=ch.sha(mixed.panel.demo.WAVE),
                training_voice_inventory_sha256=ch.sha(inventory), synthesis_script_sha256=ch.sha(Path(onset.__file__)),
                generation_files=ch.read(inventory)["files"], script_sha256=ch.sha(Path(__file__)),
                policy="One connected synthesis per voice and command using training-pool voices and historical "
                "speed and punctuation schedule. Plain, marked and removed-address tracks share the exact same "
                "balanced command samples. Mark a case unavailable for this recipe if its annotated name is "
                "shorter than the first marker offset. Panel TX is 80 ms before command. Times in 16 kHz samples "
                "from the start of the complete external file."),indent=2)+"\n")
        with ZipFile(dest) as z:
            assert z.testzip() is None
        dest.replace(PACK)


def scenes():
    v5, _ = tx.study.panel_wave()
    panel, rate = sf.read(mixed.panel.demo.WAVE,dtype="int16")
    music, _, music_id = tx.study.grid.room.rooms.read_noise("music")
    gain = ch.read(playback.headroom.OUT)["fixed_preamp_gain"]
    with ZipFile(PACK) as z:
        plan = json.loads(z.read("manifest.json"))
        assert rate == 16000 and ch.sha(mixed.panel.demo.WAVE) == plan["panel_sha256"]
        for entry in plan["records"]:
            index = entry["index"]
            vi, ci = divmod(index,len(COMMANDS))
            assert (entry["voice"],entry["command"]) == (VOICES[vi],COMMANDS[ci])
            geometry = (vi+ci) % len(tx.study.GEOMETRIES)
            (_, _, ph), (_, _, eh) = tx.diag.room(geometry)
            tracks = {k:playback.read_wav(z,item) for k,item in entry["tracks"].items()}
            group = {k:(entry,wave) for k,wave in tracks.items()}
            for db in mixed.study.LEVELS:
                src = dict(index=index, voice=entry["voice"], command=entry["command"], start=entry["start"],
                           command_at=entry["command_at"], external_db=db, preamp_gain=gain,
                           transmit_event=entry["transmit_event"], panel_at=entry["panel_at"], geometry=geometry,
                           panel_rir_sha256=ph, external_rir_sha256=eh, music_sha256=music_id,
                           music_offset=(39413*(12+index))%len(music))
                scene = mixed.room_scene(src,group,panel,v5,music,music_id)
                for name, wave in tracks.items():
                    mic, clipped = scene(wave)
                    yield src, name, mic, clipped


def host():
    counts, rows = defaultdict(Counter), []
    native = tx.panel.grid.room.rooms.c161_binding()
    aux = timing.overlap
    models, ids = aux.load_models()
    five, five_ids = aux.verifier.models()
    history = ch.read(aux.OUT)
    assert ids == history["models"] and five_ids["trained"] == history["five_sha256"]
    front = aux.fit.prior.letters.native_frontend()
    with TemporaryDirectory(dir=OUT.parent) as folder:
        work = Path(folder)
        matcher = tx.single.timed.binding(work)
        for s, name, mic, clipped in scenes():
            result = playback.host_result(matcher, native, mic, s["transmit_event"])
            mixed.study.tally(counts[name], s, result)
            row = dict(source=s, track=name, result=result, clipped=clipped)
            if name in ("plain","no_address"):
                _, _, events, checks = aux.traces({"overlap":models["overlap"]},front,mic,five["trained"],history["cutoff"])
                echoes = mixed.panel.middle.single.hits(mic,s["transmit_event"])
                for arm in ("overlap","confirm_0"):
                    raw = dict(result, c161=sorted(set(result["c161"]+events[arm])))
                    decision = mixed.panel.middle.pair.admit(raw,s["transmit_event"],echoes,np.zeros_like(mic))
                    row[arm] = dict(proposals=events[arm],opened=decision["opened"])
                    mixed.study.tally(counts[name+"."+arm],s,decision)
                row["checks"] = [c for c in checks if c["end"]==c["t"]]
            rows.append(row)
        with ZipFile(PACK) as z:
            records = json.loads(z.read("manifest.json"))["records"]
        unavailable = [r["index"] for r in records if not r["fits_marker"]]
        assert len(rows) == 108-3*len(unavailable)
        ch.save(work/OUT.name,dict(records=rows,summary=counts,unavailable=unavailable,intended_per_track=36,
                models=history["models"],five_sha256=history["five_sha256"],cutoff=history["cutoff"],
                c161=history["baseline"], pack_sha256=ch.sha(PACK),sources={p.name:ch.sha(p) for p in
                (Path(__file__),Path(mixed.__file__),playback.headroom.OUT,aux.OUT,Path(timing.__file__))},
                policy="Each new external WAV in three levels with music and panel demo. Room assignment and "
                       "preamp are frozen; new offset in the same music file for each family. Preserve all cases."))
        (work/OUT.name).replace(OUT)


def guest():
    host_report = ch.read(OUT)
    ref = ch.read(timing.GUEST)
    assert ch.sha(PACK) == host_report["pack_sha256"] and host_report["c161"] == ref["model"]
    builds = dict(bca=mixed.panel.middle.WORK/"build")
    uplink = tx.panel.grid.gap.uplink
    assert ch.sha(builds["bca"]/"nyx_mac_listener.bin") == ref["firmware"] and uplink.firmware() == ref["codec"]
    small, small_id = ch.load_reader()
    proc, model, qid = prefix.balanced.original.asr.load()
    assert qid == ref["qwen"] and small_id == ref["small"]
    turbo = tx.diag.reader(ref["turbo"])
    decoder = uplink.decoder_api()
    readers = small,turbo,decoder,proc,model
    rule = ch.read(ch.RESULTS/(prefix.rule.study.STEM+"-command-start.json"))
    assert rule["tail_samples"] == 1280 and rule["guard_samples"] == 320
    cache = Path(".local/wakeword/c161-additional-connected-voices") / ch.sha(OUT)[:16]
    cache.mkdir(parents=True,exist_ok=True)
    archive = GUEST.with_suffix(".zip")
    rows, counts = [], defaultdict(Counter)
    with TemporaryDirectory(dir=cache) as folder:
        work = Path(folder)
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs:
            for host_index, ((s,name,mic,_), h) in enumerate(zip(scenes(),host_report["records"],strict=True)):
                assert h["source"] == s and h["track"] == name and ch.digest(mic) == h["result"]["pcm_sha256"]
                if name == "no_address" and s["external_db"] != -6:
                    continue
                if name == "plain" and s["external_db"] != mixed.study.LEVELS[sum(divmod(s["index"],3))%3]:
                    continue
                i = len(rows)
                shard = cache/f"{i:03}.zip"
                if not shard.exists():
                    partial = shard.with_suffix(".tmp")
                    with ZipFile(partial,"w",ZIP_DEFLATED) as part:
                        case = host_index,s,"bca",name,mic,np.zeros_like(mic),h["result"]
                        row = playback.guest_case(work,part,i,case,readers,builds)
                        if row["allowed_fixed"] and row["opened"] and row["allowed_fixed"][0] <= row["opened"][0]:
                            cap = row["uplink"]
                            log = part.read(f"{i:02}-opus.log").decode().replace("\r\n","\n")
                            audio = uplink.decode(decoder,uplink.packets(log,i,cap["samples"],0))
                            assert ch.digest(audio) == cap["pcm_sha256"]
                            cut = max(0,row["allowed_fixed"][0]+rule["tail_samples"]-rule["guard_samples"]+1440-row["finish"][1])
                            row["cropped"] = dict(cut=cut,qwen=prefix.balanced.evaluate(proc,model,audio[cut:],s["command"]),
                                                  turbo=uplink.recognition(turbo,audio[cut:],s["command"]))
                        part.writestr("case.json",json.dumps(row))
                    partial.replace(shard)
                with ZipFile(shard) as part:
                    row = json.loads(part.read("case.json"))
                    assert part.testzip() is None and (row["source"],row["role"],row["host_index"],row["opened"],row["mic_sha256"]) == (
                        s,name,host_index,h["result"]["opened"],h["result"]["pcm_sha256"])
                    for member in part.namelist():
                        if member != "case.json":
                            logs.writestr(member,part.read(member))
                total = counts[f'{name}.{s["external_db"]}']
                timely = mixed.study.tally(total,s,row)
                full = bool(row.get("qwen",{}).get("complete"))
                cropped = row.get("cropped",{}).get("qwen",{}).get("complete",full)
                total.update(timely_full=int(timely and full),timely_cropped=int(timely and cropped))
                rows.append(row)
                if len(rows)%12 == 0:
                    print("P4",len(rows),flush=True)
        assert len(rows) == 60-3*len(host_report["unavailable"])
        with ZipFile(work/archive.name) as logs:
            assert logs.testzip() is None and len(logs.namelist()) == len(set(logs.namelist()))
        ch.save(work/GUEST.name,dict(records=rows,summary=counts,model=ref["model"],firmware=ref["firmware"],
                codec=ref["codec"],turbo=ref["turbo"],qwen=qid,small=small_id,host_sha256=ch.sha(OUT),
                rule_sha256=ch.sha(ch.RESULTS/(prefix.rule.study.STEM+"-command-start.json")),archive_sha256=ch.sha(work/archive.name),
                policy="P4 QEMU with zero reference, all available marked tracks in three levels, weakest "
                       "removed-address and one preassigned plain trial per voice/command. Identical selected "
                       "Opus read whole or from the historical marker-derived cut. No fitted decisions."))
        (work/archive.name).replace(archive)
        (work/GUEST.name).replace(GUEST)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","host","guest"))
    {"prepare":prepare,"host":host,"guest":guest}[parser.parse_args().mode]()
