"""Transfer the frozen three-marker gain fit to eight synthetic voices at equal prefix RMS."""
import argparse
from collections import Counter, defaultdict
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import check_sender_directivity as direction
import fit_single_marker_gains as fit

study, tx, ch = direction.study, direction.tx, direction.ch
OUT = ch.RESULTS / "4090-c161-connected-marker-gains.json"
GUEST = OUT.with_name(OUT.stem + "-p4.json")
PACK = ch.RESULTS / "c161-eight-voice-marker-gains.zip"


def put_wav(z, name, pcm):
    buf = BytesIO()
    sf.write(buf, pcm, 16000, format="WAV", subtype="PCM_16")
    data = buf.getvalue()
    z.writestr(name, data)
    return dict(file=name, sha256=direction.sha256(data).hexdigest(), pcm_sha256=ch.digest(pcm), samples=len(pcm))


def prepare():
    history = ch.read(fit.FIT)
    gains = history["weights"][history["selected"]]["marker_gains"]
    spec, v5, _, codes = tx.study.grid.fit.original()
    rows, sources = [], {}
    with TemporaryDirectory(dir=PACK.parent) as folder:
        path = Path(folder)/PACK.name
        with ZipFile(path, "w", ZIP_DEFLATED) as pack:
            for batch, src, name in (("first", study.prefix.balanced.PLAY, "balanced_louder"),
                                     ("second", study.PACK, "marked")):
                sources[batch] = {src.name: ch.sha(src)}
                with ZipFile(src) as original:
                    records = json.loads(original.read("manifest.json"))["records"]
                    for pos, entry in enumerate(records):
                        previous = entry["tracks"][name]
                        old = study.playback.read_wav(original, previous)
                        at = entry["command_at"]
                        prefix = old[:at].astype(float)
                        for offset, code, gain in zip(spec["code_positions"], codes, gains, strict=True):
                            start = at - len(v5) + offset
                            assert start >= entry["start"]
                            prefix[start:start+len(code)] += (gain-1) * code
                        scale = np.linalg.norm(old[:at].astype(float))/np.linalg.norm(prefix)
                        prefix = np.rint(prefix*scale)
                        assert prefix.min() >= -32768 and prefix.max() <= 32767
                        new = old.copy()
                        new[:at] = prefix.astype("<i2")
                        assert np.array_equal(new[at:], old[at:]) and not new[:32000].any()
                        delta_db = 20*np.log10(np.linalg.norm(new[:at].astype(float))/np.linalg.norm(old[:at].astype(float)))
                        assert abs(delta_db) < .001
                        root = f'{batch}/{entry["index"]:03}'
                        row = dict(batch=batch, index=entry["index"], voice=entry["voice"], command=entry["command"],
                                   start=entry["start"], command_at=at, prefix_scale=float(scale), rms_delta_db=float(delta_db),
                                   peak=int(np.abs(new[:at].astype(int)).max()), command_sha256=ch.digest(new[at:]),
                                   original=previous, connected=put_wav(pack, root+"/marked.wav", new))
                        if pos%3 == 0:
                            row["wake_only"] = put_wav(pack, "wake/"+entry["voice"]+".wav", new[32000:at])
                        rows.append(row)
            assert len(rows)==24 and sum("wake_only" in r for r in rows)==8
            pack.writestr("manifest.json", json.dumps(dict(records=rows, sources=sources, rate=16000, format="mono PCM16",
                gains=gains, fit_sha256=ch.sha(fit.FIT), model=ch.read(direction.OUT)["model"], script_sha256=ch.sha(Path(__file__)),
                policy="Apply previously fitted code gains to the three marked prefix intervals, then match the "
                       "whole prefix RMS to the original track. The complete original command suffix is retained. "
                       "The first carrier per voice supplies the wake-only WAV. All files are dry external playback."),
                indent=2)+"\n")
        with ZipFile(path) as packed:
            assert packed.testzip() is None
        path.replace(PACK)


def recordings():
    report = ch.read(direction.OUT)
    panel, rate = sf.read(direction.mixed.panel.demo.WAVE, dtype="int16")
    music, _, music_id = tx.study.grid.room.rooms.read_noise("music")
    v5, _ = tx.study.panel_wave()
    taps, info = direction.filters()
    assert rate == 16000 and info == report["filters"]
    lengths = {(r["source"]["index"], r["source"]["external_db"]):r["samples"]
               for r in ch.read(study.timing.OUT)["records"] if r["track"]=="marked" and r["timing"]=="tx80"}
    originals = {(r["batch"],r["source"]["index"],r["source"]["external_db"],r["profile"],r["role"]):r
                 for r in report["records"]}
    with ZipFile(PACK) as pack:
        manifest = json.loads(pack.read("manifest.json"))
        assert manifest["fit_sha256"] == ch.sha(fit.FIT)
        for batch, path in (("first", study.prefix.balanced.PLAY), ("second", study.PACK)):
            assert ch.sha(path)==manifest["sources"][batch][path.name]
            with ZipFile(path) as old_pack:
                for row in (r for r in manifest["records"] if r["batch"]==batch):
                    idx = row["index"]
                    old = study.playback.read_wav(old_pack,row["original"])
                    new = study.playback.read_wav(pack,row["connected"])
                    at = row["command_at"]
                    assert np.array_equal(old[at:],new[at:])
                    absent = new.copy()
                    absent[:at] = 0
                    versions = {}
                    for profile in ("flat", *direction.ANGLES):
                        def apply(wave):
                            if profile=="flat":
                                return wave
                            filtered = fftconvolve(wave.astype(float),taps[profile])
                            assert np.max(np.abs(filtered[len(wave):])) < .5
                            return filtered[:len(wave)]
                        versions[profile] = tuple(apply(wave) for wave in (old,new,absent))
                    for db in direction.mixed.study.LEVELS:
                        s = originals[batch,idx,db,"flat","marked"]["source"]
                        if batch=="first":
                            (_,_,_), (hx,_,_) = tx.diag.room(s["geometry"])
                            n = lengths.pop((idx,db)) - 4800 - len(hx)+1
                        else:
                            n = len(old)
                        mix = direction.mixed.room_scene(s, {"length":(row,range(n))}, panel,v5,music,music_id)
                        for profile,(a,b,negative) in versions.items():
                            before = originals.pop((batch,idx,db,profile,"marked"))
                            control = originals.pop((batch,idx,db,profile,"no_address"))
                            raw, _ = mix(a)
                            assert ch.digest(raw) == before["result"]["pcm_sha256"]
                            assert ch.digest(mix(negative)[0]) == control["result"]["pcm_sha256"] and not control["result"]["opened"]
                            mic, clipped = mix(b)
                            yield batch,s,profile,raw,mic,clipped,before
    assert not originals and not lengths


def host():
    rows, counts = [], defaultdict(Counter)
    native = tx.panel.grid.room.rooms.c161_binding()
    with TemporaryDirectory(dir=OUT.parent) as folder:
        work = Path(folder)
        matcher = tx.single.timed.binding(work)
        for batch,s,profile,_,mic,clipped,before in recordings():
            result = study.playback.host_result(matcher,native,mic,s["transmit_event"])
            group = counts[batch+"."+profile]
            timely = direction.mixed.study.tally(group,s,result)
            group.update(before=int(before["timely"]),gained=int(timely and not before["timely"]),
                         lost=int(before["timely"] and not timely))
            rows.append(dict(batch=batch,source=s,profile=profile,timely=timely,result=result,
                             before=before["timely"],previous_opened=before["result"]["opened"],
                             paired_back_grid=bool(profile=="back" and before["selected_for_p4"]), clipped_samples=clipped))
        assert len(rows)==288 and sum(r["paired_back_grid"] for r in rows)==24
        ch.save(work/OUT.name,dict(records=rows, summary=counts, pack_sha256=ch.sha(PACK), baseline_sha256=ch.sha(direction.OUT),
            model=ch.read(direction.OUT)["model"], fit_sha256=ch.sha(fit.FIT), script_sha256=ch.sha(Path(__file__)),
            policy="Evaluate every original marked condition and byte-check every no-address input. Run the complete "
                   "preassigned 24-case rear grid plus all changed timely decisions on P4. Include one isolated "
                   "dry cue per voice. Keep targeted changes separate from the preassigned ASR grid."))
        (work/OUT.name).replace(OUT)


def guest():
    report = ch.read(OUT)
    original = ch.read(direction.GUEST)
    assert ch.sha(PACK)==report["pack_sha256"] and original["host_sha256"]==report["baseline_sha256"]
    original_cases = {(r["batch"],r["source"]["index"],r["source"]["external_db"],r["profile"]):r
                      for r in original["records"] if r["role"]=="marked"}
    builds = dict(bca=direction.mixed.panel.middle.WORK/"build")
    uplink = tx.panel.grid.gap.uplink
    assert ch.sha(builds["bca"]/"nyx_mac_listener.bin")==original["firmware"] and uplink.firmware()==original["codec"]
    small, sid = ch.load_reader()
    proc, model, qid = study.prefix.balanced.original.asr.load()
    assert sid==original["small"] and qid==original["qwen"]
    readers = small,tx.diag.reader(original["turbo"]),uplink.decoder_api(),proc,model
    archive = GUEST.with_suffix(".zip")
    with TemporaryDirectory(dir=OUT.parent) as folder:
        work = Path(folder)
        rows = []
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs:
            def capture(host_index,s,role,mic,result):
                case = host_index,s,"bca",role,mic,np.zeros_like(mic),result
                return study.playback.guest_case(work,logs,len(rows),case,readers,builds)
            for j,(scene,h) in enumerate(zip(recordings(),report["records"],strict=True)):
                batch,s,profile,raw,mic,_,before = scene
                assert (s,batch,profile,ch.digest(mic)) == (h["source"],h["batch"],h["profile"], h["result"]["pcm_sha256"])
                if not h["paired_back_grid"] and h["timely"]==h["before"]:
                    continue
                key = batch,s["index"],s["external_db"],profile
                old = original_cases.get(key)
                if old is None:
                    old = capture(j,s,"marked",raw,before["result"])
                    rows.append(dict(old,version="original",profile=profile,batch=batch))
                assert old["mic_sha256"]==before["result"]["pcm_sha256"] and old["opened"]==h["previous_opened"]
                new = capture(j,s,"marked",mic,h["result"])
                rows.append(dict(new,version="enhanced",profile=profile,batch=batch,paired_back_grid=h["paired_back_grid"],
                                 before_qwen=old.get("qwen"),before_turbo=old["turbo"],previous_opened=old["opened"]))
                if len(rows)%10==0:
                    print("P4",len(rows),flush=True)
            matcher = tx.single.timed.binding(work)
            native = tx.panel.grid.room.rooms.c161_binding()
            with ZipFile(PACK) as pack:
                for r in json.loads(pack.read("manifest.json"))["records"]:
                    if "wake_only" not in r:
                        continue
                    dry = study.playback.read_wav(pack,r["wake_only"])
                    mic = np.pad(dry,(32000,32000))
                    s = dict(start=r["start"],command_at=r["command_at"],transmit_event=0,command="Hey Nyx",
                             voice=r["voice"],index=r["index"])
                    result = study.playback.host_result(matcher,native,mic,0)
                    board = capture(-1,s,"external",mic,result)
                    rows.append(dict(board,version="wake_only",batch=r["batch"],wav=r["wake_only"]))
        with ZipFile(work/archive.name) as logs:
            assert logs.testzip() is None and len(set(logs.namelist()))==len(logs.namelist())
        assert sum(r.get("paired_back_grid",False) for r in rows)==24 and sum(r["version"]=="wake_only" for r in rows)==8
        ch.save(work/GUEST.name,dict(records=rows, baseline_p4_sha256=ch.sha(direction.GUEST),host_sha256=ch.sha(OUT),
                archive_sha256=ch.sha(work/archive.name),pack_sha256=ch.sha(PACK),
                **{k:original[k] for k in ("model","firmware","codec","qwen","turbo","small")}))
        (work/archive.name).replace(archive)
        (work/GUEST.name).replace(GUEST)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","host","p4"))
    args = parser.parse_args()
    {"prepare":prepare,"host":host,"p4":guest}[args.mode]()
