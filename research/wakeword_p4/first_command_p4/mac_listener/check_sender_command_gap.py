"""Move only the generated marked call earlier and hold the acoustic command and panel clocks fixed."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import check_additional_connected_voices as study

ch, tx, mixed = study.ch, study.tx, study.mixed
GAP = 2560  # 160 ms at 16 kHz
OUT = ch.RESULTS / "4090-c161-sender-command-gap.json"


def advance_call(wave, boundary):
    assert not wave[:GAP].any()
    shifted = wave.copy()
    shifted[:boundary-GAP] = wave[GAP:boundary]
    shifted[boundary-GAP:boundary] = 0
    return shifted


def pairs():
    panel, rate = sf.read(mixed.panel.demo.WAVE, dtype="int16")
    music, _, music_id = tx.study.grid.room.rooms.read_noise("music")
    v5, _ = tx.study.panel_wave()
    baseline = ch.read(study.timing.OUT)
    lengths = {(r["source"]["index"],r["source"]["external_db"]):r["samples"] for r in baseline["records"]
               if r["track"]=="marked" and r["timing"]=="tx80"}
    controls = {(r["source"]["index"],r["source"]["external_db"]):r["baseline"]
                for r in baseline["records"] if r["track"]=="marked_no_address" and r["timing"]=="tx80"}
    second = ch.read(study.OUT)
    second_controls = {(r["source"]["index"],r["source"]["external_db"]):r["result"]
                       for r in second["records"] if r["track"]=="no_address"}
    first_pack = study.prefix.balanced.PLAY
    assert ch.sha(first_pack)==baseline["sources"][first_pack.name] and ch.sha(study.PACK)==second["pack_sha256"]
    for batch, pack_path, report_path, role, variant in (
            ("first",first_pack,study.timing.GUEST,"both","balanced_louder"),
            ("second",study.PACK,study.GUEST,"marked","marked")):
        old = {(r["source"]["index"],r["source"]["external_db"]):r for r in ch.read(report_path)["records"]
               if r["role"]==role}
        with ZipFile(pack_path) as z:
            plan = json.loads(z.read("manifest.json"))
            assert rate==16000 and ch.sha(mixed.panel.demo.WAVE)==plan["panel_sha256"]
            for entry in plan["records"]:
                idx, boundary = entry["index"],entry["command_at"]
                wave = study.playback.read_wav(z,entry["tracks"][variant])
                shifted = advance_call(wave,boundary)
                absent = wave.copy()
                absent[:boundary] = 0
                np.testing.assert_array_equal(absent,advance_call(absent,boundary))
                np.testing.assert_array_equal(wave[boundary:],shifted[boundary:])
                for db in mixed.study.LEVELS:
                    key = idx,db
                    previous = old.pop(key)
                    s = previous["source"]
                    assert s["command_at"]==boundary and s["transmit_event"]==boundary-1280
                    (hp,_,ph),(hx,_,eh) = tx.diag.room(s["geometry"])
                    assert (ph,eh,music_id)==(s["panel_rir_sha256"],s["external_rir_sha256"],s["music_sha256"])
                    local = fftconvolve(panel.astype(float),hp)
                    n = (lengths.pop(key) if batch=="first" else
                         max(len(wave)+len(hx)-1,s["panel_at"]+len(local))+4800)
                    anchor = fftconvolve(v5.astype(float),hp)[:len(v5)]
                    music_window = music[(s["music_offset"]+np.arange(n))%len(music)]
                    bed = music_window * (.35*np.linalg.norm(anchor)/np.sqrt(len(anchor))/np.sqrt(np.mean(music_window**2)))
                    bed[s["panel_at"]:s["panel_at"]+len(local)] += local
                    def render(track):
                        wet = fftconvolve(track.astype(float),hx)*10**(db/20)
                        return mixed.digitize(bed+np.pad(wet,(0,n-len(wet))),s["preamp_gain"])
                    original, _ = render(wave)
                    assert ch.digest(original)==previous["mic_sha256"]
                    ref = (controls if batch=="first" else second_controls).pop(key)
                    control,_ = render(absent)
                    assert ch.digest(control)==ref["pcm_sha256"] and not ref["opened"]
                    mic, clipped = render(shifted)
                    yield batch, dict(s,start=s["start"]-GAP), mic, clipped, previous, ref["pcm_sha256"], ch.digest(shifted)
        assert not old
    assert not lengths and not controls and not second_controls


def main():
    first, second = ch.read(study.timing.GUEST), ch.read(study.GUEST)
    assert all(first[k]==second[k] for k in ("model","firmware","codec","small","turbo","qwen"))
    builds = dict(bca=mixed.panel.middle.WORK/"build")
    uplink = tx.panel.grid.gap.uplink
    assert ch.sha(builds["bca"]/"nyx_mac_listener.bin")==first["firmware"] and uplink.firmware()==first["codec"]
    small,sid = ch.load_reader()
    proc,model,qid = study.prefix.balanced.original.asr.load()
    assert sid==first["small"] and qid==first["qwen"]
    readers = small,tx.diag.reader(first["turbo"]),uplink.decoder_api(),proc,model
    sources = {p.name:ch.sha(p) for p in (study.timing.GUEST,study.GUEST,study.timing.OUT,study.OUT,study.PACK,
               study.prefix.balanced.PLAY,mixed.panel.demo.WAVE,Path(study.playback.__file__),Path(__file__))}
    from hashlib import sha256
    key = sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()[:16]
    cache = Path(".local/wakeword/c161-sender-command-gap")/key
    cache.mkdir(exist_ok=True,parents=True)
    rows, counts = [], defaultdict(Counter)
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory(dir=cache) as folder:
        work = Path(folder)
        matcher = tx.single.timed.binding(work)
        native = tx.panel.grid.room.rooms.c161_binding()
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs:
            for batch,s,mic,clipped,old,absent_hash,track_hash in pairs():
                i = len(rows)
                known = study.playback.host_result(matcher,native,mic,s["transmit_event"])
                shard = cache/f"{i:03}.zip"
                if not shard.exists():
                    case = i,s,"bca","both",mic,np.zeros_like(mic),known
                    partial = shard.with_suffix(".tmp")
                    with ZipFile(partial,"w",ZIP_DEFLATED) as part:
                        r = study.playback.guest_case(work,part,i,case,readers,builds)
                        part.writestr("case.json",json.dumps(r))
                    partial.replace(shard)
                with ZipFile(shard) as part:
                    assert part.testzip() is None
                    r = json.loads(part.read("case.json"))
                    assert (r["source"],r["mic_sha256"],r["opened"])==(s,known["pcm_sha256"],known["opened"])
                    for member in part.namelist():
                        if member!="case.json":
                            logs.writestr(member,part.read(member))
                total = counts[f'{batch}.{s["external_db"]}']
                timely = mixed.study.tally(total,s,r)
                old_open, = old["opened"]
                assert old["source"]["start"] <= old_open < s["command_at"]
                for label, before, after in (("qwen",old["qwen"]["complete"],r.get("qwen",{}).get("complete",False)),
                                             ("turbo",old["turbo"],r["turbo"])):
                    total.update({label+"_before":int(before),label:int(timely and after),
                                  label+"_gained":int(timely and after and not before),
                                  label+"_lost":int(before and not (timely and after))})
                rows.append(dict(batch=batch,board=r,previous_case=old["index"],previous_open=old_open,
                                 previous_qwen=old["qwen"],previous_turbo=old["turbo"],clipped_samples=clipped,
                                 shifted_external_sha256=track_hash,no_address_sha256=absent_hash))
                if len(rows)%12==0:
                    print("P4",len(rows),flush=True)
        assert len(rows)==72 and all(c["trials"]==12 for c in counts.values())
        with ZipFile(work/archive.name) as logs:
            assert logs.testzip() is None and len(logs.namelist())==len(set(logs.namelist()))
        ch.save(work/OUT.name,dict(records=rows,summary=counts,gap_samples=GAP,sources=sources,
                model=first["model"],firmware=first["firmware"],codec=first["codec"],qwen=qid,turbo=first["turbo"],
                archive_sha256=ch.sha(work/archive.name),policy="Advance only the entire external marked prefix by "
                "160 ms into its existing leading silence. The original command, its clock, file length, room, "
                "music, panel, TX and gain stay fixed. Verify original and omitted-address microphone hashes. "
                "Run the frozen P4 listener on every changed mix, decode full 24 kb/s Opus, and score the "
                "unchanged exact command suffix. Same consumed two synthetic development cohorts."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
