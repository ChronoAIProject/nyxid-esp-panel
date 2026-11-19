"""Prepare and run the official default Willow reference on the eight-voice handoff."""
from pathlib import Path
import sys

import willow_music_cases as runner

STEM = "4090-c161-eight-voice-willow"
PLAN = runner.ROOT / "results" / (STEM + "-plan.json")
PACK = PLAN.with_name("c161-eight-voice-willow-comparison.zip")
WORK = Path(".local/wakeword/c161-eight-voice-willow")
RUNNER = Path(__file__).resolve().relative_to(Path.cwd())


def comparison_sources(study):
    import numpy as np
    import re
    from unittest.mock import patch

    first = study.mixed.study.sources()
    sources = {}
    for (voice, command), group in first.items():
        willow, pcm = group["Hey Willow"]
        nyx, nyx_pcm = group["Hey Nyx"]
        sources["first",voice,command] = (willow["synthesis"], pcm[:willow["end"]], willow["start"],
                                          nyx_pcm[:nyx["end"]], nyx["synthesis"])
    with patch.multiple(study.onset, VOICES=study.VOICES, CARRIERS=study.COMMANDS,
                        ADDRESSES=("Hey Willow",), LABELS=(3,)):
        for s, pcm in study.onset.synthesize():
            words = list(re.finditer(r"[^ ,.!?]+", s["phones"]))
            edges = np.cumsum(s["duration"], dtype=int)*400 + 32000
            pos = words[2].start()
            while s["phones"][pos] in "ˈˌ":
                pos += 1
            start, end = int(edges[words[0].start()]), int(edges[pos])
            prefix = np.pad(pcm,(32000,0))[:end]
            sources["second",s["voice"],s["carrier"]] = s,prefix,start,None,None
    return sources


def prepare():
    from io import BytesIO
    import json
    from tempfile import TemporaryDirectory
    from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo
    import numpy as np
    from scipy.signal import fftconvolve
    import soundfile as sf
    import check_connected_marker_gains as gains
    import run_c161_willow_matched_scenes_v1 as launcher

    study, ch, tx = gains.study, gains.ch, gains.tx
    result = ch.read(gains.GUEST)
    assert ch.sha(gains.PACK) == result["pack_sha256"]
    reference = ch.read(study.mixed.study.connected.PLAN)
    files = {p:reference["files"][p] for p in reference["runtime"].values()}
    source_files = (RUNNER, Path(runner.__file__), Path(launcher.__file__), Path(launcher.official.__file__),
                    launcher.official.SOURCE, gains.PACK, gains.GUEST, gains.direction.OUT, study.PACK,
                    study.mixed.study.connected.PACK, study.mixed.study.connected.PLAN, study.mixed.panel.demo.WAVE,
                    Path(study.onset.__file__), Path(gains.direction.__file__), Path(study.mixed.__file__))
    for p in source_files:
        p = p.resolve().relative_to(Path.cwd())
        files[str(p)] = ch.sha(p)
    levels = tx.balanced.dry_levels
    sources = comparison_sources(study)
    witness = {(r["batch"],r["source"]["index"],"rear" if r["version"]=="enhanced" else "isolated"): r
               for r in result["records"] if r.get("paired_back_grid") or r["version"]=="wake_only"}
    controls = {(r["batch"],r["source"]["index"],r["source"]["external_db"]):r["result"]
                for r in ch.read(gains.direction.OUT)["records"] if r["profile"]=="back" and r["role"]=="no_address"}
    lengths = {(r["source"]["index"],r["source"]["external_db"]):r["samples"]
               for r in ch.read(study.timing.OUT)["records"] if r["timing"]=="tx80" and r["track"]=="marked"}
    panel, rate = sf.read(study.mixed.panel.demo.WAVE,dtype="int16")
    bed, _, bed_id = tx.study.grid.room.rooms.read_noise("music")
    v5, _ = tx.study.panel_wave()
    taps, _ = gains.direction.filters()
    assert rate==16000
    WORK.mkdir(parents=True,exist_ok=True)
    rows, dry_tracks = [], []
    native = tx.panel.grid.room.rooms.c161_binding()
    with TemporaryDirectory(dir=WORK) as folder, ZipFile(gains.PACK) as shipped, ZipFile(study.PACK) as second:
        work = Path(folder)
        matcher = tx.single.timed.binding(work)
        second_entries = {r["index"]:r for r in json.loads(second.read("manifest.json"))["records"]}
        with ZipFile(work/PACK.name,"w",ZIP_DEFLATED) as pack:
            for r in json.loads(shipped.read("manifest.json"))["records"]:
                batch, idx, end = r["batch"],r["index"],r["command_at"]
                board = witness.pop((batch,idx,"rear"))
                s = board["source"]
                marked = study.playback.read_wav(shipped,r["connected"])
                sy, willow, willow_start, plain, nyx = sources.pop((batch,r["voice"],r["command"]))
                if plain is None:
                    entry = second_entries[idx]
                    plain = study.playback.read_wav(second,entry["tracks"]["plain"])[:end]
                    nyx = entry["synthesis"]
                assert (nyx["voice"],nyx["carrier"],nyx["speed"],nyx["separator"]) == (
                        sy["voice"],sy["carrier"],sy["speed"],sy["separator"])
                assert len(plain)==end and ch.digest(marked[end:]) == r["command_sha256"]
                start = end - len(willow) + willow_start
                assert start>0 and (len(willow)<=end or not willow[:len(willow)-end].any())
                prefix = np.zeros(end,"<i2")
                count = min(end,len(willow))
                prefix[-count:] = willow[-count:]
                gain_db = min(levels(marked[r["start"]:end])["active_rms_dbfs"]-levels(prefix[start:])["active_rms_dbfs"],
                              -1.-levels(prefix)["peak_dbfs"])
                balanced = np.rint(prefix.astype(float)*10**(gain_db/20))
                assert abs(balanced).max()<=32767
                addresses = {"nyx_marked":marked[:end],"nyx_plain":plain,"willow_native":prefix,
                             "willow_matched":balanced.astype("<i2"),"no_address":np.zeros(end,"<i2")}
                db = s["external_db"]
                if batch=="first":
                    _,(hx,_,_) = tx.diag.room(s["geometry"])
                    n = lengths[idx,db] - 4800 - len(hx)+1
                else:
                    n = len(marked)
                scene = study.mixed.room_scene(s,{"length":(s,range(n))},panel,v5,bed,bed_id)
                isolated = witness.pop((batch,idx,"isolated"),None)
                for role,head in addresses.items():
                    track = np.r_[head,marked[end:]]
                    member = f"{batch}/{idx:03}/{role}.wav"
                    buffer = BytesIO()
                    sf.write(buffer,track,16000,format="WAV",subtype="PCM_16")
                    data = buffer.getvalue()
                    info = ZipInfo(member)
                    info.compress_type = ZIP_DEFLATED
                    pack.writestr(info,data)
                    boundary = start if role.startswith("willow") else r["start"]
                    dry_tracks.append(dict(batch=batch,track=idx,role=role,file=member,voice=r["voice"],command=r["command"],
                                           start=boundary,command_at=end,panel_at=s["panel_at"],external_db=db,
                                           wav_sha256=gains.direction.sha256(data).hexdigest(),pcm_sha256=ch.digest(track),
                                           suffix_sha256=ch.digest(track[end:]),willow_gain_db=float(gain_db) if role=="willow_matched" else 0.,
                                           active_rms_dbfs=levels(head[boundary:])["active_rms_dbfs"] if role!="no_address" else None,
                                           willow_synthesis=sy if role=="willow_native" else None))
                    filtered = fftconvolve(track.astype(float),taps["back"])
                    assert np.max(np.abs(filtered[len(track):]))<.5
                    mic, clipped = scene(filtered[:len(track)])
                    views = [("rear",mic,s["transmit_event"],clipped,board)]
                    if isolated is not None:
                        views.append(("isolated",np.pad(head,(0,32000)),0,0,isolated))
                    for mode,pcm,clock,clips,expected in views:
                        host = study.playback.host_result(matcher,native,pcm,clock)
                        if role=="nyx_marked":
                            assert host["pcm_sha256"]==expected["mic_sha256"] and host["opened"]==expected["opened"]
                        if mode=="rear" and role=="no_address":
                            assert host==controls[batch,idx,db]
                        path = WORK/f"{len(rows):03}.wav"
                        sf.write(path,pcm,16000,subtype="PCM_16")
                        files[str(path)] = ch.sha(path)
                        rows.append(dict(batch=batch,track=idx,voice=r["voice"],carrier=r["command"],mode=mode,
                                         target_id=role,path=str(path),sha256=files[str(path)],samples=len(pcm),
                                         pcm_sha256=host["pcm_sha256"],dry_member=member,external_db=db if mode=="rear" else None,
                                         start=boundary,end=end,transmit_event=clock,clipped=clips,nyx=host))
            pack.writestr(ZipInfo("manifest.json"),json.dumps(dict(records=dry_tracks,rate=16000,
                policy="Original external clock, dry only; all five counterparts share identical command samples. "
                       "For isolated playback play the prefix only. Room simulation gives the external source the rear "
                       "response; a physical setup uses the original dry WAV. Values are relative, not absolute SPL."),
                       indent=2)+"\n")
        assert not witness and not sources and len(dry_tracks)==120 and len(rows)==160
        with ZipFile(work/PACK.name) as pack:
            assert pack.testzip() is None
        (work/PACK.name).replace(PACK)
        files[str(PACK)] = ch.sha(PACK)
        plan = dict(records=rows,files=files,runtime=reference["runtime"],threshold_requests=["DEFAULT"],
                    nyx_model=result["model"],policy="Eight matched training-pool voices. 24 fixed rear command "
                    "conditions and 8 isolated address conditions, each with five inputs. The isolated clock "
                    "ends at the original annotated boundary and includes 2 s additional silence. "
                    "Score before boundary, within 500 ms after and anywhere separately. Round delivery to 256 samples.")
        if PLAN.exists():
            assert ch.read(PLAN)==plan
        else:
            ch.save(PLAN,plan)


if __name__=="__main__":
    runner.PLAN, runner.WORK, runner.RUNNER = PLAN, WORK, RUNNER
    {"prepare":prepare,"run":runner.run,"case":runner.case}[sys.argv.pop(1)]()
