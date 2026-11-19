"""Match the original fixed Bella calls on the 24 room inputs used for board capture."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import willow_music_cases as runner

WORK = Path(".local/wakeword/c161-bella-room-willow")
PLAN = runner.ROOT / "results/4090-c161-bella-room-willow-plan.json"
PACK = PLAN.with_name("c161-bella-willow-command-playback.zip")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())


def member(pack, path, data):
    item = ZipInfo(path)
    item.compress_type = ZIP_DEFLATED
    pack.writestr(item, data)


def prepare():
    import numpy as np
    import soundfile as sf

    import bella_peak_willow as peak
    import check_bella_board_reference as board
    import check_bella_reference_uplink as uplink
    import eight_voice_willow as eight

    ch, bella, study, tx = board.ch, board.bella, board.study, board.tx
    base, old_plan = ch.read(bella.OUT), ch.read(peak.PLAN)
    old_eight = ch.read(eight.PLAN)
    runtime = old_plan["runtime"]
    known_files = {path:old_plan["files"][path] for path in runtime.values()}
    cached = {}
    for path in (eight.PLAN, peak.PLAN):
        prior = ch.read(path)
        assert prior["runtime"] == runtime and all(prior["files"][k] == v for k,v in known_files.items())
        for i, r in enumerate(prior["records"]):
            cached[r["pcm_sha256"]] = dict(plan=path.name, index=i)

    ref_rows = ch.read(uplink.OUT)["records"]
    controls = {(r["batch"], r["track"]):r for r in old_eight["records"]
                if r["mode"] == "rear" and r["target_id"] == "no_address"}
    with ZipFile(eight.PACK) as pack:
        anchor, = [r for r in json.loads(pack.read("manifest.json"))["records"]
                   if r["batch"] == "second" and r["track"] == 0 and r["role"] == "willow_matched"]
        track = study.playback.read_wav(pack, dict(anchor, sha256=anchor["wav_sha256"]))
    assert anchor["voice"] == "af_bella" and ch.sha(eight.PACK) == old_plan["files"][str(eight.PACK)]
    cue = track[32000:anchor["command_at"]]
    duration = anchor["command_at"] - anchor["start"]
    levels = old_plan["levels"]
    assert tx.balanced.dry_levels(cue[-duration:]) == levels["willow_original"]
    assert abs(levels["nyx_original"]["active_rms_dbfs"] - levels["willow_original"]["active_rms_dbfs"]) < .01
    quiet = np.pad(track[:anchor["command_at"]], (0,32000))
    quiet_source = cached[ch.digest(quiet)]
    assert old_eight["records"][quiet_source["index"]]["mode"] == "isolated" and quiet_source["plan"] == eight.PLAN.name

    music, _, music_id = tx.study.grid.room.rooms.read_noise("music")
    panel, rate = sf.read(study.mixed.panel.demo.WAVE, dtype="int16")
    v5, _ = tx.study.panel_wave()
    filters, _ = bella.gains.direction.filters()
    assert rate == 16000 and ch.sha(board.weak.PACK) == board.run_guest.BELLA_SHA
    WORK.mkdir(exist_ok=True, parents=True)
    cases, records, members = [], [], []
    with TemporaryDirectory(dir=WORK) as temp, ZipFile(board.weak.PACK) as originals:
        files = {r["file"]:r for r in json.loads(originals.read("manifest.json"))["records"]}
        archive = Path(temp)/PACK.name
        with ZipFile(archive,"w") as dry:
            for row, ref_row in zip(base["captures"], ref_rows, strict=True):
                s = row["source"]
                key = row["batch"], s["index"]
                control = controls.pop(key)
                name = ref_row["file"]
                assert name == f"{key[0]}/{key[1]:03}.wav" and s["external_db"] == ref_row["db"] == control["external_db"]
                orig = study.playback.read_wav(originals,files[name])
                end = s["command_at"]
                _, (hx, _, _) = tx.diag.room(s["geometry"])
                length = control["samples"] - 4800 - len(hx) + 1
                mix = study.mixed.room_scene(s, {"length":(files[name],range(length))}, panel,v5,music,music_id)
                mic, _ = mix(bella.rear_filter(orig,filters["back"]))
                assert ch.digest(mic) == row["mic_sha256"] == ref_row["mic_sha256"] and len(mic) == control["samples"]
                ref = np.zeros_like(mic)
                ref[s["panel_at"]:s["panel_at"]+len(panel)] = panel
                assert ch.digest(ref) == ref_row["continuous"]["reference_sha256"]
                wave = orig.copy()
                wave[:end] = 0
                absent, _ = mix(bella.rear_filter(wave,filters["back"]))
                assert ch.digest(absent) == control["pcm_sha256"]
                wave[end-len(cue):end] = cue
                assert ch.digest(wave[end:]) == files[name]["command_pcm_sha256"] == s["command_pcm_sha256"]
                out = BytesIO()
                sf.write(out,wave,16000,subtype="PCM_16",format="WAV")
                member(dry,name,out.getvalue())
                members.append(dict(file=name,sha256=sha256(out.getvalue()).hexdigest(),pcm_sha256=ch.digest(wave),
                                    command_pcm_sha256=s["command_pcm_sha256"],command=s["command"],start=end-duration,command_at=end))
                willow, clipped = mix(bella.rear_filter(wave,filters["back"]))
                mic_sha = ch.digest(willow)
                record = dict(file=name,target_id="willow_original",start=end-duration,end=end,
                              samples=len(willow),pcm_sha256=mic_sha)
                source = cached.get(mic_sha)
                if source is None:
                    path = WORK/f'{len(records):02}.wav'
                    sf.write(path,willow,16000,subtype="PCM_16")
                    record.update(path=str(path),sha256=ch.sha(path))
                    known_files[str(path)] = record["sha256"]
                    source = dict(plan=PLAN.name,index=len(records))
                    records.append(record)
                cases.append(dict(file=name,db=s["external_db"],command=s["command"],start=end-duration,end=end,
                                  samples=len(willow),clipped=clipped,willow_mic_sha256=mic_sha,willow=source,
                                  no_address_mic_sha256=control["pcm_sha256"],no_address=cached[control["pcm_sha256"]],
                                  nyx_mic_sha256=ch.digest(mic),reference_sha256=ch.digest(ref)))
        assert not controls and Counter(r["db"] for r in cases) == {-6:8,0:8,6:8}
        with ZipFile(archive,"a") as dry:
            member(dry,"manifest.json",json.dumps(dict(voice="af_bella",rate=16000,records=members),indent=2)+"\n")
        archive.replace(PACK)
    for path in (SCRIPT,eight.PLAN,peak.PLAN,eight.PACK,board.weak.PACK,board.OUT,bella.OUT,uplink.OUT,PACK):
        known_files[str(path)] = ch.sha(path)
    plan = dict(records=records,cases=cases,files=known_files,runtime=runtime,threshold_requests=["DEFAULT"],
                quiet=dict(source=quiet_source,pcm_sha256=ch.digest(quiet),start=anchor["start"],end=anchor["command_at"]),
                levels={k:levels[k] for k in ("nyx_original","willow_original")},
                policy="Original Bella addresses right-aligned at the same command onset. Same command suffix, rear response, "
                       "room, music, panel, microphone gain and window as the 24 board captures. Reuse native DEFAULT runs "
                       "by exact microphone PCM, including address-erased controls and one separate isolated positive. "
                       "Count every room case. Use each listener's own first 256-sample open block for its uplink.")
    if PLAN.exists():
        assert ch.read(PLAN) == plan
    else:
        ch.save(PLAN,plan)


if __name__ == "__main__":
    runner.PLAN, runner.WORK, runner.RUNNER = PLAN, WORK, SCRIPT
    {"prepare":prepare, "case":runner.case, "run":runner.run}[sys.argv.pop(1)]()
