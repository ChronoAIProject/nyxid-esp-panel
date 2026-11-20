"""Read the complete peer sends in the twelve weak-target, zero-offset presentations."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import soundfile as sf

import verify_bella_early_groups_3db as verify
import score_third_triplet_player_level as prior

c, shifts, P, scoring = verify.c, verify.shifts, verify.P, prior.scoring
OUT = c.FIT.with_name("4090-bella-early-groups-3db-commands.json")


def main():
    report = P.read(shifts.OUT)
    assert P.read(verify.OUT)["history_gate"] and P.sha(P.QEMU) == report["build"]["qemu_sha256"]
    expected = {r["label"]:r for r in report["records"] if r["role"]=="new" and r["offset_ms"]==0}
    old = P.read(shifts.level.OUT.with_stem(shifts.level.OUT.stem+"-reduced-commands"))
    evidence = P.read(prior.anchor.trial.OUT)
    scoring.sender.BUILD = prior.SENDER
    images = (scoring.clock.BUILD/"nyx_mac_listener.bin",scoring.prior.CODEC/"nyx_first_command.bin",prior.SENDER/"board.bin")
    assert all(P.sha(p)==evidence["sources"][str(p)] for p in images)
    proc, model, qid = scoring.prior.load()
    decoder = scoring.prior.decoder_api()
    assert qid==old["qwen"] and "4090" in qid["device"]
    assert decoder.opus_get_version_string().decode()==old["decoder"]
    readers = proc,model,scoring.prior.reader(old["turbo"]),decoder
    cache = c.WORK/("sends-"+P.sha(shifts.OUT)[:16]+"-"+P.sha(images[-1])[:12])
    cache.mkdir(exist_ok=True)
    scoring.clock.reference.WORK = cache
    rows, files = [], {}
    with TemporaryDirectory(dir=c.WORK) as tmp:
        work = Path(tmp)
        with ZipFile(work/"uart.zip","w",ZIP_DEFLATED) as logs:
            for parent, role, offset, mic, _, _, _, _ in shifts.inputs("shifts"):
                if (role,offset) != ("new",0):
                    continue
                key = f"{parent['angle']}-{parent['direction']}-{parent['scene']:03}-new-0"
                row = dict(expected.pop(key))
                assert row["pcm_sha256"] == P.digest(mic)
                if row["events"]["O"]:
                    opened, = row["events"]["O"]
                    path = work/"mic.wav"
                    sf.write(path,mic,16000,subtype="PCM_16")
                    record = dict(row,reference_sha256=row["ref_sha256"],path=str(path))
                    uplink, shard = scoring.score_case(record,len(rows),opened,readers,work,cache)
                    assert all(uplink[k]==0 for k in ("capture_watchdog","opus_watchdog","sender_watchdog"))
                    row["uplink"] = uplink
                    with ZipFile(shard) as saved:
                        for member in saved.namelist():
                            if member.endswith(".log"):
                                assert member not in files
                                raw = saved.read(member)
                                logs.writestr(member,raw)
                                files[member] = shifts.sha256(raw).hexdigest()
                print(row["label"],row["events"]["O"],flush=True)
                rows.append(row)
        assert not expected and len(rows)==12
        assert sum(not k.startswith("retry/") for k in files)==3*sum(bool(r["events"]["O"]) for r in rows)
        deps = (Path(__file__).resolve().relative_to(Path.cwd()),shifts.OUT,verify.OUT,c.PROTO,*images)
        P.save(work/OUT.name,dict(records=rows,files=files,archive_sha256=P.sha(work/"uart.zip"),qwen=qid,
                                  turbo=old["turbo"],decoder=old["decoder"],sources={str(p):P.sha(p) for p in deps}))
        (work/"uart.zip").replace(OUT.with_suffix(".zip"))
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
