"""Read the complete first commands on the three front-speaker 240° presentations."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import bella_6db_player as player
import score_bella_headroom_commands as previous

P, scoring = previous.P, previous.scoring
OUT = player.OUT.with_name("4090-bella-6db-front-commands.json")
ARCHIVE = player.WORK / OUT.with_suffix(".zip").name


def main():
    wakes, old = P.read(player.OUT), P.read(previous.OUT)
    assert wakes["plan_sha256"] == P.sha(player.PLAN) and wakes["build"]["qemu_sha256"] == P.sha(P.QEMU)
    baseline = {r["label"]: r for r in old["records"] if r["angle"] == 240 and r["direction"] == "front"}
    assert len(baseline) == 9
    for image in (scoring.clock.BUILD/"nyx_mac_listener.bin", scoring.prior.CODEC/"nyx_first_command.bin",
                  scoring.sender.BUILD/"board.bin"):
        assert P.sha(image) == P.read(previous.baseline.earlier.OUT)["sources"][str(image)]
    proc, model, qwen = scoring.prior.load()
    assert qwen == old["qwen"] and "4090" in qwen["device"]
    decoder = scoring.prior.decoder_api()
    assert decoder.opus_get_version_string().decode() == old["decoder"]
    readers = proc, model, scoring.prior.reader(old["turbo"]), decoder
    cache = player.WORK / ("sends-" + P.sha(player.OUT)[:16])
    cache.mkdir(exist_ok=True)
    scoring.clock.reference.WORK = cache
    rows, files = [], {}
    with TemporaryDirectory(dir=player.WORK) as folder:
        work = Path(folder)
        with ZipFile(work/ARCHIVE.name, "w", ZIP_DEFLATED) as dest:
            for number, r in enumerate(wakes["records"]):
                if r["baseline_label"] not in baseline:
                    continue
                engine = "willow" if r["role"] == "willow" else "nyx"
                opened = r[engine+"_open"]
                prior = baseline[r["baseline_label"]]
                row = dict(label=r["label"], baseline_label=prior["label"], command=r["command"],
                           scene=r["scene"], role=r["role"], opened=opened, baseline_open=prior["opened"],
                           baseline_readings=prior.get("uplink", {}).get("readings"))
                if opened is not None:
                    u, shard = scoring.score_case(dict(r, reference_sha256=r["ref_sha256"]), number, opened, readers, work, cache)
                    assert all(u[k] == 0 for k in ("capture_watchdog", "opus_watchdog", "sender_watchdog"))
                    row["uplink"] = u
                    with ZipFile(shard) as logs:
                        for name in logs.namelist():
                            if name.endswith(".log"):
                                data = logs.read(name)
                                dest.writestr(name, data)
                                files[name] = player.study.sha256(data).hexdigest()
                rows.append(row)
                print(r["label"], opened, flush=True)
        assert len(rows) == 9 and len(files) == 3*sum(r["opened"] is not None for r in rows)
        sources = (player.PLAN, player.OUT, previous.OUT, player.PROTO, Path(__file__).resolve().relative_to(Path.cwd()))
        P.save(work/OUT.name, dict(records=rows, sources={str(p): P.sha(p) for p in sources}, files=files,
                                   archive_sha256=P.sha(work/ARCHIVE.name), archive=ARCHIVE.name,
                                   qwen=qwen, turbo=old["turbo"], decoder=old["decoder"]))
        (work/ARCHIVE.name).replace(ARCHIVE)
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
