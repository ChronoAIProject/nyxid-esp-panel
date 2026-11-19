"""Compare full decoded Opus with the established 100 ms pre-OPEN command window on panel overlaps."""
from collections import Counter, defaultdict
from pathlib import Path
import sys
from zipfile import ZipFile

from faster_whisper import WhisperModel

import check_panel_marker_p4 as panel

ch = panel.ch
OUT = ch.RESULTS / "4090-c161-panel-command-window.json"


def main(paths):
    reader, small_info = ch.load_reader()
    readers = {"small":reader}
    decoder = panel.grid.gap.uplink.decoder_api()
    summary, rows, cache, sources = defaultdict(Counter), [], {}, {}
    for path in paths:
        report = ch.read(path)
        versions = report["versions"]
        assert versions["small"] == small_info and ch.sha(path.with_suffix(".zip")) == report["archive_sha256"]
        turbo_info = versions["turbo"]
        model = panel.prior.prior.prior.audio.prior.context.prior.audit.TURBO
        assert all(ch.sha(model/name)==h for name,h in turbo_info["files"].items())
        if "turbo" not in readers:
            readers["turbo"] = WhisperModel(str(model),device="cuda",compute_type="float16",cpu_threads=1,num_workers=1)
        tag = "timed" if "-timed-" in path.name else "weak"
        sources[path.name] = ch.sha(path)
        with ZipFile(path.with_suffix(".zip")) as logs:
            for r in report["records"]:
                if r["role"] == "panel":
                    continue
                group = ".".join((tag,r["name"],r["role"]))
                rec = dict(report=path.name,index=r["index"],op=r["opened"],level_db=r["source"]["external_db"])
                totals = summary[group]
                totals["trials"] += 1
                if r["opened"]:
                    cap = r["uplink"]
                    pcm, packets = panel.prior.prior.prior.audio.decoded(logs,f'{r["index"]:02}-opus.log',
                                            decoder,r["index"],cap["samples"],0,cap["pcm_sha256"])
                    assert packets == cap["packets"]
                    cut = max(0,r["opened"][0]-r["finish"][1]-1600)
                    rec.update(cut=cut, pcm_sha256=ch.digest(pcm), cropped_pcm_sha256=ch.digest(pcm[cut:]))
                    command = r["source"]["command"]
                    key = (rec["cropped_pcm_sha256"],command)
                    if key not in cache:
                        cache[key] = {n:panel.grid.gap.uplink.recognition(m,pcm[cut:],command) for n,m in readers.items()}
                    rec["after"] = cache[key]
                    totals["opens"] += 1
                    for name in readers:
                        before = r[name]
                        after = rec["after"][name]["complete_suffix"]
                        totals.update({name+"_full":before,name+"_cropped":after,name+"_gained":after and not before,
                                       name+"_lost":before and not after})
                rows.append(rec)
        print(path.name,len(rows),flush=True)
    ch.save(OUT,dict(records=rows,summary=summary,sources={**sources,Path(__file__).name:ch.sha(Path(__file__))},
        small=small_info,turbo=turbo_info,policy="Identical received Opus, full FIFO baseline versus samples from "
        "100 ms before the observed OPEN. No cue-end annotation or expected words enter the transcriber. Count "
        "closed trials as incomplete. Postprocessing only; reuse the same 100 ms window as check_fixed_cue_over_bot.py."))


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
