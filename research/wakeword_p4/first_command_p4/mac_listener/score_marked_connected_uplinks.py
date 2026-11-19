"""Read the full selected P4 uplinks and pair each command gain change at the same scene."""
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import check_marked_connected_panel as mixed
import check_panel_qwen_asr as asr

study, ch = mixed.study, mixed.ch
OUT = ch.RESULTS / "4090-c161-marked-connected-qwen.json"


def main():
    proc, model, ident = asr.load()
    assert ident == ch.read(asr.OUT)["model"]
    uplink = mixed.tx.panel.grid.gap.uplink
    decoder = uplink.decoder_api()
    rows, groups, paired = [], defaultdict(Counter), defaultdict(dict)
    dry = []
    for (voice, command), entries in study.sources().items():
        src, plain = entries["Hey Nyx"]
        wave = plain[src["end"]:]
        text = asr.transcribe(proc,model,wave)
        dry.append(dict(voice=voice, command=command, pcm_sha256=ch.digest(wave), text=text,
                        exact=asr.normalized(text) == asr.normalized(command)))
    for path, lane in ((study.GUEST,"records"), (mixed.OUT,"guest")):
        report = ch.read(path)
        archive = path.with_suffix(".zip")
        assert ch.sha(archive) == report["archive_sha256"]
        with ZipFile(archive) as z:
            for r in report[lane]:
                s = r["source"]
                name = r.get("variant", s["target"] + "." + s["view"])
                want = asr.normalized(s["command"])
                result = dict(index=r["index"], report=path.name, voice=s["voice"], command=s["command"],
                        external_db=s["external_db"], variant=name, opened=r["opened"], timely=r["timely"],
                        turbo=r["turbo"], small=r["small"], qwen=False)
                if r["opened"]:
                    cap = r["uplink"]
                    uart = z.read(f'{r["index"]:02}-opus.log').decode().replace("\r\n","\n")
                    pkts = uplink.packets(uart,r["index"],cap["samples"],0)
                    x = uplink.decode(decoder,pkts)
                    assert len(pkts) == cap["packets"] and ch.digest(x) == cap["pcm_sha256"]
                    text = asr.transcribe(proc,model,x)
                    result.update(pcm_sha256=ch.digest(x), packets=len(pkts), text=text,
                                  qwen=asr.normalized(text)[-len(want):] == want)
                groups[name].update(trials=1, opens=int(bool(r["opened"])), timely=int(r["timely"]),
                        turbo=int(r["turbo"]), small=int(r["small"]), qwen=int(result["qwen"]))
                paired[s["voice"],s["command"],s["external_db"]][name] = result
                rows.append(result)
    deltas = defaultdict(Counter)
    for (_,_,db), cases in paired.items():
        if "marked" not in cases:
            continue
        for source in ("marked","fixed-v5.both"):
            a,b = cases[source],cases["marked_louder_command"]
            for tag in ("qwen","turbo"):
                deltas[f"{source}.{tag}"].update(trials=1, original=int(a[tag]), louder=int(b[tag]),
                                         gained=int(b[tag] and not a[tag]), lost=int(a[tag] and not b[tag]))
    assert len(rows) == 144 and all(v["trials"] == 12 for v in deltas.values()) and len(deltas) == 4
    ch.save(OUT, dict(records=rows,summary=groups,paired=deltas,dry=dry,model=ident,
        sources={p.name:ch.sha(p) for p in (study.GUEST,mixed.OUT,Path(asr.__file__),Path(__file__))},
        policy="Decode all selected Opus packets on opened streams, verify the decoded PCM and apply "
        "the same English-only greedy ASR. Closed cases count as incomplete. Twelve preassigned gains "
        "have directly paired original v5, marked speech, reversed final code and louder marked command. "
        "Remaining 24 louder-command cases extend coverage; no gain comparison on unpaired inputs."))


if __name__ == "__main__":
    main()
