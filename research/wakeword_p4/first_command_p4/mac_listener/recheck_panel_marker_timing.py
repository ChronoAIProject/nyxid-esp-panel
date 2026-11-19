"""Replay every timed-panel input with a 16 ms overrun on the exact same firmware and PCM."""
from pathlib import Path
import re
import struct
from tempfile import TemporaryDirectory

import package_timed_panel as pack

ch, study, panel = pack.ch, pack.study, pack.panel
OUT = pack.BOARD.with_name(pack.HOST.stem + "-timing.json")


def main():
    history = ch.read(pack.BOARD)
    targets = {r["host_index"]:r for r in history["records"] if r["finish"][6]}
    builds = dict(abc=pack.single.BUILD, bca=panel.WORK.with_name(panel.WORK.name+"-timed")/"build")
    for name,key in (("abc","original"),("bca","panel_order")):
        assert ch.sha(builds[name]/"nyx_mac_listener.bin") == history["versions"][key]
    rows = []
    with TemporaryDirectory() as folder:
        work = Path(folder)
        selected = [(i,m) for i,m in enumerate(study.simultaneous()) if i in targets]
        for repeat in range(3):
            for i,(meta,name,role,mic,ref) in selected:
                old = targets[i]
                assert (meta,name,role,ch.digest(mic),ch.digest(ref)) == (
                    old["source"],old["name"],old["role"],old["mic_sha256"],old["reference_sha256"])
                payload = struct.pack("<3I",0x4652594e,old["index"],len(mic)) + mic.tobytes() + ref.tobytes()
                log = work/"serial.log"
                panel.phase.listener.boot(builds[name], log, work/"image.bin", b"LISTENER PASS", (2<<20,payload))
                uart = log.read_text()
                assert panel.phase.listener.values(uart,"READY") == [old["index"],len(mic),256,81600]
                finish = panel.phase.listener.values(uart,"FINISH")
                assert finish[:4] == old["finish"][:4]
                assert [int(t) for t in re.findall(r"^OPEN (\d+) ",uart,re.M)] == old["opened"]
                rows.append(dict(index=old["index"],name=name,repeat=repeat,finish=finish,original_finish=old["finish"],
                                 uart_sha256=ch.sha(log)))
    assert len(rows)==3*len(targets)
    ch.save(OUT,dict(records=rows,sources={p.name:ch.sha(p) for p in (pack.BOARD,Path(__file__))},
        policy="Repeat all per-input QEMU timing overruns three times on the same build and M/R. "
               "Check OPEN and FIFO boundaries against the original runs."))


if __name__ == "__main__":
    main()
