"""Measure toolchain worker stack frames and interleaved timing on an identical M/R stream."""
from pathlib import Path
import re
import shlex
import struct
import subprocess
from tempfile import TemporaryDirectory

import check_panel_middle_p4 as middle

tx,ch=middle.tx,middle.ch
OUT=ch.RESULTS/"4090-c161-panel-guard-cost.json"


def stack_frame(build,work):
    entry,=[r for r in ch.read(build/"compile_commands.json") if r["file"].endswith("/main/listener.cpp")]
    args=shlex.split(entry["command"])
    obj=work/"listener.o"
    args[args.index("-o")+1]=str(obj.resolve())
    assert not any(x in args for x in ("-MF","-MD","-MMD"))
    subprocess.run([*args,"-fstack-usage"],cwd=entry["directory"],capture_output=True,check=True)
    line,=[s for s in obj.with_suffix(".su").read_text().splitlines() if "void {anonymous}::worker(void*)" in s]
    _,value,kind=line.split("\t")
    return dict(worker_frame=int(value),stack_kind=kind,compiler=Path(args[0]).name,
                listener_source_sha256=ch.sha(Path(entry["file"])))


def main():
    builds=dict(electrical=tx.balanced.BUILD,timer=tx.WORK/"build",pair=middle.pair.WORK/"build",middle=middle.WORK/"build")
    versions={}
    rows=[]
    source,mic,ref=next((s,x,r) for s,n,role,x,r in tx.study.simultaneous() if n=="bca" and role=="panel" and s["external_db"]==0)
    pair=source["local_at"]+tx.event_time()
    with TemporaryDirectory() as folder:
        work=Path(folder)
        for name,b in builds.items():
            fw=b/"nyx_mac_listener.bin"
            versions[name]=dict(bytes=fw.stat().st_size,sha256=ch.sha(fw),**stack_frame(b,work))
        for run in range(3):
            for name,b in builds.items():
                i=len(rows)
                protocol=0x4652594e if name=="electrical" else 0x5854594e
                p=struct.pack("<3I",protocol,i,len(mic))+mic.tobytes()+ref.tobytes()
                if name!="electrical": p+=struct.pack("<I",pair)
                log=work/"serial.log"
                tx.panel.phase.listener.boot(b,log,work/"image.bin",b"LISTENER PASS",(2<<20,p))
                text=log.read_text()
                assert tx.panel.phase.listener.values(text,"READY")==[i,len(mic),256,81600] and not re.search(r"^OPEN ",text,re.M)
                finish=tx.panel.phase.listener.values(text,"FINISH")
                rows.append(dict(build=name,run=run,finish=finish,mean_block_us=finish[4]/finish[3],uart_sha256=ch.sha(log)))
    ch.save(OUT,dict(builds=versions,records=rows,source=source,mic_sha256=ch.digest(mic),reference_sha256=ch.digest(ref),
        sources={p.name:ch.sha(p) for p in (middle.HEADER,tx.HEADER,middle.pair.HEADER,Path(__file__))},
        policy="Three interleaved runs per firmware on the same original panel-only intact M/R and identical "
               "no-OPEN. QEMU timing and internal heap. Compile listener.cpp again with the active target compiler "
               "flags and -fstack-usage; worker frame excludes callee stack and RTOS hardware high-water."))


if __name__=="__main__":
    main()
