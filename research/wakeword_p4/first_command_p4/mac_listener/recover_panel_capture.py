"""Complete a known interrupted simulation from its UART/Opus prefix after validating all clocks."""
import argparse
from hashlib import sha256
from pathlib import Path
import re
from unittest.mock import patch
from zipfile import ZipFile

import check_panel_middle_p4 as middle

tx, pair, ch = middle.tx, middle.pair, middle.ch


def main(mode, archive, digest):
    assert ch.sha(archive)==digest
    if mode == "marked":
        import check_marked_connected_panel as marked
        run, output = marked.main, marked.OUT
    else:
        run, output = {"timer":(tx.guest_main,tx.BOARD),"pair":(pair.guest_main,pair.BOARD),
                   "middle":(middle.main,middle.OUT)}[mode]
    assert not output.exists()
    capture=tx.panel.capture
    with ZipFile(archive) as cached:
        assert cached.testzip() is None and len(cached.namelist())==len(set(cached.namelist()))
        saved = {int(n[7:-4]) for n in cached.namelist() if re.fullmatch(r"events/[0-9]+\.txt",n)}
        assert saved==set(range(len(saved))) and saved
        def checked(work,z,idx,case,small,turbo,decoder,builds,*,tx_pair=None):
            if idx not in saved:
                return capture(work,z,idx,case,small,turbo,decoder,builds,tx_pair=tx_pair)
            source_idx,s,name,role,mic,ref,known=case
            assert (ch.digest(mic),ch.digest(ref))==(known["pcm_sha256"],known["reference_sha256"])
            member=f"events/{idx:03}.txt"
            uart=cached.read(member)
            log=uart.decode()
            assert log.endswith("LISTENER PASS\n")
            vals=tx.panel.phase.listener.values
            assert vals(log,"READY")==[idx,len(mic),256,81600]
            hits=[tuple(map(int,x.split())) for x in re.findall(r"^HITS (.+)$",log,re.M)]
            assert [c for _,c,_ in hits if c>=0]==known["c161"]
            assert [f for _,_,f in hits if f>=0]==[e["t"] for e in known["fixed"]]
            for event,label in (("OPEN","opened"),("C161_ALLOW","allowed_model"),("V5_ALLOW","allowed_fixed")):
                assert [int(t) for t in re.findall("^"+event+r" (\d+)(?: |$)",log,re.M)]==known[label]
            finish=vals(log,"FINISH")
            opened=known["opened"]
            emitted,first,count,blocks=finish[:4]
            assert blocks==(len(mic)+255)//256 and emitted==blocks*256//160*160
            assert first==(max(0,opened[0]//160*160-32000) if opened else -1)
            assert count==(emitted-first if opened else 0)
            z.writestr(member,uart)
            r=dict(index=idx,host_index=source_idx,name=name,role=role,source=s,opened=opened,
                   mic_sha256=known["pcm_sha256"],reference_sha256=known["reference_sha256"],
                   allowed_model=known["allowed_model"],allowed_fixed=known["allowed_fixed"],finish=finish,
                   events_sha256=sha256(uart).hexdigest(),
                   recovered_archive_sha256=digest,small=False,turbo=False)
            if opened and role!="panel":
                path=f"{idx:02}-opus.log"
                data=cached.read(path)
                serial=data.decode().replace("\r\n","\n")
                packets=tx.panel.grid.gap.uplink.packets(serial,idx,count,0)
                tx.panel.grid.gap.uplink.packets(serial,idx,count,1)
                pcm=tx.panel.grid.gap.uplink.decode(decoder,packets)
                a,b=(tx.panel.grid.gap.uplink.recognition(model,pcm,s["command"]) for model in (small,turbo))
                r.update(uplink=dict(samples=count,packets=len(packets),pcm_sha256=ch.digest(pcm),recognition=a,turbo=b),
                         small=a["complete_suffix"],turbo=b["complete_suffix"])
                z.writestr(path,data)
            return r
        with patch.object(tx.panel,"capture",checked):
            run()
    print(mode,"recovered",len(saved),"of",len(ch.read(output)["guest" if mode == "marked" else "records"]))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("timer","pair","middle","marked"))
    p.add_argument("archive",type=Path)
    p.add_argument("sha256")
    args=p.parse_args()
    main(args.mode,args.archive,args.sha256)
