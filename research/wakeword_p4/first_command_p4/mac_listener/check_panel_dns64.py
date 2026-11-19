"""Apply the DNS-trained speech denoiser to verified whole received packets and re-read commands."""
from collections import Counter, defaultdict
from hashlib import file_digest
from pathlib import Path
import shutil
import sys
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import torch

import check_panel_parakeet as independent

diag,ch=independent.diag,independent.ch
ROOT=Path(".local/wakeword/c161-dns64")
REV="8afd7c166699bb3c8b2d95b6dd706f71e1075df0"
OUT=ch.RESULTS/"4090-c161-panel-dns64-command-audio.json"
DIGESTS={
    "denoiser/__init__.py":"8f1ef20e7938e5d2e7f27a63c54cc117db13ec40711ee4575b66fb232bbdd652",
    "denoiser/demucs.py":"8e9c21935c647e24f31cefcc63a298cb2a1c25bc99aab44bbe63a7b5570836be",
    "denoiser/resample.py":"3e8ea258036660b7d33415794fe09ee010510f4d760bdfc5d5de268d6efb40f5",
    "denoiser/utils.py":"c05fbdebadda27ef7a49e6f5c557a4c4bf3a8f1e67ad65fac22f1cb5ada83321",
    "dns64.th":"a7761ff99a7d5bb69c16a61b3089469805cf3afd82e128103f0f77e79dfead5f",
}


def prepare():
    for name,digest in DIGESTS.items():
        path=ROOT/name
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            url=(f"https://raw.githubusercontent.com/facebookresearch/denoiser/{REV}/{name}" if name.endswith(".py") else
                 "https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns64-a7761ff99a7d5bb6.th")
            with urlopen(url,timeout=60) as src, path.with_suffix(".part").open("wb") as f:
                shutil.copyfileobj(src,f)
            path.with_suffix(".part").replace(path)
        with path.open("rb") as f:
            assert file_digest(f,"sha256").hexdigest()==digest
    sys.path.insert(0,str(ROOT.resolve()))
    from denoiser.demucs import Demucs
    m=Demucs(hidden=64,sample_rate=16000).eval()
    m.load_state_dict(torch.load(ROOT/"dns64.th",weights_only=True,map_location="cpu"))
    return m.cuda()


def main():
    net=prepare()
    small,s_info=ch.load_reader()
    versions=ch.read(diag.BOARD)["versions"]
    assert s_info==versions["small"]
    turbo=diag.reader(versions["turbo"])
    decoder=diag.panel.grid.gap.uplink.decoder_api()
    summary,rows,sources=defaultdict(Counter),[],{}
    with torch.inference_mode():
        for doc in (diag.BOARD,independent.balanced.OUT,independent.timed.OUT,independent.ROOM,independent.MUSIC):
            report=ch.read(doc)
            archive=doc.with_suffix(".zip")
            assert ch.sha(archive)==report["archive_sha256"]
            sources.update({doc.name:ch.sha(doc),archive.name:report["archive_sha256"]})
            paired=doc==independent.balanced.OUT
            if doc in (diag.BOARD,independent.balanced.OUT):
                cases=independent.panel_inputs(report,paired)
            elif doc==independent.timed.OUT:
                cases=independent.ordinary_inputs(report)
            else:
                cases=independent.willow_inputs(report,doc==independent.MUSIC)
            with ZipFile(archive) as logs:
                for row,item in cases:
                    group=row["group"]
                    if paired: group="balanced."+group
                    stats=summary[group]
                    stats["trials"]+=1
                    entry=dict(**row,source_report=doc.name,study_group=group)
                    if item:
                        i,samples,digest,logfile=item
                        uart=logs.read(logfile).decode().replace("\r\n","\n")
                        pkts=diag.panel.grid.gap.uplink.packets(uart,i,samples,0)
                        x=diag.panel.grid.gap.uplink.decode(decoder,pkts)
                        assert ch.digest(x)==digest
                        wave=torch.from_numpy(x.astype(np.float32)/32768).reshape(1,1,-1).cuda()
                        enhanced=net(wave).squeeze().cpu().numpy()
                        assert enhanced.shape==x.shape and np.isfinite(enhanced).all()
                        y=diag.study.grid.room.quantize(enhanced*32768)
                        entry.update(pcm_sha256=digest,enhanced_sha256=ch.digest(y),samples=len(x))
                        stats["opened"]+=1
                        for label,asr in (("small",small),("turbo",turbo)):
                            reading=diag.panel.grid.gap.uplink.recognition(asr,y,row["command"])
                            before,after=row[label],reading["complete_suffix"]
                            entry[label+"_enhanced"]=dict(text=reading["text"],complete=after)
                            stats.update({label+"_raw":int(before),label+"_enhanced":int(after),label+"_gained":int(after and not before),
                                          label+"_lost":int(before and not after)})
                    rows.append(entry)
            print(doc.name,len(rows),flush=True)
    assert len(rows)==174
    ch.save(OUT,dict(records=rows,summary=summary,model=dict(name="DNS64",revision=REV,files=DIGESTS,
        device=torch.cuda.get_device_name(),torch=torch.__version__),small=s_info,turbo=versions["turbo"],
        sources={**sources,Path(__file__).name:ch.sha(Path(__file__))},
        policy="Full received mono 16 kHz stream, official DNS64 speech model in FP32 on RTX 4090. Use the "
               "whole-file output at unity mix, preserve packet samples, no command crop, annotation or transcription "
               "prompt. Includes previously specified Willow conditions, missed OPENs in all denominators. No P4 load."))


if __name__=="__main__":
    main()
