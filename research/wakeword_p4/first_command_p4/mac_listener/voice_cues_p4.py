"""Check external and synchronized panel playback for every coded multi-voice room input."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import replay as listener
from code_gain_guard import score
from prefix_guard import BUILD
from reference_signature import qualify
from voice_cues import PACK, RESULTS, STEM, binding, digest, read, save, scene_materials, sha


def main():
    source=RESULTS/f"{STEM}-rooms.json"
    prior={r["pcm_sha256"]:r for r in read(source)["records"] if r["mode"]=="coded"}
    assert len(prior)==16
    fw=BUILD/"nyx_mac_listener.bin"
    assert sha(fw)==read(RESULTS/"4090-c161-mac-reference-prefix-signature-rooms.json")["firmware_sha256"]
    stash=listener.WORK/"coded-voices"/sha(fw)[:16]
    stash.mkdir(parents=True,exist_ok=True)
    rows=[]
    with TemporaryDirectory() as tmp:
        temp=Path(tmp)
        full=binding(temp/"full.so",coded="v5")
        prefix=binding(temp/"pair.so",coded="aligned")
        for src,mic,cue,_ in scene_materials():
            if src["mode"]!="coded":
                continue
            actual=prior.pop(digest(mic))
            assert src["acoustic"]=="mixture"
            for role in ("external","panel"):
                i=len(rows)
                ref=np.zeros_like(mic)
                if role=="panel":
                    start=src["cue_at"]
                    ref[start:start+len(cue)]=cue
                out=stash/f"{i:02}-{digest(mic)[:16]}-{digest(ref)[:16]}.json"
                if not out.exists():
                    row=score(temp,dict(**src,playback=role),mic,ref,full,i,build=BUILD,prefix=prefix,qualify=qualify)
                    save(out,row)
                r=read(out)
                assert r["source"]==dict(**src,playback=role) and (r["mic_sha256"],r["reference_sha256"])==(digest(mic),digest(ref))
                assert r["c161"]==actual["c161"] and r["fixed"]==[e["t"] for e in actual["fixed"]]
                rows.append(r)
                print(i,src["voice"],role,r["opened"],flush=True)
    assert len(rows)==32 and not prior
    save(RESULTS/f"{STEM}-p4.json",dict(records=rows,firmware_sha256=sha(fw),qemu_sha256=sha(listener.QEMU),
         inputs_sha256=sha(source),dry_pack_sha256=sha(PACK)))


if __name__=="__main__":
    main()
