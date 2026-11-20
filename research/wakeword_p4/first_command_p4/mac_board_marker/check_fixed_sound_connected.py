"""Check the standalone fixed-sound C++ entry point on the 60 connected synthetic addresses."""
import argparse
from collections import defaultdict
from ctypes import CDLL, byref, POINTER, c_int16, c_size_t, c_uint64, c_void_p, cast, create_string_buffer
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory
import wave
from zipfile import ZipFile

from run_guest import payload
from check_replay import RESULTS


def check(board, work):
    impl = board/"components/panel_services/fixed_marker/board_marker.cpp"
    obj = work/"fixed.so"
    subprocess.run(["c++","-O2","-std=c++17","-shared","-fPIC","-DNYX_PANEL_FIXED_SOUND=1",
                    impl,"-o",obj],check=True)
    lib=CDLL(str(obj))
    lib.board_marker_bytes.restype=c_size_t
    lib.board_marker_begin.argtypes=[c_void_p]
    lib.board_marker_begin.restype=c_void_p
    lib.board_marker_feed_fixed.argtypes=[c_void_p,POINTER(c_int16),c_size_t,c_uint64]
    lib.board_marker_feed_fixed.restype=c_uint64
    history=RESULTS/"4090-c161-fixed-playback-only.json"
    data=json.loads(history.read_text())
    archive=RESULTS/"4090-c161-connected-nick-willow-audio.zip"
    assert sha256(archive.read_bytes()).hexdigest()==data["sources"][archive.name]
    rows=[]
    summary=defaultdict(lambda:dict(cases=0,opened=0,before_command=0,both_routes_opened=0,both_routes_before_command=0))
    with ZipFile(archive) as z:
        for case in data["p4_replays"]:
            if case["group"]!="connected":
                continue
            with wave.open(BytesIO(z.read(case["label"].removeprefix("connected/")))) as wav:
                assert wav.getparams()[:3]==(1,2,16000)
                mic=wav.readframes(wav.getnframes())
            raw=payload(mic,bytes(len(mic)))
            assert sha256(raw).hexdigest()==case["input_sha256"]
            n,=struct.unpack_from("<I",raw)
            state=create_string_buffer(lib.board_marker_bytes())
            handle=lib.board_marker_begin(state)
            interleaved=create_string_buffer(raw[4:])
            cues=[]
            for at in range(0,n,256):
                src=cast(byref(interleaved,4*at),POINTER(c_int16))
                end=lib.board_marker_feed_fixed(handle,src,256,at)
                if end:
                    cues.append(end)
            clocks=case["cue_only"]
            opened=(cues[0]+255)//256*256 if cues else None
            assert cues==clocks["C"] and ([opened] if cues else [])==clocks["O"]
            old=case["baseline"]["O"]
            s=summary[case["role"]]
            s["cases"]+=1
            s["opened"]+=int(opened is not None)
            s["both_routes_opened"]+=bool(old)
            s["before_command"]+=int(opened is not None and case["cue_start"]<=opened<case["command_at"])
            s["both_routes_before_command"]+=int(bool(old and case["cue_start"]<=old[0]<case["command_at"]))
            rows.append(dict(label=case["label"],role=case["role"],cue=cues,first_open=opened,
                             input_sha256=case["input_sha256"],command_at=case["command_at"],
                             dual_open=old,old_fixed_uart_sha256=case["cue_only_uart_sha256"]))
    assert len(rows)==60
    return dict(host_state_bytes=lib.board_marker_bytes(),source_sha256=sha256(impl.read_bytes()).hexdigest(),
                prior_report_sha256=sha256(history.read_bytes()).hexdigest(),records=rows,summary=summary)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("board",type=Path)
    with TemporaryDirectory() as folder:
        print(json.dumps(check(p.parse_args().board.resolve(),Path(folder)),indent=2))
