"""Materialize the exact WebRTC VAD source used in the C161 ambient-history replay."""
import argparse
import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
from urllib.request import urlopen

NAME = "webrtcvad_wheels-2.0.14.post1"
URL = "https://files.pythonhosted.org/packages/5a/8d/0597fa376df2f11dbd28fd4dca333d063d6f8fd993eb32563b80d09c6fc6/webrtcvad_wheels-2.0.14.post1.tar.gz"
SHA = "c740e93d24b5d0d7ecdd5548c43e37e2c88564826e869c861d5e3fa7f1cee7ff"


def materialize(data, output):
    if hashlib.sha256(data).hexdigest() != SHA:
        raise ValueError("WebRTC VAD source archive hash mismatch")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        component = Path(tmp)/"nyx_webrtc_vad"
        component.mkdir()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive.getmembers():
                path = Path(member.name).relative_to(NAME)
                if not member.isfile() or (path != Path("LICENSE") and path.parts[:2] != ("cbits","webrtc")):
                    continue
                if not (component/path).resolve().is_relative_to(component.resolve()):
                    raise ValueError(path)
                target = component/path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
        common = component/"cbits/webrtc/common_audio"
        sources = sorted(common.glob("signal_processing/*.c")) + sorted(common.glob("vad/*.c"))
        if not sources or not (common/"vad/include/webrtc_vad.h").is_file():
            raise ValueError("Incomplete WebRTC VAD source")
        names = "\n".join('        "'+p.relative_to(component).as_posix()+'"' for p in sources)
        (component/"CMakeLists.txt").write_text("idf_component_register(\n    SRCS\n"+names+
            '\n    INCLUDE_DIRS "cbits"\n    PRIV_REQUIRES pthread)\n'
            'target_compile_definitions(${COMPONENT_LIB} PRIVATE WEBRTC_POSIX)\n')
        component.rename(output)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new ESP-IDF component directory")
    parser.add_argument("--archive", type=Path, help="local copy of the pinned source distribution")
    args=parser.parse_args()
    if args.archive:
        data=args.archive.read_bytes()
    else:
        with urlopen(URL, timeout=30) as response:
            data=response.read()
    materialize(data,args.output)
