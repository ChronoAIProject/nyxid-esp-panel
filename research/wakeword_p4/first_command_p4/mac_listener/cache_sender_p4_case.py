"""Save each verified sender P4 pass and retain UART evidence from failed transfers."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import check_connected_playback_codecs as playback


def capture(work, destination, index, case, readers, builds, provenance):
    host_index, source, build, role, mic, ref, expected = case
    key = dict(host_index=host_index, index=index, source=source, build=build, role=role, expected=expected,
               mic_sha256=playback.ch.digest(mic), reference_sha256=playback.ch.digest(ref), provenance=provenance)
    checksum = sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    cache = Path(".local/wakeword/c161-sender-p4-cases") / provenance["study"]
    cache.mkdir(parents=True, exist_ok=True)
    shard = cache/f"{index:03}-{checksum}.zip"
    if not shard.exists():
        with TemporaryDirectory(dir=cache) as folder:
            partial = Path(folder)/"case.zip"
            for attempt in range(3):
                try:
                    with ZipFile(partial,"w",ZIP_DEFLATED) as part:
                        row = playback.guest_case(work,part,index,case,readers,builds)
                        part.writestr("case.json",json.dumps(dict(key_sha256=checksum,row=row)))
                    partial.replace(shard)
                    break
                except (AssertionError,TimeoutError,ValueError) as error:
                    # Retry only this input. Keep the raw UART produced by the listener and encoder.
                    with ZipFile(cache/f"{shard.stem}-retry-{attempt}.zip","w",ZIP_DEFLATED) as evidence:
                        evidence.writestr("exception.txt",repr(error))
                        for log in ("serial.log","opus.log"):
                            if (work/log).exists():
                                evidence.write(work/log,log)
                    if attempt==2:
                        raise
    with ZipFile(shard) as part:
        assert part.testzip() is None
        stored = json.loads(part.read("case.json"))
        assert stored["key_sha256"]==checksum
        row = stored["row"]
        assert (row["source"],row["opened"],row["mic_sha256"],row["reference_sha256"]) == (
                source,expected["opened"],key["mic_sha256"],key["reference_sha256"])
        for n in part.namelist():
            if n!="case.json":
                destination.writestr(n,part.read(n))
    for path in sorted(cache.glob(shard.stem+"-retry-*.zip")):
        destination.write(path,"case-retries/"+path.name)
    return row
