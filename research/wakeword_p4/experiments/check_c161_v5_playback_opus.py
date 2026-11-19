"""Measure Opus compression of the frozen playback cue before sending it through the room."""
import ctypes
import hashlib
import io
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import check_c161_coded_cue_v4 as scenes
from check_c161_coded_cue_v5 import HEADER, WAV, COEFF, binding
from check_c161_piper_spellings import PLAN as SPELLINGS
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from replay_c161_pcm_cues_v1 import detect
from run_c161_first_command_v5 import decode, decoder_api, transcripts
from select_c161_synthesized_cue_v2 import load_wav

STEM = "4090-c161-v5-playback-opus"
NAMES = RESULTS / "4090-c161-vad-word-check-fresh-plan.json"


def codec_api():
    lib = decoder_api()
    lib.opus_encoder_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.opus_encoder_create.restype = ctypes.c_void_p
    lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
    lib.opus_encoder_destroy.restype = None
    lib.opus_encode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int32]
    lib.opus_encoder_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int]
    return lib


def roundtrip(lib, pcm, application, bitrate):
    status = ctypes.c_int()
    enc = lib.opus_encoder_create(16000, 1, application, ctypes.byref(status))
    assert enc and status.value == 0
    packets = []
    try:
        assert lib.opus_encoder_ctl(enc, 4002, ctypes.c_int(bitrate)) == 0  # OPUS_SET_BITRATE
        delay = ctypes.c_int()
        assert lib.opus_encoder_ctl(enc, 4027, ctypes.byref(delay)) == 0  # OPUS_GET_LOOKAHEAD
        tail = delay.value + (-(len(pcm)+delay.value)) % 320
        frames = np.pad(pcm,(0,tail)).reshape(-1,320)
        out = (ctypes.c_ubyte * 1275)()
        for frame in frames:
            size = lib.opus_encode(enc, frame.ctypes.data, 320, out, len(out))
            assert 0 < size <= len(out)
            packets.append(bytes(out[:size]))
    finally:
        lib.opus_encoder_destroy(enc)
    samples = decode(lib, packets)[delay.value:delay.value+len(pcm)].copy()
    assert len(samples) == len(pcm)
    binary = b"".join(struct.pack("<H",len(pkt))+pkt for pkt in packets)
    return samples, binary, dict(lookahead=delay.value, packets=len(packets), packet_bytes=sum(map(len,packets)),
                                 packet_sha256=hashlib.sha256(binary).hexdigest(), pcm_sha256=digest(samples))


def archive_cue(z, key, pcm, binary):
    with io.BytesIO() as f:
        sf.write(f,pcm,16000,format="WAV",subtype="PCM_16")
        z.writestr(key+".wav",f.getvalue())
    z.writestr(key+".packets",binary)


def main():
    spec = RESULTS / "4090-c161-coded-cue-v5-source.json"
    fixtures = RESULTS / "4090-c161-coded-cue-v5-commands.json"
    prior = {(r["source"]["voice"],r["source"]["scene"],r["source"]["channel"]):r
             for r in read(fixtures)["records"] if r["source"]["role"] == "v5"}
    reference = read(spec)
    assert sha(COEFF) == reference["coefficients_sha256"]
    cue = load_wav(dict(path=WAV, sha256=reference["wav_sha256"]))
    reader, identity = load_reader()
    lib = codec_api()
    version = lib.opus_get_version_string().decode()
    files = {p.name:sha(p) for p in (spec, fixtures, HEADER, COEFF, WAV, SPELLINGS, NAMES, Path(__file__),Path(scenes.__file__))}
    controls = [(p.name,r,load_wav(r)) for p in (SPELLINGS, NAMES) for r in read(p)["records"]]
    for _,r,pcm in controls:
        assert digest(pcm) == r["pcm_sha256"]
    archive = RESULTS/f"{STEM}-cues.zip"
    rows = []
    configs = [("raw",None,None)] + [(f"{name}-{kb}",app,kb*1000) for name,app in (("voip",2048),("audio",2049)) for kb in (12,16,24,48)]
    with TemporaryDirectory() as t, ZipFile(archive.with_suffix(".tmp"),"w",ZIP_DEFLATED) as z:
        detector = binding(Path(t)/"cue.so",coded="v5")
        for key,app,br in configs:
            wave, metadata = cue, {}
            if app is not None:
                wave, packets, metadata = roundtrip(lib,cue,app,br)
                archive_cue(z,key,wave,packets)
            row = dict(mode=key, bitrate=br, application=app, cue_pcm_sha256=digest(wave), metadata=metadata,
                       dry_transcript=transcripts(reader,wave), trials=[], controls=[])
            with patch.object(scenes,"TARGET","v5"):
                for meta,pcm in scenes.command_streams(wave,noise_reference=cue):
                    if meta["role"] != "v5":
                        continue
                    events = detect(detector,(pcm,))
                    orig = prior[(meta["voice"],meta["scene"],meta["channel"])]
                    if app is None:
                        assert digest(pcm) == orig["pcm_sha256"] and events == orig["events"]
                    row["trials"].append(dict(source=meta, pcm_sha256=digest(pcm), events=events,
                                             hits=[v["t"] for v in events if meta["start"] <= v["t"] < meta["end"]]))
            for label,r,pcm in controls:
                received = pcm if app is None else roundtrip(lib,pcm,app,br)[0]
                row["controls"].append(dict(plan=label, source_pcm=r["pcm_sha256"], pcm_sha256=digest(received),
                                            utterances=len(r["timeline"]), samples=len(pcm), events=detect(detector,(received,))))
            rows.append(row)
            print(key, sum(bool(v["hits"]) for v in row["trials"]), flush=True)
    archive.with_suffix(".tmp").replace(archive)
    assert len(prior) == 36 and len(controls) == 8
    save(RESULTS/f"{STEM}.json",dict(files=files, opus_version=version, archive_sha256=sha(archive),
         archive_format="decoded 16 kHz WAV and sequences of uint16 little-endian length plus packet payload",
         settings="mono 16 kHz, 20 ms, VBR default, decode and remove encoder-reported lookahead, no packet loss",
         source_order="encode only the clean cue before room filtering; preserve original command, playback level, noise trace and room clocks. Controls independently encoded in complete streams.",
         asr=identity, records=rows))


if __name__ == "__main__":
    main()
