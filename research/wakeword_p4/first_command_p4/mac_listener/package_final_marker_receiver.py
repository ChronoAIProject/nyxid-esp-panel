"""Bundle original v5 paired mic/reference fixtures and the tested research-only microphone header."""
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import soundfile as sf

import check_final_marker_p4 as board
import compare_room_willow_uplink as room
from package_context_playback import add, formatted

host, ch = board.host, board.ch
PACK = ch.RESULTS/"c161-final-marker-mac-ab.zip"


def main():
    report = ch.read(board.OUT)
    board_archive = board.OUT.with_suffix(".zip")
    assert ch.sha(board_archive) == report["archive_sha256"]
    observed = {(r["board"]["source"]["group"],r["board"]["source"]["scene"],r["mode"],r["local"]):r for r in report["records"]}
    selected = {}
    for r in ch.read(host.OUT)["records"]:
        if r["opened"] == r["previous_opened"]:
            continue
        role = (r["group"], "previous_miss" if r["previous_opened"] is None else "advanced")
        selected.setdefault(role, (r["group"], r["index"]))
    assert len(selected) == 4
    pairs = {key:{} for key in selected.values()}
    for meta, mode, local, mic, ref, _, _ in board.grid_trials():
        key = (meta["group"], meta["scene"])
        if key in pairs:
            result = observed[(*key, mode, local)]["board"]
            assert result["source"] == meta and result["mic_sha256"] == ch.digest(mic) and result["reference_sha256"] == ch.digest(ref)
            pairs[key][mode,local] = result
            if local:
                pairs[key].update(mic=mic, ref=ref)
    generated = host.candidate_header().encode()
    header = board.WORK/"src/research/wakeword_p4/first_command_p4/mac_listener/main/final_marker.h"
    assert generated == header.read_bytes()
    coeff = host.HEADER.with_name("c161_coded_cue_coefficients_v5.h")
    files = {"receiver/final_marker.h":generated, "receiver/" + coeff.name:coeff.read_bytes(),
             "speaker/original-v5.wav":host.grid.fit.fixed.CUE.read_bytes()}
    rows = []
    tmp = PACK.with_suffix(".tmp")
    with ZipFile(tmp, "w") as z:
        for name, data in files.items():
            add(z, name, data)
        for (group, reason), key in selected.items():
            case = pairs[key]
            prefix = f"{group}/{key[1]:03}/"
            assets = {name:room.put_audio(z, prefix + name + ".wav", case[name]) for name in ("mic","ref")}
            clocks = {}
            for (mode,local) in (("original",False),("candidate",False),("candidate",True)):
                board_row = case[mode,local]
                label = mode + ("_self" if local else "_external")
                clocks[label] = {k:board_row[k] for k in ("opened","c161","fixed","reference_events")}
                add(z, prefix + label + ".txt", ("\n".join(board_row["uart_events"]) + "\n").encode())
            rows.append(dict(reason=reason, source=case["candidate",False]["source"], assets=assets, clocks=clocks))
        checksums = {n:dict(bytes=len(z.read(n)), sha256=hashlib.sha256(z.read(n)).hexdigest()) for n in z.namelist()}
        add(z, "manifest.json", formatted(dict(model_sha256=report["versions"]["model"], threshold=0.91796875,
            smoothing_steps=5, reference="original v5 detector", firmware_sha256={k:report["versions"][k] for k in ("original","candidate")},
            sources={p.name:ch.sha(p) for p in (host.OUT,board.OUT,board_archive,Path(__file__))}, files=checksums, cases=rows,
            purpose="Paired simulator clocks for Mac board verification. Use the frozen C161, original v5 WAV and "
                    "full original mic uplink. For the candidate mic only, instantiate nyx::FinalMarkerConfirmation "
                    "instead of nyx::CodedCueV5. Keep the original reference matcher and its guard. For digital external "
                    "playback send mic.wav and zero R; for aligned self-play send mic.wav and ref.wav. Rates 16000 Hz, "
                    "mono PCM16. For physical playback use speaker/original-v5.wav. Groups deliberately select the "
                    "first original miss and first earlier OPEN per grid; they do not estimate average performance.")))
    with ZipFile(tmp) as z:
        assert z.testzip() is None
        manifest = json.loads(z.read("manifest.json"))
        for n, item in manifest["files"].items():
            data = z.read(n)
            assert len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"]
        for row in manifest["cases"]:
            for item in row["assets"].values():
                x, rate = sf.read(io.BytesIO(z.read(item["file"])), dtype="int16")
                assert rate == 16000 and ch.digest(x) == item["pcm_sha256"]
    tmp.replace(PACK)
    print(ch.sha(PACK))


if __name__ == "__main__":
    main()
