"""Read the second cohort from matched dry, room, music, and panel stems through the original FIFO."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import soundfile as sf

import check_additional_connected_voices as study

ch, mixed, tx = study.ch, study.mixed, study.tx
balanced = study.prefix.balanced
OUT = ch.RESULTS / (study.STEM + "-stems.json")


def main():
    ref = ch.read(study.GUEST)
    cases = {(r["source"]["index"],r["source"]["external_db"]):r for r in ref["records"] if r["role"] == "marked"}
    archive = study.GUEST.with_suffix(".zip")
    assert ch.sha(archive) == ref["archive_sha256"] and ch.sha(study.PACK) == ch.read(study.OUT)["pack_sha256"]
    v5, _ = tx.study.panel_wave()
    panel, rate = sf.read(mixed.panel.demo.WAVE,dtype="int16")
    music, _, music_id = tx.study.grid.room.rooms.read_noise("music")
    uplink = tx.panel.grid.gap.uplink
    decoder = uplink.decoder_api()
    proc, model, qid = balanced.original.asr.load()
    assert qid == ref["qwen"] and uplink.firmware() == ref["codec"]
    turbo = tx.diag.reader(ref["turbo"])
    def recognize(pcm, command):
        qwen = balanced.evaluate(proc,model,pcm,command)
        whisper = uplink.recognition(turbo,pcm,command)
        return dict(qwen=qwen, turbo=dict(text=whisper["text"],complete=whisper["complete_suffix"]))

    rows, dry, summary = [], [], defaultdict(Counter)
    with ZipFile(study.PACK) as pack, ZipFile(archive) as logs:
        manifest = json.loads(pack.read("manifest.json"))
        assert rate == 16000 and ch.sha(mixed.panel.demo.WAVE) == manifest["panel_sha256"]
        for entry in manifest["records"]:
            idx, command = entry["index"], entry["command"]
            tracks = {k:study.playback.read_wav(pack,t) for k,t in entry["tracks"].items()}
            wave = tracks["marked"]
            end = entry["command_at"]
            np.testing.assert_array_equal(wave[end:],tracks["plain"][end:])
            dry.append(dict(index=idx, voice=entry["voice"], command=command,
                            standalone_suffix=recognize(wave[end:],command), marked=recognize(wave,command),
                            plain=recognize(tracks["plain"],command)))
            group = {k:(entry,x) for k,x in tracks.items()}
            for db in mixed.study.LEVELS:
                r = cases.pop((idx,db))
                s = r["source"]
                scene = mixed.room_scene(s,group,panel,v5,music,music_id)
                external = mixed.room_scene(s,group,np.zeros_like(panel),v5,music,music_id)
                both, cl = scene(wave)
                assert not cl and ch.digest(both) == r["mic_sha256"]
                pcm = balanced.fifo_input(both,r["finish"])
                cap = r["uplink"]
                assert ch.digest(pcm) == cap["fifo_sha256"]
                mic_direct = np.pad(wave.astype(float) * 10**(db/20), (0,len(both)-len(wave)))
                stems = dict(direct=mixed.digitize(mic_direct,s["preamp_gain"])[0],
                             room=external(wave,include_music=False)[0], room_music=external(wave)[0],
                             room_panel=scene(wave,include_music=False)[0], full=both)
                log = logs.read(f'{r["index"]:02}-opus.log').decode().replace("\r\n","\n")
                packets = uplink.packets(log,r["index"],cap["samples"],0)
                decoded = uplink.decode(decoder,packets)
                assert len(packets)==cap["packets"] and ch.digest(decoded)==r["qwen"]["pcm_sha256"]==cap["pcm_sha256"]
                before = {name:recognize(balanced.fifo_input(mic,r["finish"]),command) for name,mic in stems.items()}
                reads = dict(**before,opus=dict(qwen=r["qwen"],turbo=dict(text=cap["turbo"]["text"],complete=r["turbo"])))
                for lane, view in reads.items():
                    summary[lane].update(trials=1,qwen=int(view["qwen"]["complete"]),turbo=int(view["turbo"]["complete"]))
                rows.append(dict(source=s,opened=r["opened"],finish=r["finish"][:4],mic_sha256=r["mic_sha256"],
                                 packets=len(packets),reads=reads))
            print("source",idx,flush=True)
    assert not cases and len(rows)==36 and len(dry)==12
    with TemporaryDirectory(dir=OUT.parent) as folder:
        p = Path(folder)/OUT.name
        ch.save(p,dict(records=rows,speakers=dry,summary=summary,qwen=qid,turbo=ref["turbo"],codec=ref["codec"],
                sources={f.name:ch.sha(f) for f in (study.GUEST,archive,study.PACK,study.OUT,mixed.panel.demo.WAVE,
                        Path(mixed.__file__),Path(balanced.__file__),Path(__file__))},
                policy="Full unprompted transcription on shipped dry PCM and extracted command suffix. "
                       "Match each microphone length, external level, preamp and original OPEN/FIFO boundaries. "
                       "Remove only music or panel or both before microphone quantization. Direct bypasses both "
                       "room impulses. Real P4 Opus verified from archived packets. Development diagnosis."))
        p.replace(OUT)


if __name__=="__main__":
    main()
