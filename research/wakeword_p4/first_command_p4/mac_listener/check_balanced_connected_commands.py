"""Apply the established sender balance to connected commands and attribute weak-input ASR losses."""
from collections import Counter, defaultdict
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import check_marked_connected_panel as mixed
import score_marked_connected_uplinks as original

ch, tx, panel = mixed.ch, mixed.tx, mixed.panel
OUT = ch.RESULTS / "4090-c161-balanced-connected-commands.json"
PLAY = ch.RESULTS / "c161-balanced-connected-commands.zip"


def fifo_input(mic, finish):
    emitted, first, count, blocks = finish[:4]
    delayed = np.pad(mic, (1440, blocks * 256 - len(mic)))
    x = delayed[first:emitted]
    assert len(x) == count
    return x


def evaluate(proc, model, audio, expected):
    text = original.asr.transcribe(proc, model, audio)
    words = original.asr.normalized(expected)
    return dict(text=text, complete=original.asr.normalized(text)[-len(words):] == words,
                pcm_sha256=ch.digest(audio), samples=len(audio))


def main():
    report = ch.read(mixed.OUT)
    prior_asr = ch.read(original.OUT)
    originals = [r for r in report["guest"] if r["variant"] == "marked_louder_command"]
    prior = {r["index"]: r for r in prior_asr["records"] if r["variant"] == "marked_louder_command"}
    groups = mixed.study.sources()
    v5, _ = tx.study.panel_wave()
    quiet, rate = sf.read(panel.demo.WAVE, dtype="int16")
    assert rate == 16000 and ch.sha(panel.demo.WAVE) == report["sources"][panel.demo.WAVE.name]
    bed, _, bed_sha = tx.study.grid.room.rooms.read_noise("music")
    builds = dict(bca=panel.middle.WORK / "build")
    assert ch.sha(builds["bca"] / "nyx_mac_listener.bin") == report["firmware"]
    uplink = tx.panel.grid.gap.uplink
    assert uplink.firmware() == report["codec"]
    small, identity = ch.load_reader()
    assert identity == report["small"]
    turbo = tx.diag.reader(report["turbo"])
    decoder = uplink.decoder_api()
    proc, model, qid = original.asr.load()
    assert qid == prior_asr["model"]
    assert ch.sha(mixed.PLAY) == report["speakers_sha256"]
    tmp, pack_tmp = OUT.with_suffix(".tmp"), PLAY.with_suffix(".tmp")
    rows, files, stats, attribution = [], {}, defaultdict(Counter), Counter()
    with TemporaryDirectory() as folder, ZipFile(tmp,"w",ZIP_DEFLATED) as z, ZipFile(mixed.PLAY) as old_pack, \
            ZipFile(pack_tmp,"w",ZIP_DEFLATED) as pack:
        work = Path(folder)
        matcher = tx.single.timed.binding(work)
        native = tx.panel.grid.room.rooms.c161_binding()
        manifest = {r["index"]:r for r in json.loads(old_pack.read("manifest.json"))["records"]}
        for ref in originals:
            s = ref["source"]
            group = groups[s["voice"],s["command"]]
            src, plain = group["Hey Nyx"]
            end = src["end"]
            item = manifest[src["index"]]["tracks"]["marked_louder_command"]
            old_wave, rate = sf.read(BytesIO(old_pack.read(item["file"])),dtype="int16")
            assert rate == 16000 and ch.digest(old_wave) == item["pcm_sha256"]
            balanced, gains = tx.balanced.match_level(v5,plain[end:])
            command = mixed.master(balanced)
            wave = np.r_[old_wave[:end],command]
            scene = mixed.room_scene(s,group,quiet,v5,bed,bed_sha)
            before, _ = scene(old_wave)
            assert ch.digest(before) == ref["mic_sha256"]
            mic, clipped = scene(wave)
            np.testing.assert_array_equal(mic[:end], before[:end])
            no_address = wave.copy()
            no_address[:end] = 0
            command_only, _ = scene(no_address)
            zero = np.zeros_like(mic)
            clock = s["transmit_event"]

            def result(x):
                raw = tx.panel.grid.fit.fixed.ch.routes(matcher,native,x)
                return panel.middle.pair.admit(raw,clock,panel.middle.single.hits(x,clock),zero)

            without = result(command_only)
            expected = result(mic)
            assert expected["opened"] == ref["opened"] and s["start"] <= ref["opened"][0] < end
            k = len(rows)
            case = (ref["index"],s,"bca","both",mic,zero,expected)
            row = tx.panel.capture(work,z,k,case,small,turbo,decoder,builds,tx_pair=clock)
            cap = row["uplink"]
            uart = z.read(f"{k:02}-opus.log").decode().replace("\r\n","\n")
            packets = uplink.packets(uart,k,cap["samples"],0)
            reading = evaluate(proc,model,uplink.decode(decoder,packets),s["command"])
            assert reading["pcm_sha256"] == cap["pcm_sha256"] and len(packets) == cap["packets"]
            assert ch.digest(fifo_input(mic,row["finish"])) == cap["fifo_sha256"]
            old_asr = prior[ref["index"]]
            assert old_asr["pcm_sha256"] == ref["uplink"]["pcm_sha256"]
            old_qwen = old_asr["qwen"]
            row.update(qwen=reading, clipped_samples=clipped, no_address=without, gain=gains,
                       new_level=tx.balanced.dry_levels(command), old_level=tx.balanced.dry_levels(old_wave[end:]),
                       previous_qwen=old_qwen, previous_small=ref["small"], previous_turbo=ref["turbo"])
            if s["external_db"] == -6:
                raw = fifo_input(before,ref["finish"])
                if "fifo_sha256" in ref["uplink"]:
                    assert ch.digest(raw) == ref["uplink"]["fifo_sha256"]
                clean,_ = scene(old_wave,include_music=False)
                views = dict(precodec=raw, precodec_without_music=fifo_input(clean,ref["finish"]))
                row["original_weak"] = {tag:evaluate(proc,model,x,s["command"]) for tag,x in views.items()}
                attribution.update(trials=1, decoded_opus=int(old_qwen),
                    **{tag:int(v["complete"]) for tag,v in row["original_weak"].items()})
            tally = stats[str(s["external_db"])]
            tally.update(trials=1, timely=1, no_address_open=int(bool(without["opened"])), clipped=clipped,
                         saturated_cases=int(clipped>0))
            for tag,a,b in (("qwen",old_qwen,reading["complete"]),("turbo",ref["turbo"],row["turbo"]),
                            ("small",ref["small"],row["small"])):
                tally.update({tag+"_before":int(a),tag:int(b),tag+"_gained":int(b and not a),tag+"_lost":int(a and not b)})
            idx = src["index"]
            if idx not in files:
                files[idx] = mixed.save_speakers(pack,src,{"balanced_louder":wave})
                files[idx].update(panel_at=s["panel_at"],old_file=item["file"],old_pcm_sha256=item["pcm_sha256"])
            rows.append(row)
            if len(rows)%6 == 0:
                print("P4",len(rows),flush=True)
        pack.write(panel.demo.WAVE,"panel-minus6.wav")
        pack.writestr("manifest.json",json.dumps(dict(records=list(files.values()), rate=16000, format="mono PCM16",
            panel_wav="panel-minus6.wav",panel_sha256=ch.sha(panel.demo.WAVE),source_pack=mixed.PLAY.name,
            source_pack_sha256=ch.sha(mixed.PLAY), model=report["model"],firmware=report["firmware"],
            instructions="Play the full external track including leading silence. Start panel-minus6.wav at "
            "panel_at samples after that file starts, using identical position and volume for the old paired file. "
            "Each command begins at command_at. These are dry speaker tracks."),indent=2)+"\n")
    assert len(rows) == 36 and len(files) == 12 and all(x["trials"] == 12 for x in stats.values())
    for partial, final in ((tmp,OUT.with_suffix(".zip")),(pack_tmp,PLAY)):
        with ZipFile(partial) as z:
            assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist()))
        partial.replace(final)
    ch.save(OUT,dict(records=rows,summary=stats,original_weak_attribution=attribution,model=report["model"],
        firmware=report["firmware"],codec=report["codec"],small=identity,turbo=report["turbo"],qwen=qid,
        archive_sha256=ch.sha(OUT.with_suffix(".zip")),speakers_sha256=ch.sha(PLAY),sources={p.name:ch.sha(p)
        for p in (mixed.OUT,mixed.PLAY,original.OUT,panel.demo.WAVE,Path(mixed.__file__),Path(__file__))},
        policy="Transfer the existing v5 command active-RMS balance followed by the existing +6 dB smoothed "
        "peak limiter to entire connected command suffixes. Same preamp, room, noise, external and panel clocks, "
        "including clipped inputs. All 36 firmware captures. Paired actual full uplink ASR without text prompts. "
        "Diagnostic music removal is pre-codec only and retains the original OPEN/FIFO clocks."))


if __name__ == "__main__":
    main()
