"""Replay the eight delivered commands over ongoing panel speech and interrupt it on the P4 OPEN clock."""
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import coded_fixed_transport as fixed
import framed_capture as test
import transition_inputs as inputs
from stop_playback import FADE

listener, probe, bench = test.listener, test.probe, test.bench
BASE = listener.RESULTS / "4090-c161-handoff-barge-in"
PLAYER = listener.RESULTS / (inputs.STEM+"-board-wavs.zip")
COMMANDS = listener.RESULTS / (probe.fixed.STEM+"-commands.zip")


def speaker():
    with ZipFile(PLAYER) as z:
        raw = z.read("panel.wav")
        item = json.loads(z.read("manifest.json"))["files"]["panel.wav"]
    audio, rate = sf.read(io.BytesIO(raw), dtype="int16")
    assert hashlib.sha256(raw).hexdigest()==item["sha256"] and listener.digest(audio)==item["pcm_sha256"] and rate==16000
    return audio


def scenes():
    plan = listener.read(listener.RESULTS/inputs.PLAN)
    pin = listener.read(listener.RESULTS/(probe.fixed.STEM+"-p4.json"))
    assert pin["playback_sha256"]==listener.sha(COMMANDS)
    player = speaker()
    room_map = {}
    for r in plan["records"]:
        if r.get("timing")=="burst" and r["voice"]=="af_heart":
            h, name, digest = inputs.read_room(Path(r["echo_room"]).name[:5])
            assert name==r["echo_room"] and plan["files"][Path(name).name[:5]]==digest
            room_map[r["room"]] = (h, r["echo_gain"], name, digest)
    with ZipFile(COMMANDS) as z:
        manifest = json.loads(z.read("manifest.json"))
        packed = {(r["voice"],r["carrier"]):r for r in manifest["records"]}
        assert manifest["cue_sha256"]==listener.sha(probe.fixed.CUE)
        for _,s,mode,dry,cue,cmd in fixed.materials():
            if mode!="advance":
                continue
            item = packed[s["voice"],s["carrier"]]
            wave = z.read(item["file"])
            source, rate = sf.read(io.BytesIO(wave), dtype="int16")
            assert rate==16000 and hashlib.sha256(wave).hexdigest()==item["wav_sha256"]
            np.testing.assert_array_equal(source, np.concatenate((cue,cmd)))
            h, gain, room, room_hash = room_map[s["room"]]
            ref = np.resize(player,len(dry))
            meta = dict(voice=s["voice"],carrier=s["carrier"],cue_at=s["start"],command_at=s["end"],
                        room=s["room"],echo_room=room,echo_gain=gain,rir_sha256=room_hash,
                        before_echo_mic_sha256=listener.digest(dry),external_wav_sha256=item["wav_sha256"])
            yield meta, dry, ref, h


def render(mic, ref, h, gain, noise):
    echo = gain*fftconvolve(ref.astype(float),h)[:len(mic)]
    whole = np.rint(mic.astype(float)+echo)
    mixed = whole.clip(-32768,32767).astype("<i2")
    return mixed, bench.channel(ref,"noisy_line",noise), int(np.count_nonzero(whole!=mixed))


def main():
    reader, identity = listener.load_reader()
    decoder = probe.uplink.decoder_api()
    archive = BASE.with_suffix(".zip")
    rows = []
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"),"w",ZIP_DEFLATED) as logs:
        work = Path(tmp)
        full = probe.fixed.ch.binding(work/"full.so",coded="v5")
        pair = probe.fixed.ch.binding(work/"pair.so",coded="aligned")
        for j,(meta,baseline,player,h) in enumerate(scenes()):
            noise = np.random.default_rng(18221).standard_normal(len(player))
            row = dict(source=meta)
            for arm in ("continue","interrupt"):
                ref = player.copy()
                if arm=="interrupt":
                    if not row["continue"]["opened"]:
                        break
                    stop = row["continue"]["opened"][0]
                    ref[stop:stop+FADE] = np.rint(ref[stop:stop+FADE]*np.arange(FADE-1,-1,-1)/FADE)
                    ref[stop+FADE:] = 0
                mic, adc, clips = render(baseline,ref,h,meta["echo_gain"],noise)
                if arm=="continue":
                    original_mic, original_adc = mic, adc
                else:
                    np.testing.assert_array_equal(mic[:stop],original_mic[:stop])
                    np.testing.assert_array_equal(adc[:stop],original_adc[:stop])
                i = 2*j+int(arm=="interrupt")
                result = probe.score(work,meta,mic,adc,full,i,build=test.BUILD,prefix=pair,qualify=probe.qualify,
                                     wait_for_third=True,frame_activity=True,qualify_capture_frame=True)
                if arm=="interrupt":
                    assert result["opened"]==row["continue"]["opened"]
                result.update(clipped_samples=clips, speaker_sha256=listener.digest(ref))
                if result["opened"]:
                    result["uplink"] = bench.fifo(work,logs,i,mic,adc,result,reader,decoder,True,True)
                    t = result["opened"][0]
                    result["speaker_20ms_before_open_rms"] = float(np.sqrt(np.mean(ref[max(0,t-320):t].astype(float)**2)))
                logs.writestr(f"{i:02}-listener.log","\n".join(result["uart_events"])+"\n")
                row[arm] = result
                print(i,arm,result["opened"],flush=True)
            rows.append(row)
    assert len(rows)==16
    archive.with_suffix(".tmp").replace(archive)
    listener.save(BASE.with_suffix(".json"),dict(records=rows, cue_sha256=listener.sha(probe.fixed.CUE),
        command_pack_sha256=listener.sha(COMMANDS), speaker_pack_sha256=listener.sha(PLAYER),
        original_scenarios_sha256=listener.sha(listener.RESULTS/(probe.fixed.STEM+"-p4.json")),
        echo_setup_sha256=listener.sha(listener.RESULTS/inputs.PLAN),archive_sha256=listener.sha(archive),
        firmware_sha256=listener.sha(test.BUILD/"nyx_mac_listener.bin"),code_sha256=listener.sha(Path(__file__)),
        codec_sha256=listener.sha(probe.uplink.CODEC/"nyx_first_command.bin"),qemu_sha256=listener.sha(listener.QEMU),
        model_sha256=listener.C161,asr=identity,policy="External file verified against every delivered source WAV. "
        "Panel speech plays from input sample zero using the two archived af_heart echo gains and distinct room responses. "
        "Input reference uses the earlier inverted bandlimited 0.25 gain and Gaussian noise (sigma 24 PCM16). "
        "In the intervention fade over 160 samples from actual OPEN, then keep player silent and retain room decay. "
        "Microphone and ADC reference prefixes must be identical. No open means no intervention."))


if __name__ == "__main__":
    main()
