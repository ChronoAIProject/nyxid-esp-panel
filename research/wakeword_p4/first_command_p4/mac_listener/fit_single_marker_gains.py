"""Allocate a fixed PCM energy increment to the three existing markers on training rooms/music."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
import torch

import check_v5_playback_gain as prior
from replay_c161_pcm_cues_v1 import detect
from select_repeat_level import letters
from train_c161_music_proposals import MUSIC

grid, ch = prior.grid, prior.ch
STEM = "4090-c161-single-marker-gains"
FIT = ch.RESULTS / (STEM + "-fit.json")
WAV = ch.RESULTS / "hye-nyx-v5-marker-gains.wav"
DEV = ch.RESULTS / "4090-c161-music-development-sources-v1.json"
SEED = 191153


def candidates():
    meta, original, speech, codes = grid.fit.original()
    boosted, sr = sf.read(prior.WAV, dtype="int16")
    assert sr == 16000 and ch.sha(prior.WAV) == ch.read(prior.OUT)["wav_sha256"]
    norm = np.linalg.norm(boosted.astype(float))
    added = norm**2 - np.linalg.norm(original.astype(float))**2
    tracks = dict(v5=original, boosted=boosted)
    weights = {}
    for first in range(1,5):
        for second in range(1,6-first):
            parts = (first,second,6-first-second)
            cue = original.astype(float).copy()
            gains = []
            for pos, code, share in zip(meta["code_positions"], codes.astype(float), parts, strict=True):
                where = slice(pos,pos+len(code))
                # Increase this window's *actual* total energy, including speech/code cross terms.
                before = cue[where].copy()
                energy = code @ code
                cross = before @ code
                extra = (np.sqrt(cross**2 + energy*added*share/6) - cross) / energy
                cue[where] = before + extra*code
                gains.append(float(1+extra))
            q = np.rint(cue)
            name = "".join(map(str,parts))
            weights[name] = dict(parts=parts, marker_gains=gains, peak=int(np.abs(q).max()))
            if np.abs(q).max() > 32767:
                continue
            tracks[name] = q.astype("<i2")
            delta = float(20*np.log10(np.linalg.norm(q)/norm))
            assert abs(delta) < .0002
            weights[name]["rms_db_vs_boosted"] = delta
    return tracks, weights


def train():
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    waves, weights = candidates()
    reader, asr = ch.load_reader()
    readings = prior.p4.reading(waves,reader)
    rng = np.random.default_rng(SEED)
    fit = ch.read(grid.fit.FIT)
    rooms = fit["training_rooms"]
    developer_tracks = ch.read(DEV)["records"]
    excluded = {r["sha256"] for r in developer_tracks}
    inventory = ch.read(MUSIC/"ambient.json")
    prep = ch.read(ch.RESULTS/"4090-musan-music-preparation.json")
    assert prep["all_training_only"] and ch.sha(MUSIC/"ambient.json") == prep["index_sha256"]
    bank = [r for r in inventory if r["split"]=="training" and r["source_sha256"] not in excluded
            and r["source_path"] not in {s["source_path"] for s in developer_tracks}]
    chosen = rng.choice(len(bank),24,replace=False)
    music = []
    for i in chosen:
        r = bank[i]
        p = MUSIC/r["audio_path"]
        assert ch.sha(p) == r["source_sha256"]
        with sf.SoundFile(p) as f:
            assert f.samplerate == 16000 and f.channels == 1
            start = int(rng.integers(len(f)-64000+1))
            f.seek(start)
            music.append((r,start,f.read(64000)))
    names = list(waves)
    clean = torch.from_numpy(np.stack(list(waves.values())).astype(float)).cuda()
    observations = []
    trials = {n:[] for n in waves}
    with TemporaryDirectory() as tmp:
        detector = ch.binding(Path(tmp)/"v5.so",coded="v5")
        for i,r in enumerate(rooms):
            h,path,checksum = ch.read_room(Path(r["path"]).name[:5])
            assert (path,checksum) == (r["path"],r["sha256"])
            n = 1 << (clean.shape[1]+len(h)-2).bit_length()
            wet = torch.fft.irfft(torch.fft.rfft(clean,n)*torch.fft.rfft(torch.from_numpy(h).cuda(),n),n)[:,:clean.shape[1]].cpu().numpy()
            for j in range(2):
                record,offset,noise = music[(i*7+j)%len(music)]
                local = int(rng.integers(64000-32000))
                segment = noise[local:local+32000]
                onset = int(rng.integers(4096,8192))
                end = onset+clean.shape[1]
                noise_power = np.linalg.norm(segment[onset:end])
                assert noise_power > 1e-5
                for snr in (-12,-6,0):
                    mixed = np.broadcast_to(segment*(np.linalg.norm(wet[0])/noise_power*10**(-snr/20)),
                                            (len(names),len(segment))).copy()
                    mixed[:,onset:end] += wet
                    scale = min(1., 28000 / np.abs(mixed).max())
                    trial = np.rint(mixed*scale).astype("<i2")
                    observed = {}
                    for name,x in zip(names,trial,strict=True):
                        hits = [v["t"] for v in detect(detector,[x])]
                        before = bool(hits and onset <= ((hits[0]+255)//256)*256 < end)
                        trials[name].append(before)
                        if hits:
                            observed[name] = hits
                    observations.append(dict(room=path,track=record["source_path"],offset=offset+local,onset=onset,
                                             end=end,snr_db=snr,scale=scale,events=observed))
    eligible = [n for n in weights if n in readings and all(letters(t)=="HEYNIX" for t in readings[n].values())]
    eligible.sort(key=lambda n:(-sum(trials[n]),sum((v-2)**2 for v in weights[n]["parts"]),n))
    best = eligible[0] if eligible else "boosted"
    sf.write(WAV,waves[best],16000,subtype="PCM_16")
    ch.save(FIT,dict(selected=best,trials=len(observations),timely={n:sum(v) for n,v in trials.items()},
                     readings=readings,weights=weights,eligible=eligible,records=observations,asr=asr,
                     wav_sha256=ch.sha(WAV),pcm_sha256=ch.digest(waves[best]),gpu=torch.cuda.get_device_name(),
                     seed=SEED,music=[dict(path=r["source_path"],sha256=r["source_sha256"],start=o) for r,o,_ in music],
                     sources={p.name:ch.sha(p) for p in (grid.fit.FIT,DEV,prior.OUT,prior.WAV,grid.fit.fixed.CUE,Path(__file__))},
                     policy="Keep original speech, marker positions and templates. Allocate the additional PCM "
                     "energy of the +0.694 dB control in sixths among three marker windows. Exclude clipping "
                     "and require both dry readers to reproduce HEYNIX. Rank by timely native fixed hits on "
                     "48 training RIRs, 24 separate training music tracks, 2 excerpts per RIR and three SNRs. "
                     "Tie break nearest equal allocation, then lexicographic. No receiver adaptation."))


if __name__ == "__main__":
    train()
