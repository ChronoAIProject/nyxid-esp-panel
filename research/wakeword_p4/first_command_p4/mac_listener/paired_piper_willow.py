"""Compare end-aligned multi-syllable calls on the 18 frozen Piper command recordings."""
from pathlib import Path
import sys

ROOT = Path("research/wakeword_p4")
STEM = "4090-c161-piper-matched-willow"
PLAN = ROOT/"results"/(STEM+"-plan.json")
WORK = Path(".local/wakeword/c161-piper-matched-willow")
RUNNER = Path(__file__).resolve().relative_to(Path.cwd())


def prepare():
    import io
    import json
    from zipfile import ZipFile

    import numpy as np
    import soundfile as sf
    import fixed_cue_dialogue_p4 as fixed
    import voice_cues_willow as voices
    from check_c161_command_playback_level import levels
    from coded_close_join import active
    from fit_c161_history_gain_v1 import RESULTS, read, save, sha

    pack = RESULTS/(fixed.STEM+"-playback.zip")
    reference = RESULTS/(fixed.STEM+".json")
    prior = read(reference)
    v5,rate = sf.read(fixed.host.fixed.CUE,dtype="int16")
    assert rate==16000 and prior["original_cue_sha256"] == sha(fixed.host.fixed.CUE)
    runtime = read(voices.PLAN)
    files = {p:runtime["files"][p] for p in runtime["runtime"].values()}
    for p in (RUNNER, pack, reference, fixed.host.fixed.CUE, voices.DRY, voices.PLAN,
              voices.official.official.SOURCE, Path(voices.official.official.__file__)):
        files[str(p)] = sha(p)
    assert files[str(voices.DRY)] == runtime["files"][str(voices.DRY)]
    from voice_cues import DRY as nyx_dry
    files[str(nyx_dry)] = sha(nyx_dry)
    bank = voices.source_bank()
    digest = fixed.host.base.poll.digest
    clips = {"fixed":v5,"silence":np.zeros_like(v5)}
    info, unscaled = {}, {}
    with ZipFile(voices.DRY) as z:
        for v in voices.VOICES:
            willow,rate = sf.read(io.BytesIO(z.read(v+"-Willow.wav")),dtype="int16")
            assert rate==16000
            for role, source in (("willow",willow),("ordinary",bank[v,"Hey Nyx."])):
                lo, hi = active(source)
                original = source[lo:hi]
                key = v+"-"+role
                unscaled[key] = original
                info[key] = dict(source_sha256=digest(source),source_active=[lo,hi])
    unscaled["level-fixed"] = v5
    info["level-fixed"] = dict(source_sha256=digest(v5),source_active=[0,len(v5)])
    # Highest shared active RMS up to the shipped cue level with 1 dB of peak headroom.
    target_db = min(levels(v5)["active_rms_dbfs"], *(levels(x)["active_rms_dbfs"]-1-levels(x)["peak_dbfs"]
                                                      for x in unscaled.values()))
    for key,x in unscaled.items():
        gain = 10**((target_db-levels(x)["active_rms_dbfs"])/20)
        q = np.rint(x*gain)
        assert abs(q).max()<=32767 and len(q)<=len(v5)
        clips[key] = q.astype("<i2")
        info[key].update(gain=gain,pcm_sha256=digest(clips[key]),levels=levels(clips[key]))
    records = []
    WORK.mkdir(exist_ok=True,parents=True)
    with ZipFile(pack) as z:
        manifest = json.loads(z.read("manifest.json"))
        assert manifest["report_sha256"]==sha(reference)
        for group,(case,m) in enumerate(zip(prior["records"],manifest["records"],strict=True)):
            base,rate = sf.read(io.BytesIO(z.read(m["file"])),dtype="int16")
            assert rate==16000 and digest(base)==m["pcm_sha256"]==case["mic_sha256"]
            a,b = m["cue_at"],m["command_at"]
            np.testing.assert_array_equal(base[a:b],v5)
            for key,clip in clips.items():
                mic = base.copy()
                mic[a:b] = 0
                mic[b-len(clip):b] = clip
                assert digest(base[:a])==digest(mic[:a]) and digest(base[b:])==digest(mic[b:])
                if key=="fixed":
                    np.testing.assert_array_equal(mic,base)
                path = WORK / f"{len(records):03}.wav"
                sf.write(path,mic,16000,subtype="PCM_16")
                files[str(path)] = sha(path)
                records.append(dict(index=len(records),pair=group,target_id="level-fixed" if key=="level-fixed" else key.split("-")[-1],cue=key,
                                    voice=m["voice"],carrier=m["command"],start=b-len(clip),end=b,
                                    samples=len(mic),path=str(path),sha256=files[str(path)],pcm_sha256=digest(mic),
                                    cue_sha256=digest(clip),suffix_sha256=digest(mic[b:])))
    result = dict(files=files,records=records,cues=info,runtime=runtime["runtime"],threshold_requests=["DEFAULT"],
                  matched_active_rms_dbfs=target_db,fixed_active_rms_dbfs=levels(v5)["active_rms_dbfs"],
                  policy="All four original Kokoro voices, a fixed Nyx WAV and a zero cue at each of the 18 "
                         "original command clocks. Place active 20 ms frames ending at the command, using the "
                         "original -30 dB power rule and matching all cue levels with 1 dB of peak headroom. Retain every sample "
                         "before the common 1.10 s slot and from the command onwards. Same dry direct channel.")
    if PLAN.exists():
        assert read(PLAN)==result
    else:
        save(PLAN,result)


if __name__ == "__main__":
    action = sys.argv.pop(1)
    if action=="prepare":
        prepare()
    elif action=="case":
        sys.path.insert(0,str((ROOT/"qemu_hey_reference_v2").resolve()))
        import run_willow_mac_matched_prime_v1 as official
        official.PLAN = PLAN
        official.__file__ = str(RUNNER)
        official.case()
    elif action=="willow":
        import run_c161_willow_matched_scenes_v1 as official
        official.PLAN,official.OUT,official.__file__ = PLAN,WORK/"runs",str(RUNNER)
        official.willow()
    else:
        raise SystemExit("prepare | willow | case")
