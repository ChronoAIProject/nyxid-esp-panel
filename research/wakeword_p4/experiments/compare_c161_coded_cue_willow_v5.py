"""Run official Willow DEFAULT on the same v5, Willow and silence immediate-command streams."""
import sys
from pathlib import Path
from unittest.mock import patch

# The case subprocess only needs the original stdlib QEMU harness.
ROOT=Path("research/wakeword_p4")
STEM="4090-c161-coded-cue-willow-v5"
PLAN=ROOT/"results"/f"{STEM}-plan.json"
OUT=Path(".local/wakeword/c161-coded-cue-willow-v5")


def make(official):
    import hashlib
    import soundfile as sf
    import check_c161_coded_cue_v4 as playback
    from fit_c161_history_gain_v1 import RESULTS, read, save, sha
    from generate_c161_coded_cue_v5 import WAV
    from select_c161_synthesized_cue_v2 import load_wav
    ref=RESULTS/"4090-c161-coded-cue-v5-commands.json"
    spec=read(RESULTS/"4090-c161-coded-cue-v5-source.json")
    cue=load_wav(dict(path=WAV,sha256=spec["wav_sha256"]))
    old=read(RESULTS/"4090-c161-v3-command-willow-plan-v1.json")
    files={p:sha(Path(p)) for p in old["runtime"].values()}
    assert all(old["files"][p]==h for p,h in files.items())
    for p in (Path(__file__).resolve(),Path(official.runner.__file__).resolve(),Path(official.runner.official.__file__).resolve(),official.runner.official.SOURCE,ref,WAV):
        files[str(p)]=sha(p)
    rows=[]
    OUT.mkdir(exist_ok=True)
    with patch.object(playback,"TARGET","v5"):
        for (meta,pcm),prior in zip(playback.command_streams(cue),read(ref)["records"],strict=True):
            assert hashlib.sha256(pcm.tobytes()).hexdigest()==prior["pcm_sha256"]
            if meta["role"]=="v3":
                continue
            path=OUT/f"{len(rows):03d}.wav"
            sf.write(path,pcm,16000,subtype="PCM_16")
            files[str(path)]=sha(path)
            rows.append(dict(**meta,path=str(path),samples=len(pcm),target_id="nyx" if meta["role"]=="v5" else meta["role"],
                             carrier="connected command",pcm_sha256=prior["pcm_sha256"],sha256=files[str(path)]))
    assert len(rows)==108
    save(PLAN,dict(files=files,runtime=old["runtime"],records=rows,threshold_requests=["DEFAULT"]))


if __name__=="__main__":
    action=sys.argv.pop(1)
    if action=="case":
        sys.path.insert(0,str((ROOT/"qemu_hey_reference_v2").resolve()))
        import run_willow_mac_matched_prime_v1 as runner
        runner.PLAN=PLAN
        runner.__file__=str(Path(__file__).resolve())
        runner.case()
    else:
        import compare_c161_cue_willow_v1 as official
        with patch.multiple(official,PLAN=PLAN,STEM=STEM,CASES=OUT/"runs",__file__=__file__):
            if action=="make":
                make(official)
            else:
                {"run":official.execute,"summarize":official.summarize}[action]()
