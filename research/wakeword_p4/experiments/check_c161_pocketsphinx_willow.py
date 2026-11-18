"""Generate fresh connected dialogue and compare original C161, official Willow and English lexical search."""
from pathlib import Path
import sys
from unittest.mock import patch
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path("research/wakeword_p4")
RESULTS = ROOT/"results"
STEM = "4090-c161-pocketsphinx-willow"
PLAN = RESULTS/f"{STEM}-plan.json"
WORK = Path(".local/wakeword/c161-pocket-reference/dialogues")
VOICES = ("af_nova", "af_jessica", "am_santa", "bm_lewis")
CARRIERS = ("set a reminder for my bus.", "raise the screen brightness.")
NAMES = (("nyx","Hey Nicks"),("willow","Hey Willow"),("nick","Hey Nick"),("max","Hey Max"))


def generate(official):
    from kokoro_onnx import Kokoro
    import numpy as np
    from scipy.signal import resample_poly
    import soundfile as sf
    from check_c161_wake_phone_duration_v1 import session
    from check_c161_v5_command_ns import digest
    from fit_c161_history_gain_v1 import read, save, sha
    WORK.mkdir(exist_ok=True,parents=True)
    inventory = read(RESULTS/"4090-hey-common-carrier-kokoro-generation-execution-v1.json")
    assert set(VOICES) <= {r["request"]["conditioning_id"] for r in inventory["requests"] if r["request"]["split"]=="training"}
    model = Path(".local/wakeword/hey-common-carrier-kokoro-seeded-v1.onnx")
    styles = Path(".local/wakeword/kokoro-models/voices-v1.0.bin")
    prev = read(RESULTS/"4090-c161-v5-household-rooms-plan.json")
    files = {p:sha(Path(p)) for p in prev["runtime"].values()}
    assert all(prev["files"][p] == h for p,h in files.items())
    for p in (model, styles):
        assert sha(p) == inventory["files"][str(p)]
        files[str(p)] = sha(p)
    for p in (Path(__file__).resolve(),Path(official.runner.__file__).resolve(),Path(official.runner.official.__file__).resolve(),official.runner.official.SOURCE):
        files[str(p)] = sha(p)
    engine = session(str(model))
    speaker = Kokoro.from_session(engine,str(styles))
    script = [(role, f"{address}, {cmd}") for cmd in CARRIERS for role,address in NAMES]
    script.extend([("daily","The next bus leaves in five minutes."),("daily","Please mix the two colors before lunch.")])
    rng = np.random.default_rng(81091)
    order = rng.permutation(len(script))
    gaps = rng.choice([5600,8000,12000], len(script))
    archive = RESULTS/f"{STEM}-wav.zip"
    rows = []
    with ZipFile(archive,"x",ZIP_DEFLATED) as z:
        for i,voice in enumerate(VOICES):
            chunks = [np.zeros(16000,np.int16)]
            at = len(chunks[0])
            timeline = []
            for req,gap in zip(order,gaps,strict=True):
                role,text = script[req]
                phones = speaker.tokenizer.phonemize(text,"en-gb" if voice.startswith("b") else "en-us")
                ids = speaker.tokenizer.tokenize(phones)
                assert len(ids)==len(phones)
                signal,durations = engine.run(None, dict(input_ids=np.array([[0,*ids,0]],np.int64),
                    style=np.asarray(speaker.get_voice_style(voice)[len(ids)],np.float32),speed=np.ones(1,np.float32)))
                x = resample_poly(signal.ravel(),2,3)
                dur = durations.ravel()
                assert len(x)==sum(dur)*400 and np.isfinite(x).all() and len(dur)==len(phones)+2
                pcm = np.rint(x*32768).clip(-32768,32767).astype("<i2")
                first = None
                if role!="daily":
                    j=phones.index(",")+1
                    while phones[j] in " ˈˌ":
                        j+=1
                    first=at+int(sum(dur[:j+1]))*400
                timeline.append(dict(role=role,text=text,start=at,end=at+len(pcm),first_phone=first,
                                     phones=phones,durations=dur.tolist(),pcm_sha256=digest(pcm)))
                chunks.extend((pcm,np.zeros(gap,np.int16)))
                at+=len(pcm)+int(gap)
            wave = np.concatenate(chunks)
            dest=WORK/f"{i:02}.wav"
            sf.write(dest,wave,16000,subtype="PCM_16")
            files[str(dest)] = sha(dest)
            z.write(dest,dest.name)
            rows.append(dict(voice=voice,carrier="continuous dialogue",target_id="paired",path=str(dest),
                             sha256=files[str(dest)],samples=len(wave),pcm_sha256=digest(wave),timeline=timeline))
    save(PLAN, dict(records=rows,files=files,runtime=prev["runtime"],threshold_requests=["DEFAULT"],
                    requests=script,order=order.tolist(),gaps=gaps.tolist(),archive_sha256=sha(archive),
                    generation_provider=engine.get_providers()[0]))


def lexical():
    from calibrate_c161_pocketsphinx import OUT
    from check_c161_pocketsphinx import setup, search, confirmations
    from check_c161_room_word import detectors
    from check_c161_event_transcripts_v1 import choose
    from audit_c161_history_paths_v1 import decode,times
    from replay_c161_word_window_v1 import score
    from fit_c161_history_gain_v1 import read,save,sha
    from select_c161_synthesized_cue_v2 import load_wav
    from train_c161_mined_room_word import PREFIX
    source = RESULTS/f"{STEM}-v1.json"
    rows=[]
    it,inp,out = list(detectors(f"{PREFIX}-hard"))[1]
    for r in read(source)["records"]:
        pcm=load_wav(r)
        codes,clock=score(it,pcm,inp,out)
        proposals=times(decode(codes,clock))
        row=dict(voice=r["voice"],pcm_sha256=r["pcm_sha256"],proposals=proposals,search={})
        for mode,threshold in (("default", 1e-30),("calibrated",10.**(-read(OUT)["selected_exponent"]))):
            d,cmn,_=setup(threshold)
            direct=search(d,cmn,pcm,0)
            tested=confirmations(d,cmn,pcm,proposals)
            row["search"][mode]=dict(matches=direct,direct=choose([],sorted({e["returned"] for e in direct})),
                checked=tested,combined=choose(r["original_events"],[t["confirmed"] for t in tested if t["confirmed"] is not None]))
        rows.append(row)
    save(RESULTS/f"{STEM}-lexical.json",dict(source_sha256=sha(source),calibration_sha256=sha(OUT),records=rows))


if __name__=="__main__":
    action=sys.argv.pop(1)
    if action=="case":
        sys.path.insert(0,str((ROOT/"qemu_hey_reference_v2").resolve()))
        import run_willow_mac_matched_prime_v1 as runner
        runner.PLAN=PLAN
        runner.__file__=str(Path(__file__).resolve())
        runner.case()
    elif action=="lexical":
        lexical()
    else:
        import compare_c161_cue_willow_v1 as official
        with patch.multiple(official,PLAN=PLAN,STEM=STEM,CASES=WORK/"runs",__file__=__file__):
            if action=="generate":
                generate(official)
            else:
                {"willow":official.execute,"score":official.summarize}[action]()
