"""Replay fixed proposals against the adapted keyword states, including all archived backgrounds."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import soundfile as sf
from pocketsphinx import Decoder

from check_c161_compact_lexical import sources as speech
from check_c161_event_transcripts_v1 import choose
from check_c161_pocketsphinx import confirmations
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_speex_native_proposals import BASE, NEW, POST, LAG, archive_pcm, render_half
from check_c161_speex_residual import speech_reading
from check_c161_v5_command_ns import digest
from compare_c161_native_lexical import SPEECH, BG, OUT as NATIVE
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from train_c161_q8_keyword_map import STEM, ROOT, FILES, WEIGHTS


def recheck(decoder, cmn, pcm, base, native, expanded):
    checked = confirmations(decoder,cmn,pcm,sorted(set(native+expanded)))
    events = {}
    for name, allowed in (("native", native), ("dual", native+expanded)):
        events[name] = choose(base,[c["confirmed"] for c in checked if c["proposal"] in allowed and c["confirmed"] is not None])
    return dict(checked=checked,events=events)


def ordinary(decoder, cmn):
    originals = {(r["cohort"],r["key"]):r for p in (SPEECH,BG) for r in read(p)["records"]}
    reference = {(r["cohort"],r["key"]):r for r in read(NATIVE)["records"]}
    rows = []
    def replay(group,key,pcm):
        row = originals.pop((group,key))
        old = reference.pop((group,key))
        assert row["pcm_sha256"] == old["pcm_sha256"] and (pcm is None or digest(pcm) == row["pcm_sha256"])
        props = row["proposals"]
        candidate = recheck(decoder, cmn, pcm, row["baseline"], props["midlevel"],props["expanded"])
        rows.append(dict(cohort=group,key=key,pcm_sha256=row["pcm_sha256"],base_events=old["events"],**candidate))
    for group,key,pcm in speech():
        replay(group,key,pcm)
    for group in ("melo","primed","jenny"):
        for r in read(RESULTS/f"4090-c161-history-gain-{group}-v1.json")["records"]:
            if r["phase"] != "background":
                continue
            key = f"{group}_{r['index']}"
            pcm = None
            if originals[group,key]["checked"]:
                src = r["source"]
                pcm,rate = sf.read(src.get("wav_path") or src["path"],dtype="int16")
                assert rate == 16000
            replay(group,key,pcm)
    assert not originals and not reference
    return rows


def playback_inputs():
    wet_path = RESULTS/f"{POST}.json"
    mixed_path = RESULTS/f"{POST}-mix.json"
    d,w,h,new = (read(p) for p in (BASE,wet_path,mixed_path,NEW))
    plan = read(NEW.with_name(NEW.stem+"-plan.json"))
    archives = [p.with_name(p.stem+"-uart.zip") for p in (BASE,wet_path,NEW)]
    assert all(sha(p) == r["uart_archive_sha256"] for p,r in zip(archives,(d,w,new),strict=True))
    with TemporaryDirectory() as tmp, ZipFile(archives[0]) as a, ZipFile(archives[1]) as b, ZipFile(archives[2]) as c:
        log = Path(tmp)/"uart.log"
        for dry, wet, mixed in zip(d["records"],w["records"],h["records"],strict=True):
            key = dry["key"]
            meta = dry["source"]
            role = meta["role"]
            ext = role in ("external_only","external_over_reply","simultaneous_mentions")
            end = meta["external_end"] if ext else meta["self_end"]
            want = meta["external_command"] if ext else "turn on the kitchen lights." if role=="same_mic_external" else None
            linear = archive_pcm(a,log,key,dry)
            post = archive_pcm(b,log,key,wet)
            half = render_half(linear,post)
            assert digest(half) == mixed["mix_sha256"]
            yield key,role,meta["start"],end,want,[(linear,dry["detections"]),(post,wet["detections"]),(half,mixed["mix_detector"])]
        for row,meta in zip(new["records"],plan["records"],strict=True):
            key = row["key"]
            assert key == meta["key"]
            linear,post = [archive_pcm(c,log,key+"-"+m,row["dsp"][m]) for m in ("linear","post")]
            half = render_half(linear,post)
            assert digest(half) == row["half_sha256"]
            want = meta["text"].split(",",1)[1] if meta["label"] == "nyx" else None
            inputs = zip((linear,post,half),[row["detections"][m] for m in ("linear","post","half")],strict=True)
            yield key, meta["label"],meta["start"],meta["end"],want,inputs


def main(root=ROOT, stem=STEM, weights=WEIGHTS):
    fit = RESULTS/f"{stem}-fit.json"
    assert sha(weights) == sha(root/"am/sendump") == read(fit)["adapted_sendump_sha256"]
    decoder = Decoder(hmm=str(root/"am"),dict=str(FILES/"two.dict"),keyphrase="hey nicks",
                      kws_threshold=1e-15,loglevel="ERROR",lm=None)
    cmn = decoder.get_cmn()
    rows = ordinary(decoder,cmn)
    reader,identity = load_reader()
    output = []
    for key,role,start,end,want,inputs in playback_inputs():
        paths = {}
        for mode,(pcm,prior) in zip(("linear","post","half"),inputs,strict=True):
            ev = prior["events"]
            result = recheck(decoder,cmn,pcm,ev["c161"],ev["original"],ev["expanded"])
            result["pcm_sha256"] = digest(pcm)
            lag = 0 if mode=="linear" else LAG
            captures = {}
            for name,events in result["events"].items():
                hits = [r for r in events if start+lag<=r["sample"]<end+lag+160]
                capture = dict(hits=hits)
                if hits and want:
                    first = max(0,(hits[0]["sample"]+159)//160*160-32000)
                    if first not in captures:
                        captures[first] = speech_reading(reader,pcm,first,end+lag+160,want)
                    capture["command"] = captures[first]
                result[name] = capture
            paths[mode] = result
        output.append(dict(key=key,role=role,paths=paths))
    refs = [fit,SPEECH,BG,NATIVE,BASE,NEW,RESULTS/f"{POST}.json",RESULTS/f"{POST}-mix.json",
            NEW.with_name(NEW.stem+"-plan.json")]
    save(RESULTS/f"{stem}-evaluation.json",dict(sources={p.name:sha(p) for p in refs},
                                               asr=identity,ordinary=rows,playback=output))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-contrast", action="store_true")
    if parser.parse_args().name_contrast:
        import train_c161_q8_name_contrast as variant
        main(variant.ROOT, variant.STEM, variant.WEIGHTS)
    else:
        main()
