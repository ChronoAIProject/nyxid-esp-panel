"""Run the published English PocketSphinx keyword search on ordinary speech and on causal candidate windows."""
import argparse
from pathlib import Path

from pocketsphinx import Config, Decoder, get_model_path
import soundfile as sf

from check_c161_event_transcripts_v1 import choose, event_inputs
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from select_c161_synthesized_cue_v2 import load_wav
from train_c161_mined_room_word import PREFIX

STEM = "4090-c161-pocketsphinx"
CANDIDATE = f"{PREFIX}-hard"
PHRASE = "hey nicks"
PAST, WAIT, BLOCK = 24000, 4800, 480
WHEEL = "a3541ad8b6ce00230bd96fe520d48c3a252e031531bc8337dfccf7553cbca998"


def setup():
    cache = Path(".local/wakeword/c161-pocket-reference")
    wheel, = cache.glob("pocketsphinx-5.0.4-*.whl")
    assert sha(wheel) == WHEEL
    config = Config(keyphrase=PHRASE, loglevel="ERROR")
    assert config["lm"] is None and config["kws_threshold"] == 1e-30 and config["frate"] == 100
    model = Path(get_model_path())
    files = {p.relative_to(model).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p))
             for p in (*Path(config["hmm"]).iterdir(), Path(config["dict"])) if p.is_file()}
    native = Decoder(config)
    assert native.lookup_word("hey") == "HH EY" and native.lookup_word("nicks") == "N IH K S"
    # The decoder has adaptive utterance history. Reinitialize its features per independent stream.
    return native, native.get_cmn(), dict(wheel_sha256=WHEEL, package="pocketsphinx 5.0.4", phrase=PHRASE,
        files=files, config={k:config[k] for k in ("kws_threshold","kws_delay","kws_plp","samprate","cmn","frate")},
        past=PAST, future_limit=WAIT, block=BLOCK)


def search(decoder, cmn, pcm, origin):
    decoder.reinit_feat()
    decoder.set_cmn(cmn)
    decoder.start_utt()
    reports, seen = [], set()
    for offset in range(0, len(pcm), BLOCK):
        chunk = pcm[offset:offset+BLOCK]
        decoder.process_raw(chunk.tobytes(), False, False)
        for seg in decoder.seg() or ():
            identity = (seg.word.strip(), seg.start_frame, seg.end_frame)
            if identity in seen:
                continue
            seen.add(identity)
            assert identity[0] == PHRASE
            reports.append(dict(returned=origin+offset+len(chunk), start=origin+160*seg.start_frame,
                                end=origin+160*(seg.end_frame+1), prob=seg.prob))
    decoder.end_utt()
    return reports


def confirmations(decoder, cmn, pcm, proposals):
    out = []
    for t in proposals:
        at, end = max(0, t-PAST), min(len(pcm), t+WAIT)
        matches = search(decoder, cmn, pcm[at:end], at)
        available = [max(t, r["returned"]) for r in matches if t-7200 <= r["end"] <= end]
        out.append(dict(proposal=t, start=at, end=end, matches=matches, confirmed=min(available, default=None)))
    return out


def run():
    decoder, cmn, spec = setup()
    prior = RESULTS/f"{CANDIDATE}-replay-v1.json"
    sources = {(r["cohort"],r["record_key"]):r for r in read(prior)["records"]}
    records = []
    for group in ("context","forward","daily","joe","mac","bot"):
        for meta, key, _, pcm in event_inputs(group):
            old = sources[group,key]
            assert digest(pcm) == old["meta"]["pcm_sha256"]
            direct = search(decoder, cmn, pcm, 0)
            independent = choose([], sorted(set(r["returned"] for r in direct)))
            checked = confirmations(decoder, cmn, pcm, old["word"])
            merged = choose(old["base"], [r["confirmed"] for r in checked if r["confirmed"] is not None])
            records.append(dict(cohort=group, key=key, pcm_sha256=digest(pcm), direct=direct, direct_events=independent,
                                proposals=checked, merged=merged))
        print(group, len(records), flush=True)
    rooms = RESULTS/"4090-c161-v5-household-rooms-v1.json"
    for r in read(rooms)["records"]:
        pcm = load_wav(r)
        matches = search(decoder, cmn, pcm, 0)
        records.append(dict(cohort="household", key=r["pcm_sha256"], direct=matches,
                            direct_events=choose([], sorted(set(x["returned"] for x in matches)))))
    save(RESULTS/f"{STEM}-v1.json", dict(implementation=spec, reset_cmn=cmn, inputs_sha256=sha(prior),
                                        household_sha256=sha(rooms), records=records))


def background():
    decoder, cmn, spec = setup()
    prior = RESULTS/f"{CANDIDATE}-background.json"
    backgrounds = read(prior)["records"]
    records = []
    for cohort in ("melo","primed","jenny"):
        for row in read(RESULTS/f"4090-c161-history-gain-{cohort}-v1.json")["records"]:
            if row["phase"] != "background":
                continue
            key = f"{cohort}_{row['index']}"
            src = row["source"]
            path = Path(src.get("wav_path") or src["path"])
            r = backgrounds[key]
            assert src["pcm_sha256"] == r["pcm_sha256"] and not row["c161"]
            pcm, rate = sf.read(path, dtype="int16")
            assert rate == 16000 and digest(pcm) == r["pcm_sha256"]
            checked = confirmations(decoder, cmn, pcm, r["events"]["mined"])
            merged = choose([], [v["confirmed"] for v in checked if v["confirmed"] is not None])
            records.append(dict(key=key, source=r["source"], pcm_sha256=r["pcm_sha256"], samples=len(pcm),
                                proposals=checked, merged=merged))
    assert len(records)==66
    save(RESULTS/f"{STEM}-background-v1.json", dict(implementation=spec, reset_cmn=cmn,
                                                    prior_sha256=sha(prior), records=records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("speech","background"))
    if parser.parse_args().action == "speech":
        run()
    else:
        background()
