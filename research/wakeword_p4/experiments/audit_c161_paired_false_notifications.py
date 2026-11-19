"""Identify the source and transcript of each emitted background and Bot notification."""
from pathlib import Path

import soundfile as sf

import score_c161_paired_rooms as score
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from run_c161_first_command_v5 import transcripts


def main():
    stem = score.paired.STEM
    background = RESULTS / (stem + "-background.json")
    bot = RESULTS / (stem + "-bot.json")
    rows = read(background)["records"]
    reader, identity = score.word.load_reader()
    clips = []
    for cohort in ("melo", "primed", "jenny"):
        for source in read(RESULTS / f"4090-c161-history-gain-{cohort}-v1.json")["records"]:
            if source["phase"] != "background":
                continue
            key = f"{cohort}_{source['index']}"
            r, s = rows[key], source["source"]
            assert source["c161"] == [] and s["pcm_sha256"] == r["pcm_sha256"]
            events = sorted({t for values in r["events"].values() for t in values})
            if not events:
                continue
            path = Path(s.get("wav_path") or s["path"])
            assert sha(path) == (s.get("wav_sha256") or s["sha256"])
            for t in events:
                begin, end = max(0, t-80000), min(r["samples"], t+48000)
                pcm, rate = sf.read(path, dtype="int16", start=begin, stop=end)
                assert rate == 16000 and pcm.shape == (end-begin,)
                clips.append(dict(key=key, source_pcm_sha256=r["pcm_sha256"], sample=t, begin=begin, end=end,
                                  routes=[k for k,v in r["events"].items() if t in v],
                                  origin="verified word proposal", pcm_sha256=score.poll.digest(pcm),
                                  asr=transcripts(reader, pcm)))
    bot_events = []
    for r in read(bot)["records"]:
        for t in sorted({t for values in r["routes"].values() for t in values}):
            span = next((s for s in r["rows"] if s["start"] <= t < s["end"]), None)
            bot_events.append(dict(key=r["meta"]["key"], pcm_sha256=r["pcm_sha256"], sample=t,
                                   span=span["address"] if span else None, text=span["text"] if span else None,
                                   routes=[k for k,v in r["routes"].items() if t in v],
                                   origin="c161" if t in r["routes"]["c161"] else "verified word proposal"))
    save(RESULTS / (stem + "-false-notifications.json"), dict(background=clips, bot=bot_events, asr=identity,
         background_sha256=sha(background), bot_sha256=sha(bot), script_sha256=sha(Path(__file__)),
         policy="Union of emitted events on full recordings. Eight-second Whisper transcripts provide context."))


if __name__ == "__main__":
    main()
