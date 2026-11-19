"""Freeze reusable literal cues and singular-name controls on the original 18 command clocks."""
from io import BytesIO
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import check_early_gate_voices as voices
import compare_synthesized_call_context as context
from package_context_playback import add, formatted

ch = voices.ch
STEM = "4090-c161-reusable-nyx"
PLAN = ch.RESULTS / (STEM + "-plan.json")
PACK = ch.RESULTS / (STEM + "-cues.zip")
BASE = voices.study.matched.PLAN


def prefixes():
    with ZipFile(PACK) as pack:
        manifest = json.loads(pack.read("manifest.json"))
        for r in manifest["records"]:
            data = pack.read(r["file"])
            pcm, rate = sf.read(BytesIO(data), dtype="int16")
            assert rate == 16000 and ch.digest(pcm) == r["pcm_sha256"]
            yield r, pcm


def inputs():
    plan = ch.read(BASE)
    silence = [r for r in plan["records"] if r["cue"] == "silence"]
    assert len(silence) == 18
    for cue, clip in prefixes():
        for src in [None, *silence]:
            if src is None:
                base = np.zeros(80000, dtype="<i2")
                end = 48000
                extra = dict(pair=None, command=None, command_voice=None)
            else:
                base = voices.study.replay.probe.dialogue.piper.load_wav(src)
                end = src["end"]
                assert ch.digest(base) == src["pcm_sha256"]
                assert len(clip) <= end - src["start"]
                extra = dict(pair=src["pair"], command=src["carrier"], command_voice=src["voice"])
            at = end - len(clip)
            pcm = base.copy()
            assert not pcm[at:end].any()
            pcm[at:end] = clip
            yield dict(cue=cue["index"], voice=cue["voice"], role=cue["role"], style=cue["style"],
                       cue_at=at, cue_end=end, samples=len(pcm), pcm_sha256=ch.digest(pcm),
                       prefix_sha256=ch.digest(base[:at]), suffix_sha256=ch.digest(base[end:]), **extra), pcm


def main():
    rows = []
    tmp = PACK.with_suffix(".tmp")
    with ZipFile(tmp, "w", ZIP_DEFLATED) as archive:
        for src in ch.read(context.PLAN)["records"]:
            if src["carrier"] != context.COMMANDS[0] or src["target_id"] not in ("Hey Nyx", "Hey Nick"):
                continue
            assert ch.sha(Path(src["path"])) == src["sha256"]
            data = voices.study.replay.probe.dialogue.piper.load_wav(src)
            clip = data[src["start"]:src["end"]]
            assert ch.digest(clip) == src["cue"]["cue_sha256"]
            name = f'{src["voice"]}-{src["target_id"].replace(" ", "-").lower()}-{src["style"]}.wav'
            out = BytesIO()
            sf.write(out, clip, 16000, format="WAV", subtype="PCM_16")
            wav = out.getvalue()
            rows.append(dict(index=len(rows), source_index=src["index"], voice=src["voice"], role=src["target_id"],
                             style=src["style"], file=name, samples=len(clip), sha256=hashlib.sha256(wav).hexdigest(),
                             pcm_sha256=ch.digest(clip)))
            add(archive, name, wav)
        assert len(rows) == 16
        add(archive, "manifest.json", formatted(dict(records=rows, source_plan_sha256=ch.sha(context.PLAN))))
    tmp.replace(PACK)
    records = [r for r, _ in inputs()]
    assert len(records) == 304
    result = dict(records=records, sources={p.name: ch.sha(p) for p in (PACK, BASE, context.PLAN,
                  voices.gate.fit.COEFF, ch.RESULTS/"4090-c161-late-nick-gate-fit.json", Path(__file__))},
                  policy="Use every voice from the first listed context command for both synthesis styles and "
                  "both literal names. Copy the cue PCM unchanged before each of the same 18 Piper command "
                  "recordings, and alone with two seconds of silence after the cue. All voices, clips and "
                  "commands already exposed. Confirm prior and paired Nick veto, C161 standalone and first "
                  "P4 Opus complete suffix on literal Nyx at a matching first OPEN. No waveform search.")
    if PLAN.exists():
        assert ch.read(PLAN) == result
    else:
        ch.save(PLAN, result)


if __name__ == "__main__":
    main()
