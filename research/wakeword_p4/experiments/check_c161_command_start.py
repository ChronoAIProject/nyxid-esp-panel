"""Audit first-command loss and recover the recording start from past audio at each wake."""
from itertools import groupby
from pathlib import Path
import re
from time import perf_counter_ns

import torch

import check_c161_address_verb_challenge as verbs
import check_c161_melo_address_grammar as original
import check_c161_onset_letters as latest
import check_c161_onset_names_melo as names
import train_c161_onset_letters as trained
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_v5_command_ns import digest
from check_c161_v5_far_command_noise import normalized
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from run_c161_first_command_v5 import transcripts
from select_c161_synthesized_cue_v2 import load_wav

PRIOR = RESULTS / f"{latest.STEM}-evaluation.json"
OUTPUT = RESULTS / "4090-c161-command-start.json"
PLANS = {"melo": original.PLAN, "new_melo": names.PLAN, "challenge": verbs.PLAN, "next": latest.PLAN}
TARGETS = {"Hye Nyx", "Hey Nyx", "Hi Nicks", "Hey Nicks"}
INITIAL = {"switch": "s w ih ch", "open": "ow p ah n", "save": "s ey V", "turn": "t er n",
           "unlock": "ah n l aa k", "dim": "d ih m", "move": "m uw V"}
MARGIN = 1920  # 120 ms at 16 kHz
# A word-end match is required. The following word is open vocabulary.
ADDRESS = re.compile(r"^(?:HEY|HAY|HEI|HIGH|HI)[|]?(?:NICKS?|NIXS?|NYX|NECKS?|NEXT?|NEXS?)(?=\||$)")


def phone_bounds(span, carrier, separator):
    phones = span["phones"]
    assert phones[:7] == ["_", "hh", phones[2], "n", "ih", "k", "s"] and phones[2] in ("ey", "ay")
    i = 7
    if separator == ", ":
        assert phones[i] == ","
        i += 1
    expected = INITIAL[carrier.split()[0]].split()
    assert phones[i:i+len(expected)] == expected
    durations = span["token_duration_frames"]
    ids = span["phone_ids"]
    assert len(durations) == len(ids) == 2*len(phones)+1 and all(k == 0 for k in ids[::2])
    samples = lambda frames: round(frames*512*160/441)
    assert abs(samples(sum(durations)) - (span["end"]-span["start"])) <= 1
    before = sum(durations[:2*i+1])
    return tuple(span["start"] + samples(f) for f in (before, before+durations[2*i+1]))


@torch.inference_mode()
def choose_start(model, processor, prefix, fallback):
    # This function only receives audio through the notification delivery block.
    capture = trained.cropped(prefix, len(prefix))
    origin = len(prefix) - len(capture)
    values = processor(capture.astype("float32")/32768, sampling_rate=16000, return_tensors="pt").input_values
    ids = model(values.cuda()).logits.argmax(dim=-1)[0].cpu().tolist()
    tokenizer = processor.tokenizer
    frames, symbols = [], []
    for token, group in groupby(enumerate(ids), key=lambda pair: pair[1]):
        frame = next(group)[0]
        if token != tokenizer.pad_token_id:
            frames.append(frame)
            symbols.append(tokenizer.convert_ids_to_tokens(token))
    chars = "".join(symbols)
    match = ADDRESS.match(chars.lstrip("|"))
    start = fallback
    reason = "fallback"
    if match:
        pos = len(chars)-len(chars.lstrip("|"))+match.end()
        if pos+1 < len(symbols) and chars[pos] == "|":
            frame = frames[pos+1]
            reason = "following_word"
        else:
            frame = frames[pos-1]
            reason = "last_name_letter"
        start = max(fallback, origin+320*frame-MARGIN)
    return dict(first=start, reason=reason, chars=chars, input_sha256=digest(capture), frames=frames)


def main():
    torch.set_num_threads(1)
    gpu = torch.cuda.get_device_name()
    assert "4090" in gpu
    model, processor, identity = trained.teacher()
    assert model.config.inputs_to_logits_ratio == 320 and model.config.pad_token_id == processor.tokenizer.pad_token_id
    assert all(len(k) == 1 for k in processor.tokenizer.get_vocab() if k not in processor.tokenizer.all_special_tokens)
    reader, wh = load_reader()
    prior = read(PRIOR)
    assert wh == prior["asr"] and identity == prior["pretrained"]
    refs = {r["pcm_sha256"]:r for r in prior["records"] if r["cohort"] in PLANS}
    rows, views, decisions = [], {}, []
    for cohort, plan in PLANS.items():
        for src in read(plan)["records"]:
            pcm = load_wav(src)
            r = refs.pop(digest(pcm))
            assert r["cohort"] == cohort
            by_hit = {}
            # Evaluate the trim rule on every actual notification, including distractors.
            for hit in r["events"]["c161_pretrained"]:
                available = (hit+159)//160*160
                assert available <= len(pcm)
                first = max(0, available-32000)
                tick = perf_counter_ns()
                d = choose_start(model, processor, pcm[:available], first)
                elapsed_us = (perf_counter_ns()-tick)//1000
                d.update(hit=hit, available=available, base_first=first, wall_us=elapsed_us,
                         source_pcm_sha256=r["pcm_sha256"])
                by_hit[hit] = d
                decisions.append(d)
            want = normalized(src["carrier"])
            for span, archived in zip(src["timeline"], r["rows"], strict=True):
                if span["address"] not in TARGETS:
                    continue
                begin, end = span["start"], span["end"]
                assert digest(pcm[begin:end]) == span["pcm_sha256"] == archived["full_clip_asr"]
                start, stop = phone_bounds(span, src["carrier"], src["separator"])
                hits = archived["routes"]["c161_pretrained"]["hits"]
                cuts = {"full":(begin,end), "label_margin":(start-MARGIN,end+6400)}
                decision = None
                if hits:
                    decision = by_hit[hits[0]]
                    cuts.update(preroll=(decision["base_first"],end+6400), ctc=(decision["first"],end+6400))
                recordings = {}
                for label,(left,right) in cuts.items():
                    assert 0 <= left < right <= len(pcm)
                    audio = pcm[left:right]
                    key = digest(audio)
                    if key not in views:
                        saved = r["transcripts"].get(key)
                        views[key] = transcripts(reader,audio) if saved is None else saved
                    text = normalized(views[key]["text"])
                    recordings[label] = dict(first=left, end=right, sha256=key, exact=text[-len(want):] == want,
                                             first_phone_samples_lost=max(0,min(left,stop)-start))
                prior_capture = archived["routes"]["c161_pretrained"]
                if hits:
                    assert (recordings["preroll"]["sha256"], recordings["preroll"]["exact"]) == (
                        prior_capture["asr"], prior_capture["exact_command"])
                rows.append(dict(cohort=cohort, source_pcm_sha256=r["pcm_sha256"], voice=src["voice"],
                                 carrier=src["carrier"], separator=src["separator"], address=span["address"],
                                 vowel=span["phones"][2], start=begin, end=end, first_phone=[start,stop],
                                 hits=hits, reason=decision["reason"] if decision else None, recordings=recordings))
        print(cohort, len(rows), flush=True)
    assert not refs and len(rows) == 128
    save(OUTPUT,dict(policy="Frozen C161 + pretrained notifications. Standard 2 s pre-roll, 400 ms tail. Label margin uses generator durations and is diagnostic only. CTC reads the last 1.44 s at the delivered notification, uses no command vocabulary, subtracts 120 ms from the following word or last name letter, and otherwise retains the standard recording. Timing is unbatched host wall time on 4090 including CPU input and output conversion. ASR endpoint uses the same annotated end plus tail as the original evaluation.",
                     sources={p.name:sha(p) for p in [PRIOR,*PLANS.values(),Path(__file__)]}, margin_samples=MARGIN,
                     pretrained=identity, asr=wh, gpu=gpu, records=rows, decisions=decisions, transcripts=views))


if __name__ == "__main__":
    main()
