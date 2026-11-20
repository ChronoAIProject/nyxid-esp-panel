"""Compare two transcript views with the registered sentence/known-address boundary."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import confirm_received_sentence as policy
import read_parts_confirmation_commands as parts
import score_bella_received_word_times as prior

P, scoring = prior.P, prior.scoring
OUT = prior.OUT.with_name("4090-bella-confirmed-sentences.json")
PROTOCOL = prior.PROTOCOL.with_name("bella_received_suffix_protocol.md")


def compare(full, crop, command, address):
    chosen, reason = policy.confirm(full, crop, address)
    scores = {method:prior.scores(prior.normalized(text), prior.normalized(command))
              for method, text in (("full", full), ("crop", crop), ("confirmed", chosen))}
    return dict(full=full, crop=crop, confirmed=chosen, reason=reason, scores=scores)


def aggregate(records):
    counts = defaultdict(Counter)
    for r in records:
        groups = [r["group"]]
        if r["group"].startswith("interruptions/"):
            groups.append("interruptions/all")
        for group in groups:
            c = counts[group]
            c.update(cases=1, opened=r["opened"] is not None)
            for condition, readers in r["conditions"].items():
                for reader, reading in readers.items():
                    modes = reading["scores"]
                    for metric in ("exact", "suffix"):
                        key = condition + "_" + reader + "_" + metric
                        for method in modes:
                            c[key + "_" + method] += modes[method][metric]
                        c[key + "_gained"] += modes["confirmed"][metric] and not modes["full"][metric]
                        c[key + "_lost"] += modes["full"][metric] and not modes["confirmed"][metric]
    return counts


def existing(base, earlier, origin):
    source = {r["label"]:r for r in P.read(base)["records"]}
    for old in P.read(earlier)["records"]:
        label = old["label"]
        row = source[label]
        role = row.get("role", "nyx")
        group = origin + "/" + (row["mode"] + "/" + role if origin == "rooms" else str(row["level"]))
        out = dict(group=group, label=label, opened=row["opened"], command=row["command"], conditions={})
        for cond, c in old["conditions"].items():
            pairs = {}
            for reader in ("qwen", "turbo"):
                pairs[reader] = compare(row["readings"][cond][reader]["text"], c["readings"][reader]["text"],
                                        row["command"], role)
                for metric in ("suffix", "exact"):
                    for method in ("full", "crop"):
                        assert pairs[reader]["scores"][method][metric] == c[method][reader + "_" + metric]
            out["conditions"][cond] = pairs
        yield out


def new_crop(work, logs, record, command, address, readers):
    proc, model, turbo, decoder = readers
    conditions = {}
    for cond, audio, offset in prior.read_sends(work, logs, record, decoder):
        clip = audio[offset:]
        crops = dict(qwen=scoring.prior.evaluate(proc, model, clip, command),
                     turbo=scoring.prior.recognition(turbo, clip, command))
        full = record["readings"][cond]
        conditions[cond] = {reader:dict(**compare(full[reader]["text"], crops[reader]["text"], command, address),
                             offset=offset, crop_pcm_sha256=crops["qwen"]["pcm_sha256"],
                             full_pcm_sha256=full["qwen"]["pcm_sha256"]) for reader in crops}
    return conditions


def main():
    baseline = prior.trial.baseline
    historical = prior.history
    paths = (Path(__file__), Path(policy.__file__), Path(prior.__file__), PROTOCOL, historical.OUT,
             historical.historical.OUT, baseline.OUT, prior.trial.OUT, parts.OUT, parts.commands.OUT, scoring.OUT)
    records = list(existing(baseline.OUT, prior.trial.OUT, "rooms"))
    records.extend(existing(historical.historical.OUT, historical.OUT, "interruptions"))
    assert len(records) == 144
    ordinary = P.read(parts.OUT)
    original = P.read(parts.commands.OUT)
    case_by_id = {r["index"]:r for r in original["records"]}
    assert P.sha(parts.commands.TRACE) == original["archive_sha256"] == ordinary["archive_sha256"]
    proc, model, identity = scoring.prior.load()
    assert identity == original["qwen"] and "4090" in identity["device"]
    decoder = scoring.prior.decoder_api()
    assert decoder.opus_get_version_string().decode() == original["decoder"]
    readers = proc, model, scoring.prior.reader(original["turbo"]), decoder
    with TemporaryDirectory(dir=baseline.study.WORK) as folder:
        work = Path(folder)
        cache = {}
        with ZipFile(parts.commands.TRACE) as logs:
            for r in ordinary["records"]:
                idx, opened, route = r["index"], r["opened"], r["route"]
                label = f"{idx}/{opened}"
                rec = dict(group="ordinary/" + route, label=str(idx), voice=r["voice"], command=r["command"],
                           opened=opened, conditions={})
                records.append(rec)
                if opened is None:
                    continue
                src = case_by_id[idx]
                assert src["pcm_sha256"] == r["pcm_sha256"] and src["command"] == r["command"]
                if label not in cache:
                    row = dict(src["received"][str(opened)], label=label, uart=r["uart"])
                    cache[label] = new_crop(work, logs, row, r["command"],
                                            "willow" if route == "willow" else "nyx", readers)
                rec["conditions"] = cache[label]
        assert len(cache) == 20
        control = P.read(scoring.OUT)
        assert P.sha(scoring.TRACE) == control["archive_sha256"]
        r, = [r for r in control["records"] if r["role"] == "absent" and r["opened"] is not None]
        assert r["index"] == 72
        row = dict(r["uplink"], label=r["label"], uart={r["label"] + "/" + stage + ".log":
                   r["uplink"][stage + "_uart_sha256"] for stage in ("opus", "sender")})
        with ZipFile(scoring.TRACE) as logs:
            false_open = dict(label=r["label"], command=r["command"], opened=r["opened"],
                              conditions=new_crop(work, logs, row, r["command"], "nyx", readers))
        assert len(records) == 180
        P.save(work / OUT.name, dict(records=records, counts=aggregate(records), false_open=false_open,
               qwen=identity, turbo=original["turbo"], decoder=original["decoder"],
               sources={path.name:P.sha(path) for path in paths},
               scope="144 fixed-sound inputs plus 36 route observations on 24 unmarked Nyx and Willow "
                     "recordings. Twenty distinct ordinary opens. Both sender schedules. Separate false open."))
        (work / OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
