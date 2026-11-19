"""Index first ordinary-speech and Bot notifications from the frozen music-proposal replay."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results"
WORD = "4090-c161-music-proposal"
OUTPUT = RESULTS / f"{WORD}-first-notifications.json"


def main():
    hashes = {}

    def read(file):
        data = (RESULTS/file).read_bytes()
        hashes[file] = sha256(data).hexdigest()
        return json.loads(data)

    inventory = {}
    for cohort in ("melo", "primed", "jenny"):
        for row in read(f"4090-c161-history-gain-{cohort}-v1.json")["records"]:
            if row["phase"] == "background":
                inventory[f"{cohort}_{row['index']}"] = row["source"]
    spans = defaultdict(lambda: dict(spans=0, notified=0, events=0, c161_spans=0))
    background, positives = [], []
    for group in ("background", "context", "daily", "bot"):
        for row in read(f"{WORD}-{group}.json")["records"]:
            events = row["events"]
            accepted = {chk["end"]:chk for chk in row["checks"] if chk["codes"][0] >= 0}
            assert events == sorted(set(events))
            assert all(t in accepted or t in row["c161"] for t in events)

            def first(at):
                check = accepted.get(at)
                return dict(sample=at, seconds=at/16000, c161=at in row["c161"],
                            accepted_proposal=check)

            if group == "background":
                source = inventory.pop(row["key"])
                assert source["pcm_sha256"] == row["pcm_sha256"] and source["samples"] == row["samples"]
                assert [hit["sample"] for hit in row["false_notifications"]] == events
                record = dict(key=row["key"], source_id=source.get("source_id") or source["id"], pcm_sha256=row["pcm_sha256"],
                              samples=row["samples"], c161=row["c161"], events=len(events), first=None)
                if events:
                    hit = row["false_notifications"][0]
                    when = (events[0]-hit["begin"])/16000
                    words = hit["asr"]["words"]
                    nearby = "".join(w["word"] for w in words if w["end"] >= when-1.0 and w["start"] <= when+1.0)
                    record["first"] = dict(**first(events[0]), asr_nearby=nearby.strip(),
                                           excerpt_begin=hit["begin"], excerpt_end=hit["end"])
                background.append(record)
                continue

            assigned = []
            for span in row["source"]["spans"]:
                inside = [t for t in events if span["start"] < t <= span["end"]]
                assigned.extend(inside)
                summary = spans[span["address"]]
                summary["spans"] += 1
                summary["notified"] += bool(inside)
                summary["events"] += len(inside)
                summary["c161_spans"] += any(span["start"] < t <= span["end"] for t in row["c161"])
                if inside:
                    positives.append(dict(group=group, key=row["source"]["key"], pcm_sha256=row["pcm_sha256"],
                                          **span, first=first(inside[0]), events=inside))
            for t in sorted(set(events) - set(assigned)):
                positives.append(dict(group=group, key=row["source"]["key"], pcm_sha256=row["pcm_sha256"],
                                      address="outside_annotated_spans", first=first(t), events=[t]))
    assert not inventory
    OUTPUT.write_text(json.dumps(dict(sources=hashes, background=background, by_address=spans,
        annotated_notifications=positives, policy="Existing consumed streams. First merged sample in each continuous background file or annotated span. "
        "Preserve additional-event counts and source PCM. Text near a background event is the archived ASR around the notification, "
        "not a causal trigger label. C161 and accepted auxiliary checks are reported independently."), indent=2)+"\n")


if __name__ == "__main__":
    main()
