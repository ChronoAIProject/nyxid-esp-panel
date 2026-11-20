"""Compare a full and wake-cropped English transcript from the same reader."""
import argparse
import json
import re


def confirm(full_text, cropped_text, address):
    full = list(re.finditer(r"[A-Za-z]+", full_text))
    cropped = re.findall(r"[a-z]+", cropped_text.lower())
    boundary = len(full) - len(cropped)
    if not cropped or boundary <= 0 or [w[0].lower() for w in full[boundary:]] != cropped:
        return full_text, "full"
    pos = full[boundary].start()
    previous = [w[0].lower() for w in full[:boundary]]
    aliases = {"nyx": ("nyx", "nicks", "nick"), "willow": ("willow",)}
    if len(previous) == 2 and previous[0] in ("hey", "hye") and previous[1] in aliases[address]:
        return full_text[pos:], "address"
    if re.search(r"""[.!?]["'”’)]*\s*$""", full_text[:pos]):
        return full_text[pos:], "sentence"
    return full_text, "full"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", choices=("nyx", "willow"), required=True)
    parser.add_argument("--full", required=True)
    parser.add_argument("--cropped", required=True)
    args = parser.parse_args()
    text, reason = confirm(args.full, args.cropped, args.address)
    print(json.dumps(dict(text=text, rule=reason)))
