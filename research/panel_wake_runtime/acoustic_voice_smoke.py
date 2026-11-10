"""Physical wake/request/reply smoke using local owner administration adapters.

Mac playback enters the panel microphone. The board uses its own stored Voice
key; the owner's local session only reads state and stops the test's own call.
No audio injection, transcript filtering or credentials are added to firmware.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.request

from pymongo import MongoClient


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--status-url", required=True)
    p.add_argument("--expected-app-sha256", required=True)
    p.add_argument("--expected-word", required=True)
    p.add_argument("--admin-module", type=Path, required=True)
    p.add_argument("--admin-base", required=True)
    p.add_argument("--mongodb-uri", required=True)
    p.add_argument("--database", required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--expected-input", required=True)
    p.add_argument("--expected-reply", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a fresh private output file")

    def status():
        with urllib.request.urlopen(args.status_url, timeout=5) as response:
            return json.load(response)

    before = status()
    voice = before["voice"]
    if (before["app_sha256"] != args.expected_app_sha256 or voice["busy"]
            or not voice["wake"]["listening"] or voice["wake"]["word"] != args.expected_word):
        raise ValueError("Development app or idle wake state differs")
    spec = importlib.util.spec_from_file_location("wake_test_owner_adapter", args.admin_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BASE = args.admin_base
    owner = module.LocalAPI()
    mongo = MongoClient(args.mongodb_uri, serverSelectionTimeoutMS=5000, socketTimeoutMS=5000)
    messages = mongo[args.database].assistant_messages
    started = time.monotonic()
    owned = None
    player = None
    report = {"scope": "Synthetic Mac playback through the full application; not human recognition acceptance",
              "expected_word": args.expected_word, "app_sha256": args.expected_app_sha256,
              "states": [], "passed": False}

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("Stopping owned acoustic test call")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        player = subprocess.Popen(["/usr/bin/afplay", str(args.audio)])
        quiet_since = None
        deadline = started + 90
        while time.monotonic() < deadline:
            value = status()
            v = value["voice"]
            report["states"].append({"seconds": round(time.monotonic() - started, 3),
                "state": v["state"], "message": v["message"], "active": v["active"],
                "wake": v["wake"], "heap": value["heap"]})
            if v["wake"]["detections"] > voice["wake"]["detections"]:
                if v["wake"]["detections"] != voice["wake"]["detections"] + 1:
                    raise AssertionError("Unexpected additional wake event")
                if not v["wake"]["initiated_call"]:
                    raise AssertionError("Wake did not own the call")
                owned = v["session_id"] or owned
                if v["state"] == "error":
                    raise AssertionError(v["message"])
            if owned:
                rows = list(messages.find({"voice.session_id": owned},
                    {"_id": 0, "role": 1, "text": 1, "seq": 1, "voice": 1}).sort("seq", 1))
                report["messages"] = rows
                sealed = {m["role"] for m in rows if m.get("voice", {}).get("sealed")
                          and m["voice"].get("complete")}
                if ({"user", "assistant"}.issubset(sealed) and v["active"]
                        and v["received_packets"] > 20 and v["wake"]["buffered_ms"] < 500
                        and v["message"] != "Bot speaking"):
                    quiet_since = quiet_since or time.monotonic()
                    if time.monotonic() - quiet_since >= 8:
                        for role, pattern in (("user", args.expected_input), ("assistant", args.expected_reply)):
                            group = [m for m in rows if m["role"] == role]
                            if len(group) != 1 or not re.search(pattern, group[0].get("text", ""), re.I):
                                raise AssertionError(f"Unexpected {role} count or incomplete utterance")
                        report["passed"] = True
                        break
                else:
                    quiet_since = None
            time.sleep(.3)
        if not report["passed"]:
            raise TimeoutError("Physical wake, complete request or reply was not observed")
        print(json.dumps({"passed": True, "expected_word": args.expected_word,
                          "messages": len(report["messages"])}), flush=True)
    finally:
        if player:
            if player.poll() is None:
                player.terminate()
            player.wait(timeout=10)
        try:
            current = status()["voice"]
            if owned and current["session_id"] == owned and current["busy"]:
                owner.call("/proxy/s/" + before["nyx"]["slug"] + "/voice/stop", "POST", {},
                           expected=(200, 201, 202))
                report["owned_call_stop_requested"] = True
        finally:
            mongo.close()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
