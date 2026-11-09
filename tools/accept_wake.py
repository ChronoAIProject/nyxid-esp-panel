#!/usr/bin/env python3
"""Play a wake word + request on this Mac and verify the panel's real voice path.

Requires nyxid owner login, a nearby panel, and a WAV/AIFF test phrase containing
Hi ESP immediately followed by a harmless request. No provider credentials or
audio injection endpoints are used. This script hangs up its own test call.
"""
import argparse
import json
import re
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True)
    parser.add_argument("--conversation", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-input", help="Regex that must match the complete user transcript")
    parser.add_argument("--expected-reply", help="Regex that must match the complete Bot reply")
    args = parser.parse_args()

    def proxy(path, body=None):
        cmd = ["nyxid", "proxy", "request", args.service, path, "--output", "json"]
        if body is not None:
            cmd += ["-m", "POST", "-H", "Content-Type:application/json", "-d", json.dumps(body)]
        return json.loads(subprocess.check_output(cmd, text=True, timeout=30))

    def history():
        return json.loads(subprocess.check_output([
            "nyxid", "nyxbot", "history", args.conversation, "--limit", "30", "--output", "json"
        ], text=True, timeout=30))["messages"]

    accept(proxy, history, args.audio, args.output, args.expected_input, args.expected_reply)


def accept(proxy, history, audio, output, expected_input=None, expected_reply=None):
    """Exercise physical audio; adapters only supply owner status/history reads."""
    initial = proxy("/voice")
    assert not initial["busy"] and initial["wake"]["listening"], initial
    before = max((message["seq"] for message in history()), default=0)
    start = time.monotonic()
    player = subprocess.Popen(["/usr/bin/afplay", str(audio)])
    session = None
    quiet_since = None
    samples = []
    try:
        deadline = start + 90
        while time.monotonic() < deadline:
            value = proxy("/voice")
            samples.append({"at": round(time.monotonic()-start, 2), "state": value["state"], "message": value["message"],
                            "active": value["active"], "wake": value["wake"]})
            if value["wake"]["detections"] > initial["wake"]["detections"]:
                assert value["wake"]["initiated_call"]
                assert value["greeting_state"] == "user_speaking"
                session = value["session_id"] or session
                if value["state"] == "error":
                    raise AssertionError(value["message"])
            if session and value["active"] and value["received_packets"] > 20 and value["wake"]["buffered_ms"] < 500:
                messages = [message for message in history() if message["seq"] > before
                            and (message.get("voice") or {}).get("session_id") == session]
                finished = {message["role"] for message in messages
                            if message["voice"].get("sealed") and message["voice"].get("complete")}
                if {"user", "assistant"}.issubset(finished) and value["message"] != "Bot speaking":
                    if quiet_since is None:
                        quiet_since = time.monotonic()
                    if time.monotonic()-quiet_since < 8:
                        time.sleep(1)
                        continue
                    report = {"session_id": session, "samples": samples,
                              "messages": [{"role": m["role"], "text": m.get("text"), "voice": m.get("voice")} for m in messages],
                              "voice": value}
                    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
                    assert len([m for m in messages if m["role"] == "user"]) == 1, "Playback triggered another input turn; see saved report"
                    assert len([m for m in messages if m["role"] == "assistant"]) == 1, "Unexpected additional reply; see saved report"
                    for role, pattern in [("user", expected_input), ("assistant", expected_reply)]:
                        if pattern:
                            assert re.search(pattern, next(m.get("text", "") for m in messages if m["role"] == role), re.I), f"Incomplete {role} speech; see saved report"
                    print(json.dumps(report, ensure_ascii=False), flush=True)
                    return
                else:
                    quiet_since = None
            # Signed panel requests retain replay nonces for 60 seconds.
            # Leave capacity for ordinary service calls during acceptance.
            time.sleep(1)
        output.write_text(json.dumps({"session_id": session, "samples": samples,
            "messages": [{"role": m["role"], "text": m.get("text"), "voice": m.get("voice")} for m in history()
                         if m["seq"] > before and (m.get("voice") or {}).get("session_id") == session]},
            ensure_ascii=False, indent=2)+"\n")
        raise AssertionError("Wake/connection/replayed speech/response timed out")
    finally:
        if player.poll() is None:
            player.terminate()
        player.wait(timeout=10)
        value = proxy("/voice")
        if value["wake"]["initiated_call"] and value["wake"]["detections"] > initial["wake"]["detections"]:
            proxy("/voice/stop", {})


if __name__ == "__main__":
    main()
