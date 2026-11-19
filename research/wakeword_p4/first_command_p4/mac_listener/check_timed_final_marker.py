"""Replay the original v5 rooms, controls and continuous recordings with stricter timing on weak confirmations."""
from pathlib import Path

import check_final_marker_background as background

host = background.host
ch = host.ch
HEADER = host.HEADER.with_name("c161_timed_final_marker.h")
OUT = ch.RESULTS / "4090-c161-timed-final-marker.json"
BG = OUT.with_name(OUT.stem + "-background.json")


def binding(work):
    return host.binding(work, header_text=HEADER.read_text(), matcher_class="TimedFinalMarker")


def main():
    sources = (HEADER, Path(__file__))
    host.main(output=OUT, make_matcher=binding, sources=sources, policy=(
        "Original v5 and frozen C161. Preserve strong confirmations within 120 +/- 1.5 ms of a same-sign "
        "middle marker. Weak third markers require 120 +/- 0.5 ms. The original across event motivated "
        "the timing constraint. Previously consumed, paired development inputs."))
    background.main(output=BG, make_matcher=binding, grid_report=OUT, extra_sources=sources)


if __name__ == "__main__":
    main()
