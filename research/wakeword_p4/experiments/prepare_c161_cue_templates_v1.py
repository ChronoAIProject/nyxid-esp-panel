"""Freeze two time-centered templates from the delivered synthetic target recordings."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from fit_c161_history_gain_v1 import PRIME, RESULTS, features, sha
from collect_multispeaker_formal_v1 import prepare

OUT = RESULTS.parent/"portable/c161_cue_templates_v1.h"
STEM = "4090-c161-cue-templates"


def main():
    lines = ["#pragma once", "#include <cstdint>", "", "namespace nyx {",
             "inline constexpr std::int16_t kCueTemplates[2][32][10] = {"]
    rows = []
    energies = []
    for version, end in (("v1",23200),("v2",17600)):
        path = RESULTS/f"hye-nyx-c161-synthetic-cue-{version}.wav"
        pcm, rate = sf.read(path, dtype="int16")
        assert rate == 16000
        full = np.r_[np.zeros(PRIME,np.int16),pcm]
        raw = features(full)
        _, support, _ = prepare(raw,len(full))
        n = len(raw)//3
        clock = ((support[:n] + 479)//480)*480 - PRIME
        at = int(np.flatnonzero(clock >= end)[0])
        blocks = np.minimum(raw[:3*n].astype(np.int32),666).reshape(n,3,10,4).sum(axis=(1,3))//12
        window = blocks[at-31:at+1]
        centered = 32*window-window.sum(0)
        assert centered.shape == (32,10) and np.max(np.abs(centered)) < 32768
        energy = 32*int((window**2).sum()) - int((window.sum(0)**2).sum())
        assert int((centered.astype(np.int64)**2).sum()) == 32*energy and energy > 0
        energies.append(energy)
        lines.append("    {")
        for frame in centered:
            lines.append("        {"+", ".join(str(v) for v in frame)+"},")
        lines.append("    },")
        rows.append(dict(version=version,wav_sha256=sha(path),prefix_end=end,template_end=int(clock[at]),
                         start_clock=int(clock[at-31]),energy=energy,blocks=window.tolist()))
    lines.extend(["};", "inline constexpr double kCueEnergy[2] = {"+", ".join(map(str,energies))+"};",
                  "}  // namespace nyx",""])
    OUT.write_text("\n".join(lines))
    with (RESULTS/f"{STEM}-v1.json").open("x") as f:
        json.dump(dict(records=rows, header_sha256=sha(OUT), threshold=0.85, rearm=0.55, cooldown=67),f,indent=2)
        f.write("\n")


if __name__=="__main__":
    main()
