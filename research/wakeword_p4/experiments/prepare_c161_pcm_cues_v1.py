"""Select two disjoint energetic 80 ms waveform anchors per frozen synthetic cue."""
import numpy as np
import soundfile as sf
from fit_c161_history_gain_v1 import RESULTS, sha, save


def mono2k(pcm):
    groups = pcm[:len(pcm)//8*8].astype(np.int32).reshape(-1,8)
    mean = np.trunc(groups.sum(1)/8).astype(np.int32)
    highpass = 16*mean - 14*np.r_[0,mean[:-1]]
    return np.clip(np.trunc(highpass/256),-2047,2047).astype(np.int16)


def main():
    rows,anchors,deltas = [],[],[]
    for ver in ("v1","v2"):
        path=RESULTS/f"hye-nyx-c161-synthetic-cue-{ver}.wav"
        pcm,rate=sf.read(path,dtype="int16")
        assert rate==16000
        samples=mono2k(pcm)
        energy=np.square(samples.astype(np.float64))
        mass=energy.cumsum()/energy.sum()
        left,mid,right=np.searchsorted(mass,[.01,.52,.99])
        windows=np.convolve(energy,np.ones(160),"valid")
        first=left+int(np.argmax(windows[left:mid-159]))
        second=mid+int(np.argmax(windows[mid:right-159]))
        assert 160 <= second-first < 990
        values=np.stack((samples[first:first+160],samples[second:second+160]))
        assert np.all(np.abs(values)<2048)
        anchors.append(values)
        deltas.append(int(second-first))
        rows.append(dict(cue=ver,wav_sha256=sha(path),starts_2khz=[int(first),int(second)],
                         energies=(values.astype(np.int64)**2).sum(axis=1).tolist()))
    text=["#pragma once","#include <cstdint>","","namespace nyx {",
          "inline constexpr int kAnchorDelays[2] = {"+", ".join(map(str,deltas))+"};",
          "inline constexpr std::int16_t kWaveAnchors[2][2][160] = {"]
    for values in anchors:
        text.append("    {")
        for anchor in values:
            text.append("        {")
            for j in range(0,160,16):text.append("            "+", ".join(map(str,anchor[j:j+16]))+",")
            text.append("        },")
        text.append("    },")
    text.extend(["};", "inline constexpr int kAnchorEnergy[2][2] = {"])
    for r in rows:text.append("    {"+", ".join(map(str,r["energies"]))+"},")
    text.extend(["};","}  // namespace nyx",""])
    header=RESULTS.parent/"portable/c161_pcm_cue_coefficients_v1.h"
    header.write_text("\n".join(text))
    save(RESULTS/"4090-c161-pcm-cue-anchors-v1.json",dict(records=rows,header_sha256=sha(header),
         nominal_rate=2000,tolerance_samples=24,correlation=0.68,cooldown_samples=32160))


if __name__=="__main__":
    main()
