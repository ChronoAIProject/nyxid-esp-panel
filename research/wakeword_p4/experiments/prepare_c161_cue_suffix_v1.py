"""Choose an original high-frequency signature within the generated /ks/ suffix."""
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from fit_c161_history_gain_v1 import RESULTS, read, save, sha


def main():
    a = read(RESULTS/"4090-c161-vowel-name-sources-v1.json")["records"][0]
    ledger = Path(".local/wakeword/hey-broad-kokoro-native-v1/records.jsonl")
    b, = [r for r in map(json.loads,ledger.read_text().splitlines()) if r["request"]["conditioning_id"]=="af_kore"
          and r["request"]["speed"]==1 and r["request"]["text"]=="Hey Nicks."]
    duration = ledger.parent/b["files"]["duration"]["path"]
    assert sha(duration)==b["files"]["duration"]["sha256"]
    anchors = read(RESULTS/"4090-c161-pcm-cue-anchors-v1.json")["records"]
    rows, signatures = [],[]
    for i,(version, phones, durations) in enumerate((
            ("v1",a["phones"],a["duration"]), ("v2",b["metadata"]["phonemes"],np.load(duration).ravel()))):
        path=RESULTS/f"hye-nyx-c161-synthetic-cue-{version}.wav"
        pcm,rate=sf.read(path,dtype="int16")
        assert rate==16000 and sha(path)==anchors[i]["wav_sha256"] and phones.endswith("ɪks.")
        # Model durations give a search interval, not a phonetic alignment.
        starts=np.r_[0,np.cumsum(durations)].astype(int)*200  # 8 kHz
        left,right=starts[1+phones.index("k")],starts[2+phones.index("s")]
        low=np.trunc(pcm[:len(pcm)//2*2].astype(np.int32).reshape(-1,2).sum(1)/2).astype(np.int32)
        hp=np.clip(np.trunc((low-np.r_[0,low[:-1]])/16),-2047,2047).astype(np.int16)
        power=np.convolve(hp.astype(float)**2,np.ones(1024),"valid")
        at=int(left+np.argmax(power[left:right-1024+1]))
        seg=hp[at:at+1024].astype(np.int64)
        assert len(seg)==1024
        delay=at+1024-(anchors[i]["starts_2khz"][1]+160)*4
        assert delay>128 and seg@seg>0
        signatures.append(seg)
        rows.append(dict(cue=version,range_8khz=[int(left),int(right)],start_8khz=at,
                         delay_8khz=delay,energy=int(seg@seg),wav_sha256=sha(path)))
    lines=["#pragma once","#include <cstdint>","","namespace nyx {",
           "inline constexpr int kSuffixDelays[2] = {"+", ".join(str(r["delay_8khz"]) for r in rows)+"};",
           "inline constexpr int kSuffixEnergy[2] = {"+", ".join(str(r["energy"]) for r in rows)+"};",
           "inline constexpr std::int16_t kSuffixes[2][1024] = {"]
    for sig in signatures:
        lines.append("    {")
        for j in range(0,len(sig),16):lines.append("        {0},".format(", ".join(map(str,sig[j:j+16]))))
        lines.append("    },")
    lines.extend(["};","} // namespace nyx",""])
    out=RESULTS.parent/"portable/c161_cue_suffix_coefficients_v1.h"
    out.write_text("\n".join(lines))
    save(RESULTS/"4090-c161-cue-suffix-templates-v1.json",dict(records=rows,header_sha256=sha(out),threshold=.20,
         half_width_8khz=128))


if __name__=="__main__":
    main()
