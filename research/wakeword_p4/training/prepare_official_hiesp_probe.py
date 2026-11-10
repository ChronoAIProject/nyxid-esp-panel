"""Extract pinned ESP-SR official Hi ESP fixture for developer transport controls."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import numpy as np
import soundfile as sf

REVISION='b9dc43a329a68de2f72147808fc9ea29d2ace71e'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    a.output.mkdir(parents=True)
    sources=[]
    for name in ('samples/hiesp.h','test_wakenet.cpp'):
        url=f'https://raw.githubusercontent.com/espressif/esp-sr/{REVISION}/test_apps/esp-sr-wakenet/main/{name}'
        data=urllib.request.urlopen(url,timeout=40).read()
        path=a.output/Path(name).name;path.write_bytes(data)
        sources.append(dict(url=url,sha256=sha(path),bytes=len(data)))
    text=(a.output/'hiesp.h').read_text()
    match=re.search(r'const unsigned char hiesp\[\]\s*=\s*\{(.*?)\};',text,re.S)
    if not match:raise ValueError('Changed fixture declaration')
    tokens=re.findall(r'0x([0-9a-fA-F]{2})',match[1])
    if re.sub(r'0x[0-9a-fA-F]{2}|[\s,]','',match[1]):raise ValueError('Unsupported byte initializer')
    raw=bytes(int(t,16) for t in tokens)
    if len(raw)!=62464:raise ValueError('Changed official fixture length')
    pcm=np.frombuffer(raw,dtype='<i2');records=[]
    for lead in (0,28800):
        signal=np.pad(pcm,(lead,16000)).astype('<i2');path=a.output/f'official-lead{lead}.wav'
        sf.write(path,signal,16000,subtype='PCM_16')
        decoded,rate=sf.read(path,dtype='int16');np.testing.assert_array_equal(signal,decoded)
        records.append(dict(path=path.name,role='development-transport-probe',split='validation',
                            text='Official ESP-SR Hi ESP fixture; transcript inferred from official test selection',
                            target='hi_esp',voice='official-test-fixture-unknown-provenance',samples=len(signal),
                            lead_samples=lead,tail_samples=16000,source_samples=len(pcm),
                            sha256=sha(path),pcm_sha256=hashlib.sha256(signal.tobytes()).hexdigest()))
    report=dict(role='Official fixture developer transport controls, not independent validation',revision=REVISION,
                source_sha256=sha(Path(__file__)),sources=sources,original_pcm_sha256=hashlib.sha256(raw).hexdigest(),
                records=records,independent_final_audio_accessed=False,
                limits='One official model test fixture, identity/recording provenance unknown and possibly used in model development. Two padding variants are correlated. Not a new speaker, recognition benchmark, phonetic endpoint or matched-FAR evidence. Original PCM unchanged; add0/1.8s lead and1s tail. Official test resets detection threshold; our replay uses the existing explicit0.63 setter for comparison and records getter.')
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
