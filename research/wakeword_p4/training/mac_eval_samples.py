"""Generate local-only evaluation audio from a separate, untrained TTS engine."""
import argparse
import hashlib
import json
from math import gcd
from pathlib import Path
import subprocess

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    installed=subprocess.check_output(['say','-v','?'],text=True)
    voices=[v for v in ['Samantha','Daniel','Karen','Moira','Tessa','Serena'] if any(line.startswith(v+' ') for line in installed.splitlines())]
    positives=['Hey Nicks.','Hey, Nicks.','Hey Nicks!','Hey Nicks?','Hey Nix.']
    negatives=['Hey Nick.','Hey Next.','Hey Mix.','Hey Six.','Hey Max.','Hey Nice.','Hey Nexus.',
               'Hey Alex.','Hi ESP.','Hey Siri.','The next thing is ready.','I mixed the ingredients.',
               'Please check the volume.','Hello, can you hear me?','How can I help you?',
               'Thanks, that is all.','The television is playing.','I will be back in six minutes.',
               'He connects the next two cables.','I heard Nick speaking.']
    records=[]
    for voice in voices:
        for label,texts in [(1,positives),(0,negatives)]:
            for index,text in enumerate(texts):
                stem=f'{voice}-{label}-{index:02d}'
                aiff=args.output/(stem+'.aiff')
                subprocess.run(['say','-v',voice,'-o',str(aiff),text],check=True,timeout=30)
                samples,rate=sf.read(aiff,dtype='float32')
                divisor=gcd(rate,16000)
                samples=resample_poly(samples,16000//divisor,rate//divisor)
                path=args.output/(stem+'.wav')
                sf.write(path,samples,16000,subtype='PCM_16')
                records.append({'path':path.name,'voice':voice,'label':label,'text':text,
                                'samples':len(samples),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        print(json.dumps({'voice':voice,'total':len(records)}),flush=True)
    (args.output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    print(json.dumps({'voices':voices,'clips':len(records),'scope':'Independent TTS engine holdout, not real human keyword recall. Local evaluation audio only.'}),flush=True)


if __name__=='__main__':main()
