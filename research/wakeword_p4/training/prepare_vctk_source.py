"""Pin a previously unused Piper VCTK synthesis checkpoint and reserve voice IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import urllib.request

REVISION='ae9d8352b73c1b1750748843e753a504b00e05b0'
MODEL_SHA='4e9fc85ab9009385319fc6bae7f55577f8a2d7ee77fd9159a5500eb6531f41e6'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    base=f'https://huggingface.co/rhasspy/piper-voices/resolve/{REVISION}/en/en_GB/vctk/medium/'
    files={}
    for name in ['MODEL_CARD','en_GB-vctk-medium.onnx.json','en_GB-vctk-medium.onnx']:
        dest=a.output/name
        with urllib.request.urlopen(base+name,timeout=60) as response,dest.open('wb') as out:
            while chunk:=response.read(1024*1024): out.write(chunk)
        files[name]=dict(url=base+name,size_bytes=dest.stat().st_size,sha256=sha(dest))
        print(json.dumps(dict(downloaded=name,**files[name])),flush=True)
    assert files['en_GB-vctk-medium.onnx']['size_bytes']==76952753
    assert files['en_GB-vctk-medium.onnx']['sha256']==MODEL_SHA
    config=json.loads((a.output/'en_GB-vctk-medium.onnx.json').read_text())
    assert config['num_speakers']==109 and len(config['speaker_id_map'])==109
    voices=sorted(config['speaker_id_map']);random.Random(88726).shuffle(voices)
    parts={'development':voices[:20],'reserved_final':voices[20:]}
    assert not set(parts['development'])&set(parts['reserved_final'])
    report=dict(repository='rhasspy/piper-voices',revision=REVISION,files=files,
        speaker_id_map=config['speaker_id_map'],partitions=parts,partition_seed=88726,
        script_sha256=sha(Path(__file__)),audio_generated=False,wake_models_scored=False,
        role='New synthesis source and voice reservation only; no training use',
        limits='Different weights/data from project LibriTTS Piper and Kokoro, same Piper architecture. Model card reports VCTK dataset and Lessac medium finetuning. Preset IDs do not independently certify physical speaker identity or disjoint pretraining; not a human-generalization benchmark. Reserved final voices must not be generated/scored until an explicit frozen protocol.')
    (a.output/'source.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'source_sha256':sha(a.output/'source.json'),'development':20,'reserved_final':89}),flush=True)


if __name__=='__main__': main()
