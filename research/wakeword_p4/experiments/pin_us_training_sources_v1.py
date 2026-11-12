"""Fetch immutable training-only Piper presets and record original artifacts."""
import hashlib,json,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
revision='ae9d8352b73c1b1750748843e753a504b00e05b0'
root=Path('.local/wakeword/piper-us-training-source-v1');root.mkdir(exist_ok=False)
voices=['lessac','ryan','ljspeech']
def fetch(item):
 voice,name=item;url=f'https://huggingface.co/rhasspy/piper-voices/resolve/{revision}/en/en_US/{voice}/medium/{name}'
 data=urllib.request.urlopen(url,timeout=120).read();p=root/voice/name;p.parent.mkdir(exist_ok=True);p.write_bytes(data)
 return str(p.relative_to(root)),dict(url=url,size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
items=[(v,n) for v in voices for n in ['MODEL_CARD',f'en_US-{v}-medium.onnx.json',f'en_US-{v}-medium.onnx']]
with ThreadPoolExecutor(max_workers=3) as pool:files=dict(pool.map(fetch,items))
r=dict(repository='rhasspy/piper-voices',revision=revision,voices=voices,role='training-only coverage pilot; never development/final',files=files,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(root/'source.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
