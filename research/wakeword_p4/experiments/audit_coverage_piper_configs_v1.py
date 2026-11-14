"""Pinned source config metadata only; no model download, synthesis or role admission."""
import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
R=Path('research/wakeword_p4/results');OUT=Path('.local/wakeword/hey-coverage-configs-v1')
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 source=R/'4090-hey-coverage-source-cards-v1.json';catalog=R/'4090-hey-voice-coverage-catalog-v1.json';s=json.loads(source.read_text());c=json.loads(catalog.read_text());revision=c['revision'];assert s['revision']==revision;OUT.mkdir(exist_ok=False)
 def fetch(card):
  voice=card['voice'];entry=next(x for x in c['english_entries'] if x['key']==voice);rel=next(x for x in entry['files'] if x.endswith('.onnx.json'));url='https://huggingface.co/rhasspy/piper-voices/resolve/'+revision+'/'+rel
  try:
   b=urllib.request.urlopen(url,timeout=45).read();p=OUT/(voice+'.json');p.write_bytes(b);d=json.loads(b);tokens=d['phoneme_id_map'];assert isinstance(tokens,dict) and len(tokens)>0
   return dict(voice=voice,family=card['family'],url=url,path=str(p),bytes=len(b),sha256=sha(b),phoneme_type=d.get('phoneme_type'),espeak=d.get('espeak'),audio=d.get('audio'),piper_version=d.get('piper_version'),num_speakers=d.get('num_speakers'),num_symbols=d.get('num_symbols'),inference=d.get('inference'),token_map_sha256=sha(json.dumps(tokens,sort_keys=True,separators=(',',':')).encode()),token_keys=sorted(tokens),token_map= tokens,source_admitted=False)
  except Exception as e:return dict(voice=voice,url=url,error=repr(e),source_admitted=False)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(fetch,s['cards']))
 groups={}
 for r in rows:
  if 'error' not in r:groups.setdefault(r['token_map_sha256'],[]).append(r['voice'])
 out=dict(schema='hey-coverage-piper-config-metadata-v1',revision=revision,files={str(p):sha(p.read_bytes()) for p in [source,catalog,Path(__file__)]},configs=rows,exact_token_map_groups=groups,success=sum('error' not in r for r in rows),failed=sum('error' in r for r in rows),generation_authorized=False,limits='Pinned canonical model configs only; token equality does not establish physical voice independence or source admission, generated pronunciation, runtime compatibility or quality. No model downloads, audio or wake inference. Historical conditioning IDs are not transferred.')
 (R/'4090-hey-coverage-piper-config-metadata-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(success=out['success'],failed=out['failed'],token_map_groups=len(groups))))
if __name__=='__main__':main()
