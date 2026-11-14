"""Conservative LibriTTS reader-name overlap from pinned configs, no role assignment."""
import hashlib,json,re
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 source=R/'4090-hey-coverage-piper-config-metadata-v1.json';d=json.loads(source.read_text());rows={r['voice']:r for r in d['configs']};names=['en_US-libritts-high','en_US-libritts_r-medium'];maps=[];files={str(source):sha(source),str(Path(__file__)):sha(__file__)}
 for name in names:
  r=rows[name];path=Path(r['path']);assert sha(path)==r['sha256'];files[str(path)]=sha(path);c=json.loads(path.read_text());m=c['speaker_id_map'];assert len(m)==c['num_speakers']==904;normalized={}
  for k,v in m.items():
   match=re.fullmatch('p?([0-9]+)',k);assert match
   key=match[1];assert key not in normalized;normalized[key]=dict(original_key=k,conditioning_id=v)
  maps.append(normalized)
 assert set(maps[0])==set(maps[1]);pairs=[dict(reader_id=k,left=maps[0][k],right=maps[1][k]) for k in sorted(maps[0],key=int)]
 out=dict(schema='libritts-config-reader-alias-overlap-v1',files=files,families=names,speakers_each=904,normalized_reader_ids_exact=True,same_numeric_conditioning_ids=sum(x['left']['conditioning_id']==x['right']['conditioning_id'] for x in pairs),pairs=pairs,policy='Conservatively treat matching numeric LibriTTS reader names as one source-reader group across these two families. Never split these aliases into training and heldout or count as independent voices.',limits='Configuration-label overlap, not acoustic speaker verification or proof of shared weights. Does not certify mapping to the historical672conditioning sampler. No source admission, role reservation, model download, synthesis or inference.')
 (R/'4090-libritts-config-reader-alias-overlap-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['speakers_each','normalized_reader_ids_exact','same_numeric_conditioning_ids']}))
if __name__=='__main__':main()
