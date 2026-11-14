"""Strict text-to-phoneme/token feasibility only, no acoustic model inference."""
import hashlib,importlib.metadata,json
from pathlib import Path
from piper_phonemize import phonemize_espeak
R=Path('research/wakeword_p4/results')
PHRASES=['Hey Willow.','Hey Nicks.','Hey Nick.','Hey Next.','Hey Mix.','Hey Muse.','Hi ESP.']
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 source=R/'4090-hey-coverage-piper-config-metadata-v1.json';p=json.loads(source.read_text());rows=[]
 for r in p['configs']:
  path=Path(r['path']);assert sha(path.read_bytes())==r['sha256'];d=json.loads(path.read_text());mapping=d['phoneme_id_map'];voice=d.get('espeak',{}).get('voice');custom=d.get('phoneme_map',{})
  for text in PHRASES:
   if d.get('phoneme_type')!='espeak' or not voice:
    rows.append(dict(voice=r['voice'],text=text,passed=False,reason='unsupported phoneme type'));continue
   phones=phonemize_espeak(text,voice);assert len(phones)==1
   # Diagnose direct old strict encoder; do not invent per-model remapping.
   missing=sorted(set(phones[0])-set(mapping));controls=[x for x in ['^','_','$'] if x not in mapping];tokens=None
   if not missing and not controls:
    tokens=list(mapping['^'])+list(mapping['_'])
    for ph in phones[0]:tokens+=list(mapping[ph])+list(mapping['_'])
    tokens+=list(mapping['$'])
   passed=not missing and not controls and not custom
   rows.append(dict(voice=r['voice'],espeak_voice=voice,text=text,phonemes=phones[0],missing_symbols=missing,missing_controls=controls,config_phoneme_map=custom,token_ids=tokens,passed=passed,reason='direct strict encoder compatible' if passed else 'requires actual config-specific phonemization/remapping validation'))
 groups={}
 for r in rows:groups[r['voice']]=groups.get(r['voice'],True) and r['passed']
 out=dict(schema='hey-coverage-phrase-token-feasibility-v1',source_sha256=sha(source.read_bytes()),script_sha256=sha(Path(__file__).read_bytes()),phonemizer_version=importlib.metadata.version('piper-phonemize'),phrases=PHRASES,rows=rows,voices_all_phrases_direct_compatible=[k for k,v in groups.items() if v],voices_requiring_adapter=[k for k,v in groups.items() if not v],generation_authorized=False,limits='Current piper-phonemize1.1.0 direct strict encoder only; configs requiring newer Piper/custom remapping are not automatically compatible. Phonemes are generator outputs, not acoustic truth. No ONNX model/audio generation, role admission, speaker isolation claim, human listening or wake scoring.')
 (R/'4090-hey-coverage-phrase-token-feasibility-v1.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(json.dumps(dict(rows=len(rows),compatible=len(out['voices_all_phrases_direct_compatible']),requires_adapter=out['voices_requiring_adapter'])))
if __name__=='__main__':main()
