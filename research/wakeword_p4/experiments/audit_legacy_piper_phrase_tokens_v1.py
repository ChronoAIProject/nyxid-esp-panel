"""Explicit legacy missing-type eSpeak diagnostic; not default runtime compatibility."""
import hashlib,importlib.metadata,json
from pathlib import Path
from piper_phonemize import phonemize_espeak
from audit_coverage_phrase_tokens_v1 import PHRASES
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 cp=R/'4090-hey-coverage-piper-config-metadata-v1.json';old=R/'4090-hey-coverage-phrase-token-feasibility-v1.json';configs=json.loads(cp.read_text());previous=json.loads(old.read_text());selected=set(previous['voices_requiring_adapter']);rows=[]
 for c in configs['configs']:
  if c['voice'] not in selected:continue
  p=Path(c['path']);assert sha(p)==c['sha256'];d=json.loads(p.read_text());assert 'phoneme_type' not in d and d['piper_version']=='0.2.0' and not d.get('phoneme_map');voice=d['espeak']['voice'];mapping=d['phoneme_id_map']
  for text in PHRASES:
   ph=phonemize_espeak(text,voice);assert len(ph)==1;missing=sorted(set(ph[0])-set(mapping));ids=None
   if not missing:
    ids=list(mapping['^'])+list(mapping['_'])
    for symbol in ph[0]:ids+=list(mapping[symbol])+list(mapping['_'])
    ids+=list(mapping['$'])
   rows.append(dict(voice=c['voice'],text=text,explicit_diagnostic_espeak_voice=voice,phonemes=ph[0],missing_symbols=missing,token_ids=ids,strict_encodable=not missing))
 out=dict(schema='legacy-piper-explicit-espeak-token-diagnostic-v1',files={str(p):sha(p) for p in [cp,old,Path(__file__),Path('.local/wakeword/piper/generate_samples.py')]},phonemizer_version=importlib.metadata.version('piper-phonemize'),rows=rows,all_strict_encodable=all(r['strict_encodable'] for r in rows),correction='Three legacy configs omit phoneme_type; they do not declare a different phoneme type. Original fail-closed v1 policy result preserved.',limits='Explicit eSpeak diagnostic using existing config voice; does not establish modern Piper default handling, actual ONNX signature/duration support, generated quality, source independence or admission. Existing generator skips missing tokens, while this diagnostic requires complete coverage. No generation/model inference.')
 (R/'4090-legacy-piper-explicit-espeak-token-diagnostic-v1.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(json.dumps(dict(rows=len(rows),all_strict_encodable=out['all_strict_encodable'])))
if __name__=='__main__':main()
