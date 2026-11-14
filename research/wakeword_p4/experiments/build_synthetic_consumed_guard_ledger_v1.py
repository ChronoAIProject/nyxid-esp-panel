"""Conservative exclusion ledger; mentions do not prove actual consumption."""
import hashlib,json,re,subprocess
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 paths=sorted(p for p in R.iterdir() if p.suffix in ['.json','.md'] and 'synthetic-consumed-guard-ledger' not in p.name)
 evidence={};voices={};models=set()
 # Exclude all mentioned Piper family identities conservatively, including
 # reserved/metadata-only voices. Never infer unused from a missing match.
 pattern=re.compile(r'\ben_(?:US|GB)-[A-Za-z0-9_]+(?:-(?:low|medium|high|x_low))?\b')
 for p in paths:
  text=p.read_text();found=sorted(set(pattern.findall(text)))
  if found:
   evidence[str(p)]=sha(p)
   for v in found:voices.setdefault(v,[]).append(str(p))
 for p in sorted(Path('.local/wakeword').rglob('*.onnx')):
  if p.is_file():models.add(sha(p))
 output=dict(schema='synthetic-consumed-identities-v1',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),voice_ids=sorted(voices),model_sha256=sorted(models),physical_speaker_ids=[],voice_evidence=voices,evidence_sha256=evidence,coverage_limits='Conservative exclusion superset: all Piper voice mentions in current top-level tracked research JSON/MD and all accessible local ONNX bytes, including metadata-only/reserved/unused models. Mentions do not prove consumption. Does not cover deleted/external files, non-Piper voice aliases, latent physical-speaker/pretraining overlap, or unrecorded history. Unknown is not independent. Do not use this alone to authorize holdout.',generation_run=False,detector_invocations=0)
 dest=R/'4090-synthetic-consumed-guard-ledger-v1.json'
 with dest.open('x') as f:f.write(json.dumps(output,indent=2)+'\n')
 print(json.dumps(dict(voice_ids=len(voices),local_model_hashes=len(models),evidence_files=len(evidence),bytes=dest.stat().st_size,limits=output['coverage_limits'])))
if __name__=='__main__':main()
