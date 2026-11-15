"""Bind training-only raw sources for a distinct pretrained-window hypothesis."""
import hashlib
import json
from pathlib import Path
import re
import soundfile as sf
R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword')
OUT=R/'4090-matchboxnet-training-material-plan-v1.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 synthetic=BASE/'speecht5-training-rendered-v1/records.jsonl'
 audit=R/'4090-speecht5-training-render-audit-v1.json';assert json.loads(audit.read_text())['all_passed']
 cache=BASE/'libri-train-clean-100-features-v1/cache.json';provenance=json.loads(cache.read_text())
 assert provenance['speaker_disjoint'] and provenance['independent_final_audio_accessed'] is False
 corpus=BASE/'libri-train-clean-100-source-v1/LibriSpeech/train-clean-100'
 speakers=sorted(int(p.name) for p in corpus.iterdir() if p.is_dir());assert speakers==provenance['speakers'] and len(speakers)==251
 classes={'background':0,'nyx':1,'official':2,'nick':3,'next':4,'mix':5,'muse':6}
 records=[]
 for row in map(json.loads,synthetic.read_text().splitlines()):
  if row['carrier']!='bare':continue
  assert row['role']=='training' and row['passed'] and row['phrase_id'] in classes
  path=Path(row['path']);assert sha(path)==row['sha256'];info=sf.info(path)
  assert info.samplerate==16000 and info.channels==1 and info.frames==row['samples'] and info.subtype=='PCM_16'
  full_samples=row['transformed_source_samples']+row['rir_tail_samples']
  for after in [256,512,1024]:
   start=row['lead_samples']+full_samples+after-20480;end=start+20480
   assert 0<=start<=row['lead_samples'] and end>=row['lead_samples']+full_samples and end<=row['samples']
  records.append(dict(kind='synthetic-full-bare',role='training',class_id=classes[row['phrase_id']],phrase_id=row['phrase_id'],voice_id=row['voice_id'],path=str(path),sha256=sha(path),pcm_sha256=row['pcm_sha256'],samples=row['samples'],source_lead=row['lead_samples'],complete_source_and_rir_samples=full_samples,after_source_offsets=[256,512,1024],source_id=row['source_request_id']))
 assert len(records)==378 and all(sum(r['class_id']==c for r in records)==63 for c in range(1,7))
 # Deterministic source choice before any model call: first four sorted original
 # utterances per all251already-audited training speakers, no held-out files.
 selected_transcripts={}
 for speaker in speakers:
  paths=sorted((corpus/str(speaker)).rglob('*.flac'))[:4];assert len(paths)==4
  for path in paths:
   transcript=path.parent/(path.parent.parent.name+'-'+path.parent.name+'.trans.txt')
   if transcript not in selected_transcripts:
    relative=str(transcript.relative_to(corpus));bound=next(r for r in provenance['transcripts'] if r['path']==relative);assert sha(transcript)==bound['sha256']
    lines={line.split(maxsplit=1)[0]:line.split(maxsplit=1)[1] for line in transcript.read_text().splitlines()}
    selected_transcripts[transcript]=lines
   text=selected_transcripts[transcript][path.stem]
   assert not re.search(r'\bhey\s+(?:nyx|nix|nicks|knicks|nick\s+s)\b',text,re.I)
   info=sf.info(path);assert info.samplerate==16000 and info.channels==1 and info.frames>0
   records.append(dict(kind='libri-training-background',role='training',class_id=0,path=str(path),sha256=sha(path),samples=info.frames,speaker_id=speaker,utterance_id=path.stem,transcript=text))
 assert len(records)==1382
 files=[Path(__file__),synthetic,audit,cache,R/'4090-speecht5-training-native-quality-v1.json',R/'4090-matchboxnet-source-semantics-audit-v1.json',R/'4090-matchboxnet-gpu-function-v1.json',Path('research/wakeword_p4/training/matchboxnet_pretrained_v1.py')]+list(selected_transcripts)
 plan=dict(schema='matchboxnet-training-material-plan-v1',hypothesis='Test a different pretrained acoustic representation/full-window task package after small source-coverage and same-encoder lexical-loss interventions failed. Not an isolated pretraining causal experiment: frontend, encoder, window readout and training task jointly differ. No claim that this adapter solves lexical discrimination.',classes=classes,records=records,files={str(p):sha(p) for p in files},synthetic_bare_presentations=378,synthetic_complete_window_positions_per_presentation=3,training_background_original_utterances=1004,training_background_speakers=251,window_pcm_samples=20480,window_mfcc_frames=128,full_bare_source_and_rir_preserved=True,metadata_audio_format_checks=True,neural_calls=0,training_executed=False,acceptance_accessed=False,promotion_qualified=False,policy='Only previously recorded training sources. All Melo models/voices, reserved acceptance, protectedMike and three historicalfinal audio excluded. Bare positive/nearword sources fit all three complete-window placements; no speech trimming or invented command keyword end. Libri source identities inherited from earlier checksum/role audit and transcript hashes revalidated; text screening is not phonetic proof. Upstream pretrained physical-speaker overlap unknown. Manifest only: source-matched ambient/silence generation, sampler, end-to-end finite-window cadence, loss/optimizer/budget, representative data, one INT8 export and diagnostic protocol must be separately locked before training. This model cannot enter the closed two-candidate Melo v2 selection; any future promotion requires fresh unconsumed development and independent acceptance.')
 with OUT.open('x') as f:f.write(json.dumps(plan,indent=2)+'\n')
 print(json.dumps(dict(records=len(records),synthetic=378,background=1004,sha256=sha(OUT))),flush=True)
if __name__=='__main__':main()
