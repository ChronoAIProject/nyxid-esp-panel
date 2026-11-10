"""Prepare detector-independent MFA diagnostics from existing development audio."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import soundfile as sf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('paired-manifest','official-manifest','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    paired=json.loads(a.paired_manifest.read_text());official=json.loads(a.official_manifest.read_text())
    if not paired.get('paired_wakephrases') or paired['independent_final_audio_accessed'] or official['independent_final_audio_accessed']:
        raise ValueError('Only existing development/control fixtures')
    records=[]
    for manifest,root,kind in [(paired,a.paired_manifest.parent,'paired'),(official,a.official_manifest.parent,'official')]:
        for row in manifest['records']:
            if row['split']!='validation' or row['role'] not in ('development-paired-validation','development-transport-probe'):
                raise ValueError('Only development roles')
            if row['target'] not in ('hey_nyx','hi_esp'):continue
            if kind=='paired' and row['role']!='development-paired-validation':raise ValueError('Wrong paired role')
            if kind=='official' and (row['role']!='development-transport-probe' or not official.get('sources')):
                raise ValueError('Need official-fixture provenance')
            path=(root/row['path']).resolve()
            if not path.is_relative_to(root.resolve()) or sha(path)!=row['sha256']:raise ValueError('Changed/escaped waveform')
            pcm,rate=sf.read(path,dtype='int16')
            if rate!=16000 or pcm.ndim!=1 or len(pcm)!=row['samples'] or hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()!=row['pcm_sha256']:
                raise ValueError('Changed PCM')
            text='HEY NICKS' if row['target']=='hey_nyx' else 'HI E S P'
            voice=row['voice'] if kind=='paired' else 'official_unknown'
            relative=Path(voice)/(kind+'_'+Path(row['path']).name)
            records.append(dict(path=str(relative),source_path=row['path'],source_kind=kind,target=row['target'],
                                sha256=row['sha256'],pcm_sha256=row['pcm_sha256'],samples=len(pcm),transcript=text,
                                transcript_source='Synthesis intent' if kind=='paired' else 'Official test fixture wake-word identity',
                                original_wav=str(path)))
    if len(records)!=26 or len({r['path'] for r in records})!=26:
        raise ValueError('Expected24 paired targets and2 official padding controls')
    a.output.mkdir(parents=True)
    for row in records:
        path=a.output/'corpus'/row['path'];path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(row['original_wav'],path);path.with_suffix('.lab').write_text(row['transcript']+'\n')
    report=dict(role='Detector-independent automatic alignment development inputs; not phonetic truth',
                source_sha256=sha(Path(__file__)),paired_manifest_sha256=sha(a.paired_manifest),official_manifest_sha256=sha(a.official_manifest),
                records=[{k:v for k,v in r.items() if k!='original_wav'} for r in records],
                intended_method='Montreal Forced Aligner pretrained english_us_arpa acoustic model and dictionary; SQLite; no wake scores supplied; preserve word/phone tiers and alignment diagnostics.',
                independent_final_audio_accessed=False,wake_scores_accessed=False,records_filtered_using_asr=False,
                existing_windows_or_labels_changed=False,
                limits='English intended transcript forced onto possibly imperfect TTS/official fixture audio. Alignment errors and wrong pronunciations remain possible. Outputs are diagnostic, need independent quality checking before any latency ground-truth claim. Four reused TTS presets plus one unknown official fixture; no new independent evaluation people.')
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'}))


if __name__=='__main__':main()
