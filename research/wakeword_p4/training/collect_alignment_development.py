"""Collect automatic development alignments without consulting wake events."""
import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from praatio import textgrid


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('inputs','alignment','models','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    manifest=json.loads((a.inputs/'manifest.json').read_text())
    if manifest['independent_final_audio_accessed'] or manifest['wake_scores_accessed']:
        raise ValueError('Only detector-independent development inputs')
    downloads=json.loads((a.models/'download.json').read_text())
    for record in downloads:
        name='english_us_arpa.zip' if record['kind']=='acoustic' else 'english_us_arpa.dict'
        if sha(a.models/name)!=record['sha256']:raise ValueError('Changed alignment model/dictionary')
    analysis_path=a.alignment/'alignment_analysis.csv'
    with analysis_path.open(newline='') as stream:
        quality={(r['speaker'],r['file']):r for r in csv.DictReader(stream)}
    records=[];expected=set()
    for row in manifest['records']:
        source=a.inputs/'corpus'/row['path'];relative=Path(row['path']).with_suffix('.TextGrid');path=a.alignment/relative
        expected.add(path.resolve())
        if sha(source)!=row['sha256']:raise ValueError('Changed waveform')
        tg=textgrid.openTextgrid(str(path),includeEmptyIntervals=True)
        if set(tg.tierNames)!={'words','phones'}:raise ValueError('Unexpected alignment tiers')
        tiers={name:[dict(start=float(e.start),end=float(e.end),label=e.label) for e in tg.getTier(name).entries] for name in tg.tierNames}
        duration=row['samples']/16000
        if any(e['start']<0 or e['end']<e['start'] or e['end']>duration+.001 for tier in tiers.values() for e in tier):
            raise ValueError('Invalid alignment bounds')
        words=[e for e in tiers['words'] if e['label']]
        phones=[e for e in tiers['phones'] if e['label'] not in ('','sil','sp','spn','<eps>')]
        if not words or not phones:raise ValueError('Missing speech alignment')
        key=(relative.parent.name,relative.stem)
        if key not in quality:raise ValueError('Missing alignment quality row')
        records.append(dict(source=row,textgrid_sha256=sha(path),tiers=tiers,quality=quality[key],
                            expected_words_preserved=[w['label'] for w in words]==row['transcript'].lower().split(),
                            automatic_word_start_seconds=words[0]['start'],automatic_word_end_seconds=words[-1]['end'],
                            automatic_last_phone=phones[-1],alignment_accepted_as_phonetic_ground_truth=False))
    actual={p.resolve() for p in a.alignment.rglob('*.TextGrid')}
    if actual!=expected:raise ValueError('Incomplete/extra aligned files')
    report=dict(role='Detector-independent automatic MFA development alignments, not phonetic ground truth',
                source_sha256=sha(Path(__file__)),input_manifest_sha256=sha(a.inputs/'manifest.json'),
                analysis_sha256=sha(analysis_path),mfa_version=version('montreal-forced-aligner'),model_sources=downloads,
                records=records,files=len(records),all_expected_words_preserved=all(r['expected_words_preserved'] for r in records),
                wake_scores_accessed=False,independent_final_audio_accessed=False,old_windows_or_labels_changed=False,
                command_policy='MFA align with english_us_arpa3.0.0 acoustic/dictionary;single_speaker disables speaker adaptation;SQLite;4jobs. No transcript correction or ASR filtering.',
                limits='Intended text forced onto imperfect TTS/unknown official-fixture speech. Likelihood/duration diagnostics are uncalibrated and not proof of pronunciation. No manual phonetic audit, new recall/latency score, altered success window or final acceptance. Preserve every word/phone interval and all quality rows.')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(files=len(records),all_expected_words_preserved=report['all_expected_words_preserved'],wake_scores_accessed=False)))


if __name__=='__main__':main()
