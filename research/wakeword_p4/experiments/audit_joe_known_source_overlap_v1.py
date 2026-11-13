"""Audit actual training originals and immutable historical hash metadata; no scores."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess

import soundfile as sf

ROOT=Path('research/wakeword_p4/results')
BASE=Path('.local/wakeword/features-gpu-ami-extra-v1')
OUT=Path('.local/wakeword/joe-known-source-overlap-v1')
HISTORICAL='43254c0'
HEX=re.compile(rb'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def targets():
    rows=[]
    source=Path('.local/wakeword/joe-development-native-transport-v1')
    for r in map(json.loads,(source/'records.jsonl').read_text().splitlines()):
        path=source/r['path'];assert sha(path)==r['sha256'] and r['audit']['passed']
        pcm,rate=sf.read(path,dtype='int16');assert rate==16000 and pcm.ndim==1
        rows.append(dict(id=r['id'],role='Joe-native-development',encoded=r['sha256'],
                         pcm=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()))
    for r in json.loads(Path('.local/wakeword/prospective-background-whole-waveform-v1/manifest.json').read_text())['records']:
        rows.append(dict(id=r.get('id',r.get('recording_id',r.get('path'))),
                         role='new-background-development',encoded=r['sha256'],pcm=r['pcm_sha256']))
    assert len(rows)==162
    # Exact duplicates among requested source objects would disqualify a pack.
    assert len({r['encoded'] for r in rows})==162 and len({r['pcm'] for r in rows})==162
    return rows


def main():
    OUT.mkdir(exist_ok=False)
    new=targets();wanted={r[k] for r in new for k in ('encoded','pcm')}
    # Scan original committed evidence, including final hashes, never final audio.
    commit=subprocess.check_output(['git','rev-parse',HISTORICAL],text=True).strip()
    entries=subprocess.check_output(['git','ls-tree','-rl',commit,'research/wakeword_p4/results'],text=True).splitlines()
    historical=[];known=set();matches=[]
    with subprocess.Popen(['git','cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE) as proc:
        for entry in entries:
            head,path=entry.split('\t',1)
            if not path.endswith(('.json','.jsonl')):continue
            _,kind,oid,size=head.split();assert kind=='blob'
            proc.stdin.write((oid+'\n').encode());proc.stdin.flush()
            header=proc.stdout.readline().split();assert header[1]==b'blob'
            data=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n'
            hashes={v.decode() for v in HEX.findall(data)};known.update(hashes)
            overlap=sorted(hashes&wanted)
            historical.append(dict(path=path,git_blob=oid,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),hash_count=len(hashes)))
            if overlap:matches.append(dict(path=path,hashes=overlap))
        proc.stdin.close();assert proc.wait()==0
    (OUT/'historical-file-inventory.json').write_text(json.dumps(historical,indent=2)+'\n')
    print(json.dumps(dict(historical_files=len(historical),unique_hash_values=len(known),new_hash_matches=matches)),flush=True)
    training_path=Path('.local/wakeword/model-per-draw-noise-fixed-formal-v1/training.json')
    training=json.loads(training_path.read_text())
    assert sha(BASE/'features.json')==training['features_index_sha256']
    feature_rows=json.loads((BASE/'features.json').read_text())
    originals=sorted({(r['path'],r['sha256']) for r in feature_rows if r['split']=='training'})
    def decode(item):
        name,digest=item;path=BASE/'dataset'/name
        assert sha(path)==digest,name
        pcm,rate=sf.read(path,dtype='int16');info=sf.info(path)
        assert rate==16000 and pcm.ndim==1 and info.subtype=='PCM_16',name
        pcm_sha=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()
        return dict(path=name,encoded_sha256=digest,pcm_sha256=pcm_sha,
                    samples=len(pcm),overlap=digest in wanted or pcm_sha in wanted)
    details=[]
    with (OUT/'actual-main-training-originals.jsonl').open('x') as log,ThreadPoolExecutor(2) as pool:
        for r in pool.map(decode,originals):
            details.append(r);log.write(json.dumps(r)+'\n')
            if len(details)%1000==0:
                log.flush();print(json.dumps(dict(training_originals_checked=len(details),planned=len(originals))),flush=True)
    # Use authenticated original current pack reports for additional source PCM.
    pins_path=ROOT/'4090-complete-recovery-execution-pins-v1.json'
    pins=json.loads(pins_path.read_text())['file_sha256']
    packs=[]
    for name in ['british','piper-balanced','us','factorized']:
        p=Path(f'.local/wakeword/{name}-training-features-v1/report.json')
        assert sha(p)==pins[str(p)]
        hashes={v.decode() for v in HEX.findall(p.read_bytes())};known.update(hashes)
        packs.append(dict(path=str(p),sha256=sha(p),recorded_hash_values=len(hashes),overlap=sorted(wanted&hashes)))
    # Reuse precise human/ambient/legacy source ledgers, with their full caveats.
    ledgers=[]
    for name in ['4090-new-background-training-overlap-v1.json','4090-legacy-training-human-pcm-audit-v1.json']:
        p=ROOT/name;hashes={v.decode() for v in HEX.findall(p.read_bytes())}
        # Those background audit reports intentionally include the new backgrounds!
        # They are evidence references, NOT treated as historical source conflicts.
        ledgers.append(dict(path=str(p),sha256=sha(p)))
    report=dict(schema='joe-known-source-overlap-audit-v1',script_sha256=sha(__file__),
        historical_commit=commit,historical_files=len(historical),
        historical_bytes=sum(r['bytes'] for r in historical),historical_hash_matches=matches,
        actual_training_json_sha256=sha(training_path),features_index_sha256=sha(BASE/'features.json'),
        actual_main_training_originals=len(details),actual_main_training_pcm_overlaps=[r for r in details if r['overlap']],
        additional_pack_reports=packs,new_source_objects=new,known_source_overlap_found=bool(matches or any(r['overlap'] for r in details) or any(r['overlap'] for r in packs)),
        human_ambient_audit_references=ledgers,
        inventory_sha256=sha(OUT/'historical-file-inventory.json'),training_originals_sha256=sha(OUT/'actual-main-training-originals.jsonl'),
        final_audio_opened=False,acceptance_audio_opened=False,models_scored=False,
        all_possible_source_isolation_proven=False,scoring_authorized=False,
        limits='Exact original container/PCM and recorded hash comparisons only. Historical metadata scan is broad hash evidence, not semantic provenance certification; unrecorded PCM, transformed/partial/physical-person/pretraining overlap not excluded. Additional pack source PCM checked when recorded; pack feature arrays are not inverted. Shared Joe/Mike rdh parent remains disclosed. Existing human/ambient audits retain original scope and caveats.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['new_source_objects','additional_pack_reports']}),flush=True)
    if report['known_source_overlap_found']:raise SystemExit(1)


if __name__=='__main__':main()
