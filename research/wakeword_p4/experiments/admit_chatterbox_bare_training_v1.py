"""Provisional training labels from independent ASR plus mechanical alignment."""
import hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    generation=R/'4090-chatterbox-bare-training-generation-original-v1.json';asr=R/'4090-chatterbox-bare-training-asr-details-original-v1.jsonl';alignment=R/'4090-chatterbox-bare-training-prefix-alignment-original-v1.json'
    g=json.loads(generation.read_text());a=[json.loads(s) for s in asr.read_text().splitlines()];f=json.loads(alignment.read_text())
    assert g['actual_exit_code']==0 and f['execution']['exit_code']==0 and len(g['records'])==len(a)==len(f['records'])==48
    admitted=[];quarantined=[]
    for row,trans,aligned in zip(g['records'],a,f['records'],strict=True):
        assert row['request']['role']=='training-only' and row['request']['requested_label']==1
        wav,=[p for p in row['files'] if p['path'].endswith('.wav')]
        assert Path(wav['path']).name==trans['path'] and wav['sha256']==trans['sha256']==aligned['source']['sha256']==sha(wav['path'])
        assert Path(wav['path']).stat().st_size==wav['bytes']
        assert aligned['source']['id']==f"{row['index']:03d}"
        usable=bool(trans['text_agreement'] and aligned['mechanically_usable'] and aligned['automatic_prefix_end_seconds'] is not None)
        record=dict(index=row['index'],request=row['request'],wav=wav,asr=trans,automatic_prefix_end_seconds=aligned['automatic_prefix_end_seconds'],alignment_sha256=aligned['textgrid_sha256'],provisional_label=True)
        if usable:admitted.append(record)
        else:quarantined.append(dict(**record,reasons=[reason for reason,test in [('independent_transcript_disagreement',trans['text_agreement']),('alignment_not_mechanically_usable',aligned['mechanically_usable'])] if not test]))
    report=dict(schema='chatterbox-bare-training-admission-v1',source_files={str(p):sha(p) for p in [Path(__file__),generation,asr,alignment]},admitted=admitted,quarantined=quarantined,policy='Training-only provisional positive labels. Existing auditor fixed target aliases, no newly invented ASR aliases, wake scoring, score-based filtering or retries. All transcript disagreements quarantined unchanged; no ASR mistake claim. Forced HEY NICKS supplied transcript yields mechanical endpoint only, never independent phonetic truth. Text agreement+alignment is not guaranteed correct pronunciation; no human/generalization qualification. Full original PCM retained, no developer/acceptance conditioning. No model training by admission alone.')
    with (R/'4090-chatterbox-bare-training-admission-v1.json').open('x') as out:out.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(admitted=len(admitted),quarantined=len(quarantined))))
if __name__=='__main__':main()
